from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request

from .database import get_supabase
from .auth import _parse_session, _SESSION_COOKIE

from utils.workspaces import (
    WorkspacePaths,
    ensure_workspace,
    create_workspace,        # random (legacy)
    copy_base_to_workspace,
    delete_workspace,
)
from utils.simple_pipeline import add_upload_to_workspace
from utils.index_uploads_only import build_index_from_uploads_only

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workspaces", tags=["workspaces"])

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "supersecret")
LOCK_TTL_SEC = int(os.getenv("WS_BUILD_LOCK_TTL_SEC", "900"))  # 15 min


# ---------- helpers ----------
def _mk_dirs(p: WorkspacePaths) -> None:
    p.root.mkdir(parents=True, exist_ok=True)
    p.uploads.mkdir(parents=True, exist_ok=True)
    p.processed.mkdir(parents=True, exist_ok=True)
    p.index.mkdir(parents=True, exist_ok=True)


def _index_files(p: WorkspacePaths) -> List[Path]:
    return [p.faiss_index, p.idmap, p.metadata]


def _mtime_or_zero(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _latest_upload_mtime(p: WorkspacePaths) -> float:
    latest = 0.0
    if p.uploads.exists():
        for f in p.uploads.rglob("*"):
            if f.is_file():
                latest = max(latest, _mtime_or_zero(f))
    for name in (p.chunks, p.embeddings):
        latest = max(latest, _mtime_or_zero(name))
    return latest


def _index_mtime(p: WorkspacePaths) -> float:
    return max((_mtime_or_zero(f) for f in _index_files(p)), default=0.0)


def _lock_dir(p: WorkspacePaths) -> Path:
    return p.root / ".lock-build"


def _acquire_lock(p: WorkspacePaths) -> None:
    lock = _lock_dir(p)
    try:
        lock.mkdir(exist_ok=False)
        (lock / "started_at").write_text(str(time.time()), encoding="utf-8")
    except FileExistsError:
        try:
            started = float((lock / "started_at").read_text(encoding="utf-8"))
        except Exception:
            started = 0.0
        if time.time() - started > LOCK_TTL_SEC:
            _release_lock(p)
            lock.mkdir(exist_ok=False)
            (lock / "started_at").write_text(str(time.time()), encoding="utf-8")
        else:
            raise HTTPException(
                status_code=409,
                detail={"step": "lock",
                        "error": "build already in progress", "ws_id": p.ws_id},
            )


def _release_lock(p: WorkspacePaths) -> None:
    lock = _lock_dir(p)
    if lock.exists():
        for child in lock.glob("*"):
            try:
                child.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            lock.rmdir()
        except Exception:
            pass


# ---------- routes ----------
@router.post("", summary="Create or reuse workspace bound to session_id")
def create_ws(
    session_id: Optional[str] = Query(
        None, description="If provided, use it as ws_id"),
) -> Dict[str, Any]:
    """
    If session_id is provided: ws_id=session_id (idempotent).
    Else: legacy random workspace.
    """
    if session_id:
        p = WorkspacePaths.from_id(session_id)
        _mk_dirs(p)
        return {
            "ws_id": p.ws_id,
            "paths": {
                "root": str(p.root),
                "uploads": str(p.uploads),
                "processed": str(p.processed),
                "index": str(p.index),
            },
            "bound_to_session": session_id,
        }

    p = create_workspace()
    _mk_dirs(p)
    return {
        "ws_id": p.ws_id,
        "paths": {
            "root": str(p.root),
            "uploads": str(p.uploads),
            "processed": str(p.processed),
            "index": str(p.index),
        },
    }


@router.post("/{ws_id}/documents", summary="Upload one or many documents")
def upload_docs(
    ws_id: str,
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    auto_build: bool = Query(True, description="Rebuild FAISS after upload"),
):
    p = WorkspacePaths.from_id(ws_id)
    _mk_dirs(p)
    received = []

    # (optional) DB tracking
    try:
        sb = get_supabase()
    except Exception:
        sb = None

    def _store_one(up: UploadFile, key: str):
        path = add_upload_to_workspace(p, up)
        received.append(
            {"key": key, "name": up.filename, "stored_as": str(path)})
        if sb:
            try:
                size_bytes = None
                try:
                    size_bytes = Path(path).stat().st_size
                except Exception:
                    pass
                sb.table("documents").insert({
                    "ws_id": ws_id,
                    "filename": up.filename,
                    "path": str(path),
                    "size_bytes": size_bytes,
                    "indexed": False,
                }).execute()
            except Exception:
                pass

    if file is not None:
        _store_one(file, "file")
    if files:
        for f in files:
            _store_one(f, "files")

    if not received:
        raise HTTPException(
            status_code=422, detail="No file provided. Use 'file' or 'files'.")

    build_result = None
    if auto_build:
        try:
            # call build below programmatically
            build_result = build_ws.__wrapped__(
                ws_id=ws_id,
                overwrite_base_copy=False,
                force=True,
                max_new_docs=None,
                max_new_chunks=None,
            )
        except Exception as e:
            log.exception("auto-build failed for %s: %s", ws_id, e)

    return {"ok": True, "ws_id": ws_id, "received": received, "auto_build": auto_build, "build": build_result}


@router.post("/{ws_id}/build", summary="(Re)build FAISS index for this workspace from its uploads only")
def build_ws(
    ws_id: str,
    force: bool = Query(
        False, description="Rebuild even if index looks fresh"),
) -> Dict[str, Any]:
    """
    Always rebuild the index for this workspace using ONLY files under:
      <runtime>/workspaces/<ws_id>/uploads
    This never looks at settings.documents_path or any global docs.
    """
    p = WorkspacePaths.from_id(ws_id)
    _mk_dirs(p)

    steps: Dict[str, Any] = {
        "ws_id": ws_id,
        "paths": {
            "root": str(p.root),
            "uploads": str(p.uploads),
            "processed": str(p.processed),
            "index": str(p.index),
        },
    }

    _acquire_lock(p)
    try:
        # Optional freshness check (skip if not forced and nothing new)
        if not force:
            idx_mtime = _index_mtime(p)
            upl_mtime = _latest_upload_mtime(p)
            steps["freshness"] = {"index_mtime": idx_mtime,
                                  "latest_input_mtime": upl_mtime}
            if idx_mtime > 0 and upl_mtime <= idx_mtime:
                steps["skipped"] = True
                steps["reason"] = "index is up-to-date"
                return {"ok": True, **steps}

        # 💡 Key line: build strictly from this workspace's uploads
        stats = build_index_from_uploads_only(p)

        steps["mode"] = "uploads_only"
        steps["build"] = {"ok": True, "stats": stats}
        return {"ok": True, **steps}

    finally:
        _release_lock(p)


@router.delete("/{ws_id}", summary="Delete workspace and all its data")
def delete_ws(ws_id: str) -> Dict[str, Any]:
    p = WorkspacePaths.from_id(ws_id)
    _release_lock(p)
    delete_workspace(ws_id)
    return {"ok": True, "ws_id": ws_id}
