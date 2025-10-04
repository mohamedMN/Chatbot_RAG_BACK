# api/workspaces.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from utils.workspaces import (
    create_workspace,
    ensure_workspace,
    copy_base_to_workspace,
    delete_workspace,
    promote_workspace_to_base,
    WorkspacePaths,
)
from utils.simple_pipeline import (
    add_upload_to_workspace,
    build_workspace_from_uploads,  # copy base -> chunk -> embed -> rebuild faiss
)

# Optional admin dependency (header x-admin-key). Reuse your existing admin key if you have one
import os
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "supersecret")


def require_admin(x_admin_key: str = Depends(lambda: None)):
    # we implement a minimal header reader (no Pydantic dep) — Starlette Request is heavier
    # FastAPI shortcut: use a custom dependency if you prefer
    return True


router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", summary="Create a new ephemeral workspace")
def create_ws() -> Dict[str, Any]:
    p = create_workspace()
    return {"ws_id": p.ws_id, "paths": {
        "root": str(p.root), "uploads": str(p.uploads), "processed": str(p.processed), "index": str(p.index)
    }}


@router.post("/{ws_id}/documents", summary="Upload one or many documents")
def upload_docs(
    ws_id: str,
    file: Optional[UploadFile] = File(None),               # single 'file'
    files: Optional[List[UploadFile]] = File(None),        # many 'files'
):
    p = ensure_workspace(ws_id)
    received = []
    try:
        if file is not None:
            path = add_upload_to_workspace(p, file)
            received.append(
                {"key": "file", "name": file.filename, "stored_as": str(path)})
        if files:
            for f in files:
                path = add_upload_to_workspace(p, f)
                received.append(
                    {"key": "files", "name": f.filename, "stored_as": str(path)})
        if not received:
            raise HTTPException(
                status_code=422, detail="No file field found. Use 'file' or 'files'.")
        return {"ok": True, "ws_id": ws_id, "received": received}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"upload failed: {e}")


@router.post("/{ws_id}/build", summary="Copy base → add uploaded docs → rebuild FAISS for workspace")
def build_ws(ws_id: str, overwrite_base_copy: bool = False) -> Dict[str, Any]:
    p = ensure_workspace(ws_id)
    try:
        copy_base_to_workspace(p, overwrite=overwrite_base_copy)
        stats = build_workspace_from_uploads(p)
        return {"ok": True, "ws_id": ws_id, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"build failed: {e}")


@router.delete("/{ws_id}", summary="Delete workspace and all its data")
def delete_ws(ws_id: str) -> Dict[str, Any]:
    delete_workspace(ws_id)
    return {"ok": True, "ws_id": ws_id}


@router.post("/{ws_id}/promote", summary="Admin: promote workspace copy to become the new base (overwrites base)")
def promote_ws(ws_id: str, x_admin_key: Optional[str] = Query(None, alias="x-admin-key")) -> Dict[str, Any]:
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="invalid admin key")
    promote_workspace_to_base(ws_id)
    return {"ok": True, "ws_id": ws_id, "promoted_to_base": True}
