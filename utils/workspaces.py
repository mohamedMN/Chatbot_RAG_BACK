from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from config.settings import settings

WS_ROOT = Path(settings.runtime_path) / "workspaces"
WS_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class WorkspacePaths:
    ws_id: str
    root: Path
    uploads: Path
    processed: Path
    index: Path
    chunks: Path
    embeddings: Path
    faiss_index: Path
    idmap: Path
    metadata: Path
    build_log: Path

    @staticmethod
    def from_id(ws_id: str) -> "WorkspacePaths":
        base = WS_ROOT / ws_id
        return WorkspacePaths(
            ws_id=ws_id,
            root=base,
            uploads=base / "uploads",
            processed=base / "processed",
            index=base / "index",
            chunks=base / "processed" / "chunks.json",
            embeddings=base / "processed" / "embeddings.json",
            faiss_index=base / "index" / "faiss.index",
            idmap=base / "index" / "idmap.json",
            metadata=base / "index" / "metadata.json",
            build_log=base / "build_log.json",
        )


def create_workspace() -> WorkspacePaths:
    ws_id = uuid.uuid4().hex[:12]
    p = WorkspacePaths.from_id(ws_id)
    p.root.mkdir(parents=True, exist_ok=True)
    p.uploads.mkdir(parents=True, exist_ok=True)
    p.processed.mkdir(parents=True, exist_ok=True)
    p.index.mkdir(parents=True, exist_ok=True)
    p.build_log.write_text(json.dumps(
        {"ws_id": ws_id, "status": "created"}, indent=2), encoding="utf-8")
    return p


def ensure_workspace(ws_id: str) -> WorkspacePaths:
    p = WorkspacePaths.from_id(ws_id)
    if not p.root.exists():
        raise FileNotFoundError(f"Workspace {ws_id} not found.")
    return p


def copy_base_to_workspace(p: WorkspacePaths, overwrite: bool = False) -> None:
    if overwrite or not p.chunks.exists():
        if Path(settings.chunks_path).exists():
            shutil.copy2(settings.chunks_path, p.chunks)
    if overwrite or not p.embeddings.exists():
        if Path(settings.embeddings_path).exists():
            shutil.copy2(settings.embeddings_path, p.embeddings)
    if overwrite or not p.faiss_index.exists():
        if Path(settings.faiss_index_path).exists():
            shutil.copy2(settings.faiss_index_path, p.faiss_index)
    if overwrite or not p.idmap.exists():
        if Path(settings.idmap_path).exists():
            shutil.copy2(settings.idmap_path, p.idmap)
    if overwrite or not p.metadata.exists():
        if Path(settings.metadata_path).exists():
            shutil.copy2(settings.metadata_path, p.metadata)


def delete_workspace(ws_id: str) -> None:
    p = WorkspacePaths.from_id(ws_id)
    if p.root.exists():
        shutil.rmtree(p.root, ignore_errors=True)


def load_runtime_from_workspace(ws_id: str) -> Dict[str, any]:
    import faiss  # type: ignore

    p = ensure_workspace(ws_id)
    if not p.faiss_index.exists():
        raise FileNotFoundError("Workspace FAISS index missing")

    index = faiss.read_index(str(p.faiss_index))

    if p.idmap.exists():
        idmap = json.loads(p.idmap.read_text(encoding="utf-8"))
    else:
        idmap = _synthesize_idmap_from_chunks(p)

    meta = {}
    if p.metadata.exists():
        try:
            meta = json.loads(p.metadata.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    for k in ("ids", "ordinal", "content", "subject", "source"):
        idmap.setdefault(k, [])
    meta.setdefault("metric", "ip")
    return {"index": index, "idmap": idmap, "meta": meta}


def _synthesize_idmap_from_chunks(p: WorkspacePaths) -> Dict[str, list]:
    data = []
    if p.chunks.exists():
        data = json.loads(p.chunks.read_text(encoding="utf-8"))
    ids, ords, contents, subjects, sources = [], [], [], [], []
    for i, ch in enumerate(data):
        if isinstance(ch, str):
            contents.append(ch)
            subjects.append("")
            sources.append("")
            ords.append(i)
            ids.append(i)
        elif isinstance(ch, dict):
            contents.append(ch.get("content", ""))
            subjects.append(ch.get("subject", ""))
            sources.append(ch.get("source", ""))
            ords.append(int(ch.get("ordinal", i)))
            ids.append(int(ch.get("id", i)))
        else:
            contents.append(str(ch))
            subjects.append("")
            sources.append("")
            ords.append(i)
            ids.append(i)
    return {"ids": ids, "ordinal": ords, "content": contents, "subject": subjects, "source": sources}
