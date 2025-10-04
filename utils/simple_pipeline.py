# utils/simple_pipeline.py
from __future__ import annotations
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer
import numpy as np

import io
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

from fastapi import UploadFile

from config.settings import settings, EMBEDDING_MODEL, EMBEDDING_DIMENSION, FAISS_NORMALIZE_VECTORS
from utils.workspaces import WorkspacePaths

# Try your real modules first; otherwise fallback to simple extractors
_HAS_LOADER = False
try:
    # expected signature: List[Dict]
    from core.document_loader import load_documents
    _HAS_LOADER = True
except Exception:
    _HAS_LOADER = False

# Embeddings (fallback)

# FAISS


# --------------------- Upload ---------------------
def add_upload_to_workspace(p: WorkspacePaths, file: UploadFile) -> Path:
    target = p.uploads / file.filename
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return target


# --------------------- Build (copy base -> add docs -> reindex) ---------------------
def build_workspace_from_uploads(p: WorkspacePaths) -> Dict[str, Any]:
    """
    Steps:
      1) Read workspace uploads -> new chunks
      2) Merge new chunks with workspace processed/chunks.json
      3) Embed only new chunks; merge with workspace embeddings.json
      4) Rebuild FAISS at workspace/index
      5) Build idmap.json & metadata.json
    """
    # 1) Read uploads
    new_chunks = _extract_chunks_from_uploads(p)
    # 2) Merge with existing chunks
    all_chunks, new_count = _merge_chunks(p.chunks, new_chunks)
    p.chunks.write_text(json.dumps(
        all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Embed
    model = SentenceTransformer(EMBEDDING_MODEL)
    # {"vectors":[[...]], "dim":384}
    embeddings = _load_embeddings_json(p.embeddings)
    vectors = embeddings.get("vectors", [])

    start_idx = len(vectors)
    new_texts = [c["content"] for c in new_chunks]
    if new_texts:
        new_vecs = model.encode(
            new_texts, normalize_embeddings=FAISS_NORMALIZE_VECTORS)
        if new_vecs.dtype != np.float32:
            new_vecs = new_vecs.astype("float32")
        for v in new_vecs:
            vectors.append(v.tolist())

    merged_embeddings = {"vectors": vectors, "dim": EMBEDDING_DIMENSION}
    p.embeddings.write_text(json.dumps(merged_embeddings), encoding="utf-8")

    # 4) Rebuild FAISS
    _build_faiss_from_embeddings(p, vectors)

    # 5) idmap + metadata aligned with chunks
    idmap, metadata = _build_idmap_and_metadata(all_chunks)
    p.idmap.write_text(json.dumps(
        idmap, ensure_ascii=False, indent=2), encoding="utf-8")
    p.metadata.write_text(json.dumps(
        metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "uploads": len(list(p.uploads.glob("*"))),
        "new_chunks": new_count,
        "total_chunks": len(all_chunks),
        "total_vectors": len(vectors),
        "faiss_dim": EMBEDDING_DIMENSION,
    }


# --------------------- helpers ---------------------
def _extract_chunks_from_uploads(p: WorkspacePaths) -> List[Dict[str, Any]]:
    files = list(p.uploads.glob("*"))
    if not files:
        return []

    if _HAS_LOADER:
        # Use your real loader if available
        # expect list of dicts with 'content','source','subject'
        docs = load_documents([str(f) for f in files])
        return [
            {
                "id": i,
                "ordinal": i,
                "content": d.get("content", ""),
                "source": d.get("source", str(files[i])) if isinstance(d, dict) else str(files[i]),
                "subject": d.get("subject", ""),
            }
            for i, d in enumerate(docs)
        ]

    # Fallback: minimal loader for .txt/.pdf/.docx
    chunks: List[Dict[str, Any]] = []
    ordinal = 0
    for f in files:
        text = _read_text_fallback(f)
        for piece in _split_simple(text, max_chars=600, overlap=60):
            chunks.append({
                "id": ordinal,
                "ordinal": ordinal,
                "content": piece,
                "source": str(f),
                "subject": f.name,
            })
            ordinal += 1
    return chunks


def _read_text_fallback(path: Path) -> str:
    suf = path.suffix.lower()
    try:
        if suf == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        if suf == ".pdf":
            try:
                import PyPDF2
                text = []
                with open(path, "rb") as fp:
                    reader = PyPDF2.PdfReader(fp)
                    for page in reader.pages:
                        text.append(page.extract_text() or "")
                return "\n".join(text)
            except Exception:
                return ""
        if suf in {".docx"}:
            try:
                import docx2txt
                return docx2txt.process(str(path)) or ""
            except Exception:
                return ""
        # other types → raw bytes to string
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _split_simple(text: str, max_chars: int = 600, overlap: int = 60) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j]
        chunks.append(chunk)
        i = j - overlap
        if i <= 0:
            i = j
    return chunks


def _merge_chunks(existing_path: Path, new_chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    all_chunks: List[Dict[str, Any]] = []
    if existing_path.exists():
        try:
            data = json.loads(existing_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_chunks = data
        except Exception:
            all_chunks = []
    # append new
    base_len = len(all_chunks)
    all_chunks.extend(new_chunks)
    # reassign ids/ordinals to be 0..len-1
    for i, ch in enumerate(all_chunks):
        ch["id"] = i
        ch["ordinal"] = i
        ch.setdefault("subject", "")
        ch.setdefault("source", "")
    return all_chunks, len(all_chunks) - base_len


def _load_embeddings_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"vectors": [], "dim": EMBEDDING_DIMENSION}
    return {"vectors": [], "dim": EMBEDDING_DIMENSION}


def _build_faiss_from_embeddings(p: WorkspacePaths, vectors_list: List[List[float]]) -> None:
    if not vectors_list:
        # Create empty index of correct dim
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        faiss.write_index(index, str(p.faiss_index))
        return
    mat = np.array(vectors_list, dtype="float32")
    index = faiss.IndexFlatIP(mat.shape[1])  # cosine if normalized
    index.add(mat)
    faiss.write_index(index, str(p.faiss_index))


def _build_idmap_and_metadata(chunks: List[Dict[str, Any]]) -> Tuple[Dict[str, list], Dict[str, Any]]:
    ids, ords, contents, subjects, sources = [], [], [], [], []
    for i, ch in enumerate(chunks):
        ids.append(i)
        ords.append(i)
        contents.append(ch.get("content", ""))
        subjects.append(ch.get("subject", ""))
        sources.append(ch.get("source", ""))
    idmap = {"ids": ids, "ordinal": ords, "content": contents,
             "subject": subjects, "source": sources}
    metadata = {"metric": "ip", "count": len(chunks)}
    return idmap, metadata
