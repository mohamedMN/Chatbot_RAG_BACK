# utils/index_uploads_only.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import faiss  # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import docx2txt
except Exception:
    docx2txt = None

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, FAISS_NORMALIZE_VECTORS
from utils.workspaces import WorkspacePaths

ALLOWED_EXTS = {".txt", ".pdf", ".docx"}
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
CHUNK_CHARS = 700
CHUNK_OVERLAP = 80


def build_index_from_uploads_only(p: WorkspacePaths) -> Dict[str, Any]:
    p.root.mkdir(parents=True, exist_ok=True)
    p.uploads.mkdir(parents=True, exist_ok=True)
    p.processed.mkdir(parents=True, exist_ok=True)
    p.index.mkdir(parents=True, exist_ok=True)

    files = _list_allowed(p.uploads)
    texts, skipped = _extract_texts(files)
    chunks = _chunk_all(texts)

    _write_json(p.chunks, chunks)

    contents = [c["content"] for c in chunks]
    vectors: List[List[float]] = []
    if contents:
        model = SentenceTransformer(EMBEDDING_MODEL)
        vecs = model.encode(
            contents, normalize_embeddings=FAISS_NORMALIZE_VECTORS)
        real_dim = int(vecs.shape[1]) if hasattr(
            vecs, "shape") else len(vecs[0])
        if real_dim != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}, model={EMBEDDING_MODEL} -> {real_dim}")
        if getattr(vecs, "dtype", None) != np.float32:
            vecs = vecs.astype("float32")
        vectors = [v.tolist() for v in vecs]

    _write_json(p.embeddings, {"vectors": vectors, "dim": EMBEDDING_DIMENSION})
    _write_faiss(p, vectors)

    idmap, meta = _idmap_and_meta(chunks, len(vectors))
    _write_json(p.idmap, idmap)
    _write_json(p.metadata, meta)

    return {
        "uploads": len(files),
        "new_chunks": len(chunks),
        "total_chunks": len(chunks),
        "total_vectors": len(vectors),
        "faiss_dim": EMBEDDING_DIMENSION,
        "skipped": skipped,
    }


def _list_allowed(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    out = []
    for f in sorted(folder.glob("*")):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
            out.append(f)
    return out


def _extract_texts(files: List[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    texts, skipped = [], []
    for f in files:
        reason = None
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                reason = "too_big"
            else:
                txt = _read_text(f)
                if not (txt or "").strip():
                    reason = "empty_extraction"
                else:
                    texts.append(
                        {"source": str(f), "subject": f.name, "text": txt[:MAX_TEXT_CHARS]})
        except Exception as e:
            reason = f"extract_error:{e}"
        if reason:
            skipped.append({"path": str(f), "reason": reason})
    return texts, skipped


def _read_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suf == ".pdf" and PyPDF2:
        out = []
        with open(path, "rb") as fp:
            reader = PyPDF2.PdfReader(fp)
            for p in reader.pages:
                out.append(p.extract_text() or "")
        return "\n".join(out)
    if suf == ".docx" and docx2txt:
        return docx2txt.process(str(path)) or ""
    return ""


def _chunk_all(texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks, ordinal = [], 0
    for item in texts:
        for piece in _split(item["text"], CHUNK_CHARS, CHUNK_OVERLAP):
            chunks.append({
                "id": ordinal,
                "ordinal": ordinal,
                "content": piece,
                "subject": item["subject"],
                "source": item["source"],
            })
            ordinal += 1
    return chunks


def _split(text: str, max_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(1, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars - 1))
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        out.append(text[i:j])
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return out


def _write_faiss(p: WorkspacePaths, vectors: List[List[float]]) -> None:
    if not vectors:
        idx = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        faiss.write_index(idx, str(p.faiss_index))
        return
    mat = np.array(vectors, dtype="float32")
    if mat.ndim != 2:
        raise RuntimeError(f"Bad embeddings shape: {mat.shape}")
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    faiss.write_index(idx, str(p.faiss_index))


def _idmap_and_meta(chunks: List[Dict[str, Any]], vec_count: int):
    ids, ords, contents, subjects, sources = [], [], [], [], []
    for i, ch in enumerate(chunks):
        ids.append(i)
        ords.append(i)
        contents.append(ch.get("content", ""))
        subjects.append(ch.get("subject", ""))
        sources.append(ch.get("source", ""))
    idmap = {"ids": ids, "ordinal": ords, "content": contents,
             "subject": subjects, "source": sources}
    meta = {"metric": "ip", "dimension": EMBEDDING_DIMENSION,
            "total_vectors": vec_count}
    return idmap, meta


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False,
                    indent=2), encoding="utf-8")
