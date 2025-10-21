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
_HAS_SMART_CHUNKER = False
try:
    # put your file at core/document_chunker.py (class DocumentChunker as you posted)
    from core.chunker import DocumentChunker
    _chunker = DocumentChunker()
    _HAS_SMART_CHUNKER = True
except Exception:
    _HAS_SMART_CHUNKER = False




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
def build_workspace_from_uploads(
    p: WorkspacePaths,
    max_new_docs: int | None = None,
    max_new_chunks: int | None = None,
) -> Dict[str, Any]:
    # 1) Read uploads
    new_chunks = _extract_chunks_from_uploads(
        p, max_docs=max_new_docs, max_chunks=max_new_chunks)

    # collect which files produced chunks
    sources_indexed = sorted({(ch.get("source") or "").strip()
                             for ch in new_chunks if ch.get("source")})

    # 2) Merge with existing chunks
    all_chunks, new_count = _merge_chunks(p.chunks, new_chunks)
    p.chunks.write_text(json.dumps(
        all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Embed
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = _load_embeddings_json(p.embeddings)
    vectors = embeddings.get("vectors", [])
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

    # 5) idmap + metadata
    idmap, metadata = _build_idmap_and_metadata(all_chunks)
    p.idmap.write_text(json.dumps(
        idmap, ensure_ascii=False, indent=2), encoding="utf-8")
    p.metadata.write_text(json.dumps(
        metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "uploads": len(list(p.uploads.glob("*"))),
        "sources_indexed": sources_indexed,    # <-- NEW
        "new_chunks": new_count,
        "total_chunks": len(all_chunks),
        "total_vectors": len(vectors),
        "faiss_dim": EMBEDDING_DIMENSION,
    }



# --------------------- helpers ---------------------
def _extract_chunks_from_uploads(
    p: WorkspacePaths,
    max_docs: int | None = None,
    max_chunks: int | None = None,
) -> List[Dict[str, Any]]:
    files = [f for f in p.uploads.glob("*") if f.is_file()]

    # Optional: restrict to known text-ish types to avoid binaries
    ALLOWED = {".txt", ".pdf", ".docx"}
    files = [f for f in files if f.suffix.lower() in ALLOWED]

    if not files:
        return []

    if max_docs is not None:
        files = files[:max_docs]

    chunks: List[Dict[str, Any]] = []
    ordinal = 0

    # -------- 1) If you have a doc loader, use it --------
    if _HAS_LOADER:
        docs = load_documents([str(f) for f in files])
        for i, d in enumerate(docs):
            content = d.get("content", "") if isinstance(d, dict) else ""
            if not content:
                continue
            source = d.get("source", str(files[min(i, len(files) - 1)]))
            subject = d.get("subject", Path(source).name)
            predefined_sections = d.get("predefined_sections", [])

            # Prefer SMART chunker if available
            if _HAS_SMART_CHUNKER:
                doc_chunks = _chunker.chunk_document({
                    "id": i,
                    "content": content,
                    "source": source,
                    "predefined_sections": predefined_sections,
                })
                for ch in doc_chunks:
                    chunks.append({
                        "id": ordinal,
                        "ordinal": ordinal,
                        "content": ch.get("content", ""),
                        "source": source,
                        "subject": subject,
                        # keep the section your chunker computed (nice for UI/ranking)
                        "section": ch.get("section", "Section Principale"),
                    })
                    ordinal += 1
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        return chunks
            else:
                # If no smart chunker, at least capture full doc (previous behavior)
                # (or you can keep your 600/60 simple splitter here)
                piece = content[:MAX_TEXT_CHARS]
                if piece:
                    chunks.append({
                        "id": ordinal,
                        "ordinal": ordinal,
                        "content": piece,
                        "source": source,
                        "subject": subject,
                    })
                    ordinal += 1
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        return chunks
        return chunks

    # -------- 2) No loader: read file → SMART chunker → fallback splitter --------
    for f in files:
        text = _read_text_fallback(f)
        if not text:
            continue
        source = str(f)
        subject = f.name

        if _HAS_SMART_CHUNKER:
            doc_chunks = _chunker.chunk_document({
                "id": 0,
                "content": text[:MAX_TEXT_CHARS],
                "source": source,
                "predefined_sections": [],
            })
            for ch in doc_chunks:
                chunks.append({
                    "id": ordinal,
                    "ordinal": ordinal,
                    "content": ch.get("content", ""),
                    "source": source,
                    "subject": subject,
                    "section": ch.get("section", "Section Principale"),
                })
                ordinal += 1
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks
        else:
            # last resort: simple fixed-size splitter
            for piece in _split_simple(text, max_chars=600, overlap=60):
                chunks.append({
                    "id": ordinal,
                    "ordinal": ordinal,
                    "content": piece,
                    "source": source,
                    "subject": subject,
                })
                ordinal += 1
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks

    return chunks




MAX_FILE_BYTES = 5 * 1024 * 1024      # 5 MB per file (tune)
MAX_TEXT_CHARS = 2_000_000            # 2M chars per file (tune)


def _read_text_fallback(path: Path) -> str:
    suf = path.suffix.lower()
    try:
        # Quick size check to skip massive/binary uploads
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                return ""  # or return a short notice; we skip gigantic files
        except Exception:
            pass

        if suf == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")[:MAX_TEXT_CHARS]

        if suf == ".pdf":
            try:
                import PyPDF2
                text = []
                with open(path, "rb") as fp:
                    reader = PyPDF2.PdfReader(fp)
                    for page in reader.pages:
                        if len("".join(text)) > MAX_TEXT_CHARS:
                            break
                        text.append(page.extract_text() or "")
                return ("\n".join(text))[:MAX_TEXT_CHARS]
            except Exception:
                return ""

        if suf == ".docx":
            try:
                import docx2txt
                return (docx2txt.process(str(path)) or "")[:MAX_TEXT_CHARS]
            except Exception:
                return ""

        # everything else: treat as text but cap
        return path.read_text(encoding="utf-8", errors="ignore")[:MAX_TEXT_CHARS]
    except Exception:
        return ""



def _split_simple(text: str, max_chars: int = 600, overlap: int = 60) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Safety: keep overlap sane
    max_chars = max(1, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars - 1))

    chunks: List[str] = []
    n = len(text)
    i = 0

    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])

        if j >= n:
            # we reached the end — stop (prevents infinite loop)
            break

        # advance with overlap but always move forward
        next_i = j - overlap
        if next_i <= i:
            # ensure progress even if overlap is misconfigured
            next_i = j

        i = next_i

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
