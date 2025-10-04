# backend/apps/ingestion/store_supabase.py
from __future__ import annotations
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Literal, Optional
import re
from datetime import datetime

from supabase import create_client
from core.config.config import settings
from apps.ingestion.schema import Section

DocIfExists = Literal["skip", "replace", "merge"]

EMBED_DIM = int(getattr(settings, "EMBED_DIM", 768))


def _supa():
    if not settings.supabaseUrl or not settings.supabaseKey:
        raise RuntimeError("Missing SUPABASE_URL/KEY in .env")
    # IMPORTANT: use the service role key server-side
    if "anon" in settings.supabaseKey:
        raise RuntimeError(
            "SUPABASE_KEY appears to be the anon key. Use the service_role key on the server.")
    return create_client(settings.supabaseUrl, settings.supabaseKey)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _mime_from_name(name: str) -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or "application/octet-stream"


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _chunk_cid(s: Section) -> str:
    norm = _norm_text(s.content)
    basis = f"{s.source or ''}|{len(norm)}|{norm}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _must_ok(resp, op: str):
    if hasattr(resp, "error") and resp.error:
        raise RuntimeError(f"Supabase {op} error: {resp.error}")
    if isinstance(resp, dict) and "error" in resp and resp["error"]:
        raise RuntimeError(f"Supabase {op} error: {resp['error']}")


def _get_document_by_sha(sb, sha: str) -> Optional[Dict]:
    resp = sb.table("documents").select("*").eq("sha256", sha).limit(1).execute()
    _must_ok(resp, "select documents by sha")
    data = resp.data or []
    return data[0] if data else None


# 3) INSERT document WITHOUT .select() chain; then re-select
def _insert_document(sb, *, filename: str, mime_type: str, size_bytes: int,
                     sha256: str, uploaded_by: Optional[str], source_url: Optional[str]) -> Dict:
    payload = {
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "uploaded_by": uploaded_by,
        "source_url": source_url,
        "metadata": {"slug": Path(filename).stem},
    }
    ins = sb.table("documents").insert(payload).execute()
    _must_ok(ins, "insert document")

    # Some client versions return the row in ins.data; if so, use it.
    data = getattr(ins, "data", None) or []
    if isinstance(data, list) and data:
        return data[0]

    # Otherwise, fetch it by sha256
    doc = _get_document_by_sha(sb, sha256)
    if not doc:
        raise RuntimeError("Inserted document but could not re-select by sha256")
    return doc


# 4) delete chunks (unchanged)
def _delete_chunks_for_document(sb, document_id: str) -> None:
    resp = sb.table("chunks").delete().eq("document_id", document_id).execute()
    _must_ok(resp, "delete chunks")



def _existing_cids_for_document(sb, document_id: str) -> set[str]:
    resp = sb.table("chunks").select("metadata").eq(
        "document_id", document_id).execute()
    _must_ok(resp, "select chunk cids")
    rows = resp.data or []
    cids: set[str] = set()
    for r in rows:
        md = r.get("metadata") or {}
        cid = md.get("cid")
        if cid:
            cids.add(str(cid))
    return cids


def _insert_chunks_batch(sb, *, document_id: str,
                         sections: List[Section], vectors: List[List[float]],
                         start_ordinal: int = 0) -> int:
    assert len(sections) == len(vectors), "sections/vectors length mismatch"

    for idx, v in enumerate(vectors):
        if not isinstance(v, (list, tuple)):
            raise TypeError(
                f"Embedding at {idx} is {type(v)}, expected list[float]")
        if len(v) != EMBED_DIM:
            raise RuntimeError(
                f"Embedding dim {len(v)} != EMBED_DIM {EMBED_DIM}")

    now = datetime.utcnow().isoformat() + "Z"
    rows = []
    for i, (s, v) in enumerate(zip(sections, vectors)):
        rows.append({
            "document_id": document_id,
            "ordinal": start_ordinal + i,
            "content": s.content,
            "embedding": [float(x) for x in v],
            "metadata": {
                "subject": s.subject,
                "source":  s.source,
                "cid":     _chunk_cid(s),
                "created_at": now,
            },
        })
    if not rows:
        return 0

    ins = sb.table("chunks").insert(rows).execute()
    _must_ok(ins, "insert chunks")
    return len(rows)


def upsert_document_with_chunks(
    *,
    file_path: Path,
    raw_bytes: bytes,
    sections: List[Section],
    vectors: List[List[float]],
    if_exists: DocIfExists = "skip",
    uploaded_by: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Dict:
    sb = _supa()

    sha = _sha256_bytes(raw_bytes)
    mime = _mime_from_name(file_path.name)
    size_bytes = len(raw_bytes)

    existing = _get_document_by_sha(sb, sha)

    if existing and if_exists == "skip":
        return {
            "document_id": existing["id"],
            "action": "skipped",
            "chunks_inserted": 0,
        }

    if not existing:
        doc = _insert_document(
            sb,
            filename=file_path.name,
            mime_type=mime,
            size_bytes=size_bytes,
            sha256=sha,
            uploaded_by=uploaded_by or getattr(settings, "UPLOADED_BY", None),
            source_url=source_url,
        )
        document_id = doc["id"]
        policy = "created"
    else:
        document_id = existing["id"]
        policy = if_exists

    if if_exists == "replace" and existing:
        _delete_chunks_for_document(sb, document_id)

    if if_exists == "merge" and existing:
        have = _existing_cids_for_document(sb, document_id)
        filt_sections: List[Section] = []
        filt_vectors: List[List[float]] = []
        for s, v in zip(sections, vectors):
            cid = _chunk_cid(s)
            if cid not in have:
                filt_sections.append(s)
                filt_vectors.append(v)
        sections, vectors = filt_sections, filt_vectors

    inserted = _insert_chunks_batch(
        sb,
        document_id=document_id,
        sections=sections,
        vectors=vectors,
        start_ordinal=0,
    )

    return {
        "document_id": document_id,
        "action": policy,
        "chunks_inserted": inserted,
    }
