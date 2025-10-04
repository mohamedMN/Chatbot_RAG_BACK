# backend/apps/ingestion/supa_upsert.py
from typing import Any, Dict, List, Optional
from core.storage.supabase import supa_client

TABLE = "chunks"


def upsert_chunk(
    document_id: str,
    ordinal: int,
    content: str,
    meta: Optional[dict],
    embedding: List[float],
) -> Dict[str, Any]:
    sb = supa_client()
    row = {
        "document_id": document_id,
        "ordinal": ordinal,
        "content": content,
        "metadata": meta or {},
        "embedding": embedding,
    }
    resp = sb.table(TABLE).insert(row).execute()
    return (getattr(resp, "data", None) or [row])[0]
