# backend/core/storage/supa_storage.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any
from core.storage.supabase import supa_client

__all__ = ["upload_blob"]

_DEFAULT_SIGNED_URL_TTL = 3600  # 1 hour


def upload_blob(
    bucket: str,
    path: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    add_date_prefix: bool = True,
    signed_url_ttl: int = _DEFAULT_SIGNED_URL_TTL,
) -> Dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("upload_blob: 'data' must be bytes")

    sb = supa_client()
    final_path = f"{datetime.utcnow().date()}/{path}" if add_date_prefix else path

    sb.storage.from_(bucket).upload(final_path, data, {
        "contentType": content_type, "upsert": True})
    resp = sb.storage.from_(bucket).create_signed_url(
        final_path, signed_url_ttl)

    signed_url = (
        (resp or {}).get("signedURL")
        or (resp or {}).get("signedUrl")
        or (resp or {}).get("signed_url")
        or ((resp or {}).get("data") or {}).get("signedUrl")
        or ((resp or {}).get("data") or {}).get("signedURL")
    ) or f"/storage/v1/object/{bucket}/{final_path}"

    return {"bucket": bucket, "path": final_path, "public_url": signed_url}
