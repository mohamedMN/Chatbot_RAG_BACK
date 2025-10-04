# api/admin.py
from __future__ import annotations
import os
import hmac
import hashlib
import json
import base64
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Header, HTTPException, Request
from config.settings import settings

router = APIRouter()

# ---------- session helpers (mirror auth.py; ideally factor to utils/session.py) ----------
_SESSION_COOKIE = "session"
_SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me").encode()


def _sign(b: bytes) -> bytes:
    return hmac.new(_SESSION_SECRET, b, hashlib.sha256).digest()


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64u(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _parse_session(token: Optional[str]) -> Optional[dict]:
    if not token or "." not in token:
        return None
    body, sig = token.split(".", 2)
    good = _b64u(_sign(body.encode("ascii")))
    if not hmac.compare_digest(sig, good):
        return None
    try:
        return json.loads(_unb64u(body))
    except Exception:
        return None

# ---------- authorization ----------


def _require_admin(request: Request, x_admin_key: Optional[str]) -> dict:
    sess = _parse_session(request.cookies.get(_SESSION_COOKIE))
    # 1) prefer cookie role
    if sess and sess.get("role") == "admin":
        return sess
    # 2) fallback header key if configured
    want = os.getenv("ADMIN_API_KEY")
    if want and x_admin_key == want:
        return {"user_id": "admin-key", "email": "admin-key@local", "role": "admin"}
    # otherwise reject
    raise HTTPException(status_code=403, detail="Admin privileges required")

# ---------- endpoints ----------


@router.get("/ping", summary="Admin ping")
def admin_ping(request: Request, x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_admin(request, x_admin_key)
    return {"ok": True, "msg": "Welcome to admin dashboard"}


@router.get("/stats", summary="Admin stats")
def admin_stats(request: Request, x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_admin(request, x_admin_key)
    # TODO: wire real metrics here
    # Example placeholders that match your UI keys:
    return {
        "conversations_24h": 12,
        "delta_24h": 8.3,
        "documents_total": 154,
        "documents_7d": 9,
        "retrieval_topk_avg": 4.0,
        "similarity_avg": 0.723,
        "latency_ms_avg": 412,
        "safe_answer_rate": 95.2,
    }


@router.post("/reindex", summary="Rebuild FAISS index")
def reindex(request: Request, x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_admin(request, x_admin_key)
    # TODO: from indexing.index_builder import rebuild_all
    # stats = rebuild_all()
    stats = {"indexed": 0, "status": "noop (wire your builder)"}
    return {"ok": True, "stats": stats}


@router.post("/flush-index", summary="Delete FAISS artifacts")
def flush_index(request: Request, x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_admin(request, x_admin_key)
    removed = []
    for p in (settings.faiss_index_path, settings.idmap_path, settings.metadata_path):
        p = Path(p)
        if p.exists():
            p.unlink()
            removed.append(str(p))
    return {"ok": True, "removed": removed}
