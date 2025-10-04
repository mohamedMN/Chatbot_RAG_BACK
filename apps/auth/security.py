# backend/apps/auth/security.py
from __future__ import annotations
import os
import secrets
import time
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, status, Header, Cookie, Request

# Demo users (replace by DB/IdP later)
USERS: Dict[str, Dict[str, str]] = {
    "admin@orange.com": {"password": "admin123", "role": "admin"},
    # "user@orange.com": {"password": "user123", "role": "user"},
}

# In-memory session store: { session_id: {"email":..., "role":..., "exp":...} }
# For production, back this with Redis or your DB (and rotate secrets).
_SESSIONS: Dict[str, Dict[str, Any]] = {}

# Session lifetime (seconds)
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1h default


def _now() -> int:
    return int(time.time())


def create_session(email: str, role: str) -> str:
    sid = secrets.token_urlsafe(32)
    _SESSIONS[sid] = {"email": email,
                      "role": role, "exp": _now() + SESSION_TTL}
    return sid


def invalidate_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


def _load_session(session_id: Optional[str]) -> Dict[str, Any]:
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="no session")
    s = _SESSIONS.get(session_id)
    if not s:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid session")
    if s["exp"] < _now():
        _SESSIONS.pop(session_id, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="session expired")
    # sliding expiration (optional): refresh TTL on use
    s["exp"] = _now() + SESSION_TTL
    return s


def current_user(
    request: Request,
    session_cookie: Optional[str] = Cookie(default=None, alias="session_id"),
    session_header: Optional[str] = Header(default=None, alias="X-Session-Id"),
) -> Dict[str, Any]:
    """
    Resolve the current user from either cookie 'session_id' or header 'X-Session-Id'.
    """
    sid = session_cookie or session_header
    sess = _load_session(sid)
    return {"email": sess["email"], "role": sess["role"]}


def admin_required(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="admin required")
    return user
