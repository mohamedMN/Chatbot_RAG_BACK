# api/history.py
from __future__ import annotations

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from datetime import datetime

from .database import get_supabase
from .auth import _parse_session, _SESSION_COOKIE

router = APIRouter()

# ------------ helpers ------------


def _current_user(request: Request) -> Optional[dict]:
    """Return {'user_id','email','role'} or None from signed cookie."""
    return _parse_session(request.cookies.get(_SESSION_COOKIE))


def _is_admin(user: Optional[dict]) -> bool:
    return bool(user and user.get("role") == "admin")


def _require_user(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _session_belongs_to_user(sb, session_id: str, user: dict) -> bool:
    """Check if session is owned by user via user_id or email."""
    r = sb.table("sessions").select("id,user_id,email").eq(
        "id", session_id).limit(1).execute()
    rows = r.data or []
    if not rows:
        # Not found
        raise HTTPException(status_code=404, detail="Session not found")
    row = rows[0]
    # owner match (allow null-safe matches)
    return (
        (user.get("user_id") and row.get("user_id") == user.get("user_id"))
        or (user.get("email") and row.get("email") == user.get("email"))
    )

# ------------ schemas ------------


class SessionRow(BaseModel):
    id: str
    user_id: str | None = None
    email: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class MessageRow(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime

# ------------ endpoints ------------


@router.get(
    "/ask/sessions",
    response_model=List[SessionRow],
    summary="List sessions for current user; admins see all",
    tags=["history"],
)
def list_sessions(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    sb = get_supabase()
    user = _current_user(request)

    q = sb.table("sessions").select("id,user_id,email,started_at,ended_at")

    if _is_admin(user):
        # Admin: see all
        q = q.order("started_at", desc=True).range(offset, offset + limit - 1)
    else:
        # Regular user must be logged in
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        # Filter by ownership (prefer user_id, fallback email)
        if user.get("user_id"):
            q = q.eq("user_id", user["user_id"])
        elif user.get("email"):
            q = q.eq("email", user["email"])
        else:
            # No identity info => no sessions
            return []
        q = q.order("started_at", desc=True).range(offset, offset + limit - 1)

    res = q.execute()
    return res.data or []


@router.get(
    "/ask/messages",
    response_model=List[MessageRow],
    summary="List messages for a session (owner or admin)",
    tags=["history"],
)
def list_messages(
    request: Request,
    session_id: str = Query(..., alias="session_id"),
    limit: int = Query(1000, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    sb = get_supabase()
    user = _current_user(request)

    # Authorization: admin can read any; user must own the session
    if not _is_admin(user):
        user = _require_user(request)
        if not _session_belongs_to_user(sb, session_id, user):
            raise HTTPException(status_code=403, detail="Forbidden")

    # Fetch messages
    q = (
        sb.table("messages")
        .select("id,session_id,role,content,created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)  # newest first
        .range(offset, offset + limit - 1)
    )
    res = q.execute()
    data = res.data or []

    # Your frontend likely wants ascending (oldest→newest) in the chat view.
    # If so, reverse here to keep the API consistent:
    data.sort(key=lambda r: r.get("created_at") or "")
    return data
