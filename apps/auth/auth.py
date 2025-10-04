# backend/apps/api/routers/auth.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, Depends, Request, Cookie
from pydantic import BaseModel
from typing import Optional

from apps.auth.security import (
    USERS,
    create_session,
    invalidate_session,
    current_user,
    _load_session
)

router = APIRouter()


class LoginReq(BaseModel):
    email: str
    password: str


class LoginResp(BaseModel):
    ok: bool
    email: str
    role: str


class UserInfo(BaseModel):
    ok: bool
    email: str
    role: str


@router.post("/login", response_model=LoginResp)
def login(p: LoginReq, response: Response):
    """Login endpoint that sets session cookie"""
    u = USERS.get(p.email)
    if not u or u["password"] != p.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    sid = create_session(p.email, u["role"])

    # Set an HTTP-only cookie with the session id
    response.set_cookie(
        key="session_id",
        value=sid,
        httponly=True,
        secure=False,  # Set True if you serve over HTTPS
        samesite="lax",
        max_age=60 * 60,  # Align with SESSION_TTL
        path="/",
    )
    return LoginResp(ok=True, email=p.email, role=u["role"])


@router.post("/logout")
def logout(
    response: Response,
    request: Request,
    session_cookie: Optional[str] = Cookie(default=None, alias="session_id")
):
    """Logout endpoint that clears session"""
    # Invalidate session if exists
    if session_cookie:
        invalidate_session(session_cookie)

    # Delete cookie
    response.delete_cookie("session_id", path="/")
    return {"ok": True}


@router.get("/me", response_model=UserInfo)
def get_current_user(user=Depends(current_user)):
    """Get current user info from session"""
    return UserInfo(
        ok=True,
        email=user["email"],
        role=user["role"]
    )


@router.get("/check")
def check_auth(user=Depends(current_user)):
    """Check if user is authenticated"""
    return {"authenticated": True, "email": user["email"], "role": user["role"]}
