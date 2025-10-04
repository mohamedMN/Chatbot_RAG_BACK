# api/auth.py (top)
from __future__ import annotations
import base64
import os
import hmac
import hashlib
import json
from typing import Optional, Dict, Any
# ⬅️ add Request, Response
from fastapi import APIRouter, HTTPException, Request, Response
from .database import get_supabase
from .models import SignupRequest, LoginRequest, AuthResponse

router = APIRouter()

_PBKDF2_ITERS = 210_000

# --- session cookie config ---
_SESSION_COOKIE = "session"
_SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me").encode()


def _sign(b: bytes) -> bytes:
    return hmac.new(_SESSION_SECRET, b, hashlib.sha256).digest()


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64u(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _create_session(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    body = _b64u(raw)
    sig = _b64u(_sign(body.encode("ascii")))
    return f"{body}.{sig}"


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


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return "pbkdf2${iters}${salt}${digest}".format(
        iters=_PBKDF2_ITERS,
        salt=base64.b64encode(salt).decode("ascii"),
        digest=base64.b64encode(dk).decode("ascii"),
    )


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters_str, salt_b64, digest_b64 = stored.split("$", 3)
        if scheme != "pbkdf2":
            return False
        iters = int(iters_str)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        digest = base64.b64decode(digest_b64.encode("ascii"))
        test = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(test, digest)
    except Exception:
        return False

# ---------- Endpoints ----------


@router.post("/signup", response_model=AuthResponse, summary="Create a user (DB-based)")
def signup(payload: SignupRequest) -> AuthResponse:
    sb = get_supabase()

    # 1) unique email
    q = sb.table("users_local").select("id").eq(
        "email", payload.email.lower()).limit(1).execute()
    if q.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 2) hash + insert, force role='user'
    pw_hash = _hash_password(payload.password)
    ins = sb.table("users_local").insert({
        "email": payload.email.lower(),
        "password_hash": pw_hash,
        "role": payload.role if payload.role else "user",
    }).execute()

    if not ins.data:
        raise HTTPException(status_code=400, detail="Signup failed")

    row = ins.data[0]
    return AuthResponse(user_id=row["id"], email=row["email"], role=row["role"])


@router.post("/login", response_model=AuthResponse, summary="Login (DB-based, no JWT)")
def login(payload: LoginRequest, response: Response) -> AuthResponse:   # ⬅️ add Response
    sb = get_supabase()
    q = sb.table("users_local").select("id,email,password_hash,role").eq(
        "email", payload.email.lower()
    ).limit(1).execute()
    if not q.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    row = q.data[0]
    if not _verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # set a signed session cookie containing id/email/role
    token = _create_session(
        {"user_id": row["id"], "email": row["email"], "role": row.get("role") or "user"})
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        path="/",
        httponly=True,
        samesite="lax",
        secure=False,  # set True behind HTTPS
        max_age=60 * 60 * 24 * 7,  # 7 days
    )

    return AuthResponse(
        user_id=row["id"],
        email=row["email"],
        role=row.get("role") or "user",
    )




@router.get("/me", response_model=AuthResponse)
def me(request: Request) -> AuthResponse:
    sess = _parse_session(request.cookies.get(_SESSION_COOKIE))
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Prefer role from session; if missing, fetch from DB (for older cookies)
    role = sess.get("role")
    if not role:
        sb = get_supabase()
        q = sb.table("users_local").select("role").eq(
            "id", sess["user_id"]).limit(1).execute()
        role = (q.data[0]["role"] if q.data and q.data[0].get(
            "role") else "user")

    return AuthResponse(user_id=sess["user_id"], email=sess["email"], role=role)


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(_SESSION_COOKIE, path="/")
    return {"ok": True}