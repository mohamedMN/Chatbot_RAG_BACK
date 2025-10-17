# api/models.py
from __future__ import annotations
import re
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# simple email sanity check (good enough for backend forms)
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ---------- Auth ----------


class SignupRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    # ignored by public signup unless you choose otherwise
    role: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("invalid email format")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("invalid email format")
        return v


class AuthResponse(BaseModel):
    user_id: str
    email: str
    role: str  # add this

# ---------- Ask ----------


class AskRequest(BaseModel):
    q: str = Field(..., description="User question")
    k: int = Field(4, ge=1, le=20)
    min_score: Optional[float] = Field(None)
    include_context: bool = Field(True)
    session_id: Optional[str] = None      # NEW
    workspace: Optional[str] = None  

class Hit(BaseModel):
    score: float
    subject: str
    content: str
    source: str
    ordinal: int
    id: int


class AskResponse(BaseModel):
    query: str
    top_k: int
    min_score: Optional[float]
    hits: List[Hit]
    context: Optional[str] = None
    answer: str
    session_id: Optional[str] = None      # NEW: return the session we used
    debug: Optional[dict] = None
