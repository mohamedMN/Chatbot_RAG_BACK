# api/ask.py
from __future__ import annotations

from time import perf_counter
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Query

from pydantic import BaseModel

from .database import get_supabase
from .models import AskRequest, AskResponse, Hit
from .runtime_llm import STATE as RUNTIME_LLM_STATE

from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator
from rag.helpers import format_context_for_llm

# signed cookie helpers
from .auth import _parse_session, _SESSION_COOKIE
from .runtime_llm import get_active_llm, STATE as RUNTIME_LLM_STATE

router = APIRouter()

# ---------------- Singletons ----------------
_retriever = RAGRetriever()
_generator: Optional[RAGGenerator] = None


def _get_generator() -> RAGGenerator:
    global _generator, _last_llm_obj
    llm = get_active_llm()  # may be None
    if _generator is None or llm is not _last_llm_obj:
        # strict, context-anchored generator
        _generator = RAGGenerator(llm=llm)
        _last_llm_obj = llm
    return _generator


def _current_user(request: Request) -> Optional[dict]:
    """Return {'user_id','email','role'} or None from signed cookie."""
    return _parse_session(request.cookies.get(_SESSION_COOKIE))


def _ensure_session(sb, user: Optional[dict], session_id: Optional[str]) -> str:
    """Return a session_id; create a new session if none provided."""
    if session_id:
        return session_id
    r = sb.table("sessions").insert({
        "user_id": user.get("user_id") if user else None,
        "email": user.get("email") if user else None,
    }).execute()
    return r.data[0]["id"]


# ---------------- Ask endpoint ----------------
@router.post("", response_model=AskResponse, summary="Ask RAG", tags=["ask"])
def ask(payload: AskRequest, request: Request) -> AskResponse:
    try:
        user = _current_user(request)  # may be None
        query = (payload.q or "").strip()
        if not query:
            raise HTTPException(
                status_code=400, detail="Question cannot be empty")

        top_k = max(1, min(20, payload.k))
        min_score = payload.min_score or 0.3

        sb = get_supabase()

        # Ensure session
        session_id = _ensure_session(
            sb, user, getattr(payload, "session_id", None))

        # Log user message
        try:
            sb.table("messages").insert({
                "session_id": session_id,
                "user_id": user.get("user_id") if user else None,
                "email": user.get("email") if user else None,
                "role": "user",
                "content": query,
            }).execute()
        except Exception:
            # non-blocking
            pass

        # Retrieval
        t0 = perf_counter()
        hits_raw = _retriever.retrieve(query, top_k=top_k, min_score=min_score)
        if not hits_raw and min_score > 0.2:
            hits_raw = _retriever.retrieve(query, top_k=top_k, min_score=0.2)
        hits = [Hit(**h) for h in hits_raw]

        # Optional context
        context_str = None
        if payload.include_context and hits:
            context_str = format_context_for_llm(
                [h.model_dump() for h in hits])

        # Generation
        generator = _get_generator()
        answer = generator.generate_answer(
            question=query,
            context_hits=[h.model_dump() for h in hits],
            max_chars=2000,
        )
        latency_ms = int((perf_counter() - t0) * 1000)

        # KPIs (simple)
        avg_similarity = None
        if hits:
            scores = [h.score for h in hits if h.score is not None]
            if scores:
                avg_similarity = sum(scores) / len(scores)
        safe_answer = True  # wire your real checker if needed

        # Log assistant message
        try:
            sb.table("messages").insert({
                "session_id": session_id,
                "user_id": user.get("user_id") if user else None,
                "email": user.get("email") if user else None,
                "role": "assistant",
                "content": answer,
            }).execute()
        except Exception:
            pass

        # Log answer KPIs
        try:
            sb.table("answers").insert({
                "session_id": session_id,
                "user_id": user.get("user_id") if user else None,
                "email": user.get("email") if user else None,
                "question": query,
                "answer": answer,
                "similarity": avg_similarity,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "safe_answer": safe_answer,
            }).execute()
        except Exception:
            pass

        debug_info = {
            "hits_count": len(hits),
            "retriever_initialized": _retriever.runtime_data is not None,
            "provider": RUNTIME_LLM_STATE.provider,
            "llm_ready": RUNTIME_LLM_STATE.ready,
            "last_error": RUNTIME_LLM_STATE.last_error,
        }

        # ⚠️ If your AskResponse doesn't have session_id, remove it here.
        return AskResponse(
            query=query,
            top_k=top_k,
            min_score=min_score,
            hits=hits,
            context=context_str,
            answer=answer,
            session_id=session_id,   # requires field in models.AskResponse
            debug=debug_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


# ---------------- Questions listing (per-user) ----------------
class QuestionRow(BaseModel):
    id: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    question: str
    workspace: Optional[str] = None
    created_at: datetime


@router.get(
    "/questions",
    response_model=List[QuestionRow],
    summary="List my questions",
    tags=["ask"],
)
def list_my_questions(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """
    Prefer messages (role='user') as the source of 'questions'.
    Falls back to chat_questions if that table exists.
    """
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    sb = get_supabase()

    # Try messages first
    try:
        q = (
            sb.table("messages")
            .select("id, user_id, email, role, content, created_at")
            .eq("user_id", user["user_id"])
            .eq("role", "user")
        )
        if start:
            q = q.gte("created_at", f"{start}T00:00:00Z")
        if end:
            q = q.lt("created_at", f"{end}T23:59:59Z")
        q = q.order("created_at", desc=True).range(offset, offset + limit - 1)
        res = q.execute()
        if res.data is not None:
            rows = [
                {
                    "id": r["id"],
                    "user_id": r.get("user_id"),
                    "email": r.get("email"),
                    "question": r.get("content", ""),
                    "workspace": None,  # messages table has no workspace column
                    "created_at": r["created_at"],
                }
                for r in res.data
            ]
            # pydantic will coerce
            return rows
    except Exception:
        # fall through to chat_questions
        pass

    # Fallback: chat_questions table (legacy)
    try:
        q = sb.table("chat_questions").select(
            "*").eq("user_id", user["user_id"])
        if start:
            q = q.gte("created_at", f"{start}T00:00:00Z")
        if end:
            q = q.lt("created_at", f"{end}T23:59:59Z")
        q = q.order("created_at", desc=True).range(offset, offset + limit - 1)
        res = q.execute()
        return res.data or []
    except Exception:
        return []
