from __future__ import annotations

from time import perf_counter
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .database import get_supabase
from .auth import _parse_session, _SESSION_COOKIE
from .models import AskRequest, AskResponse, Hit
from .runtime_llm import get_active_llm, STATE as RUNTIME_LLM_STATE

from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator
from rag.helpers import format_context_for_llm

from utils.workspaces import ensure_workspace, load_runtime_from_workspace

router = APIRouter()

# --------- singletons ---------
_retriever = RAGRetriever()  # global runtime already loaded in your app
_generator: Optional[RAGGenerator] = None
_last_llm_obj = None


def _current_user(request: Request) -> Optional[dict]:
    return _parse_session(request.cookies.get(_SESSION_COOKIE))


def _ensure_session(sb, user: Optional[dict], session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    r = sb.table("sessions").insert({
        "user_id": user.get("user_id") if user else None,
        "email": user.get("email") if user else None,
    }).execute()
    return r.data[0]["id"]


def _get_generator() -> RAGGenerator:
    global _generator, _last_llm_obj
    llm = get_active_llm()
    if _generator is None or llm is not _last_llm_obj:
        _generator = RAGGenerator(llm=llm)
        _last_llm_obj = llm
    return _generator


# ---------- GLOBAL ASK ----------
@router.post("", response_model=AskResponse, summary="Ask global index", tags=["ask"])
def ask(payload: AskRequest, request: Request) -> AskResponse:
    try:
        user = _current_user(request)
        query = (payload.q or "").strip()
        if not query:
            raise HTTPException(
                status_code=400, detail="Question cannot be empty")

        top_k = max(1, min(20, payload.k))
        min_score = payload.min_score or 0.3

        sb = get_supabase()
        session_id = _ensure_session(
            sb, user, getattr(payload, "session_id", None))

        # log user msg
        try:
            sb.table("messages").insert({
                "session_id": session_id,
                "user_id": user.get("user_id") if user else None,
                "email": user.get("email") if user else None,
                "role": "user",
                "content": query,
            }).execute()
        except Exception:
            pass

        # retrieve from global
        t0 = perf_counter()
        hits_raw = _retriever.retrieve(query, top_k=top_k, min_score=min_score)
        if not hits_raw and min_score > 0.2:
            hits_raw = _retriever.retrieve(query, top_k=top_k, min_score=0.2)
        hits = [Hit(**h) for h in hits_raw]

        context_str = None
        if payload.include_context and hits:
            context_str = format_context_for_llm(
                [h.model_dump() for h in hits])

        # generate
        gen = _get_generator()
        answer = gen.generate_answer(
            question=query,
            context_hits=[h.model_dump() for h in hits],
            max_chars=2000,
        )
        latency_ms = int((perf_counter() - t0) * 1000)

        # KPIs
        avg_similarity = None
        if hits:
            scrs = [h.score for h in hits if h.score is not None]
            if scrs:
                avg_similarity = sum(scrs) / len(scrs)

        # log assistant msg
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

        # log answer row
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
                "safe_answer": True,
            }).execute()
        except Exception:
            pass

        debug = {
            "hits_count": len(hits),
            "retriever_initialized": _retriever.runtime_data is not None,
            "provider": RUNTIME_LLM_STATE.provider,
            "llm_ready": RUNTIME_LLM_STATE.ready,
            "last_error": RUNTIME_LLM_STATE.last_error,
        }

        return AskResponse(
            query=query,
            top_k=top_k,
            min_score=min_score,
            hits=hits,
            context=context_str,
            answer=answer,
            session_id=session_id,
            debug=debug,
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# ---------- WORKSPACE ASK ----------
class AskWorkspaceRequest(AskRequest):
    """Same payload as AskRequest; session_id REQUIRED to bind workspace."""
    pass


@router.post(
    "/workspace",
    response_model=AskResponse,
    summary="Ask the per-session workspace index",
    tags=["ask"],
)
def ask_workspace(payload: AskWorkspaceRequest, request: Request) -> AskResponse:
    try:
        user = _current_user(request)
        query = (payload.q or "").strip()
        if not query:
            raise HTTPException(
                status_code=400, detail="Question cannot be empty")

        ws_id = getattr(payload, "session_id", None)
        if not ws_id:
            raise HTTPException(
                status_code=400, detail="session_id is required for workspace ask")

        # workspace must exist & be built
        try:
            ensure_workspace(ws_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail="Workspace not found for this session")

        try:
            runtime = load_runtime_from_workspace(ws_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=409, detail="Workspace index missing. Build the index first.")

        top_k = max(1, min(20, payload.k))
        min_score = payload.min_score or 0.3

        sb = get_supabase()
        session_id = _ensure_session(sb, user, ws_id)  # keep == ws_id

        # log user msg
        try:
            sb.table("messages").insert({
                "session_id": session_id,
                "user_id": user.get("user_id") if user else None,
                "email": user.get("email") if user else None,
                "role": "user",
                "content": query,
            }).execute()
        except Exception:
            pass

        # bind retriever to workspace runtime
        t0 = perf_counter()
        retr = RAGRetriever()
        retr.runtime_data = runtime
        hits_raw = retr.retrieve(query, top_k=top_k, min_score=min_score)
        if not hits_raw and min_score > 0.2:
            hits_raw = retr.retrieve(query, top_k=top_k, min_score=0.2)
        hits = [Hit(**h) for h in hits_raw]

        context_str = None
        if payload.include_context and hits:
            context_str = format_context_for_llm(
                [h.model_dump() for h in hits])

        # generate
        gen = _get_generator()
        answer = gen.generate_answer(
            question=query,
            context_hits=[h.model_dump() for h in hits],
            max_chars=2000,
        )
        latency_ms = int((perf_counter() - t0) * 1000)

        # KPIs
        avg_similarity = None
        if hits:
            scrs = [h.score for h in hits if h.score is not None]
            if scrs:
                avg_similarity = sum(scrs) / len(scrs)

        # log assistant msg
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

        # log answer
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
                "safe_answer": True,
            }).execute()
        except Exception:
            pass

        debug = {
            "hits_count": len(hits),
            "workspace_id": ws_id,
            "provider": RUNTIME_LLM_STATE.provider,
            "llm_ready": RUNTIME_LLM_STATE.ready,
            "last_error": RUNTIME_LLM_STATE.last_error,
        }

        return AskResponse(
            query=query,
            top_k=top_k,
            min_score=min_score,
            hits=hits,
            context=context_str,
            answer=answer,
            session_id=session_id,
            debug=debug,
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
