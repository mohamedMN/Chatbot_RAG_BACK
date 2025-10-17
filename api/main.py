from __future__ import annotations

import logging
from typing import Any, Dict
from pathlib import Path

from fastapi import APIRouter

from config.settings import settings

# Sub-routers
from .auth import router as auth_router     # public
from .ask import router as ask_router       # public
from .admin import router as admin_router   # protected via x-admin-key
from .runtime_llm import router as llm_router  # LLM management
from .workspaces import router as workspaces_router  # LLM management

api_router = APIRouter()
log = logging.getLogger("chatbot_rag.api")

# Include routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(ask_router,  prefix="/ask",  tags=["ask"])
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
api_router.include_router(workspaces_router)
api_router.include_router(llm_router)  # Already has prefix="/llm" in the file

# ---------- /api/warmup ----------


@api_router.get(
    "/warmup",
    tags=["meta"],
    summary="Manual warmup: FAISS + embeddings (+ LLM if configured)"
)
def warmup() -> Dict[str, Any]:
    def _check_faiss() -> Dict[str, Any]:
        idx = settings.faiss_index_path
        out = {"present": Path(idx).exists(), "path": str(idx)}
        if out["present"]:
            try:
                import faiss  # type: ignore
                index = faiss.read_index(str(idx))
                out["readable"] = True
                out["dimension"] =int(index.d)
            except Exception as e:
                out["readable"] = False
                out["error"] = str(e)
        return out

    def _warmup_embeddings() -> Dict[str, Any]:
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer("all-MiniLM-L6-v2")  # cache to disk
            return {"ok": True, "model": "all-MiniLM-L6-v2"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _check_llm() -> Dict[str, Any]:
        from .runtime_llm import STATE
        return {
            "ready": STATE.ready,
            "provider": STATE.provider,
            "last_error": STATE.last_error,
        }

    result: Dict[str, Any] = {
        "faiss": _check_faiss(),
        "embeddings": _warmup_embeddings(),
        "llm": _check_llm(),
    }

    return result
