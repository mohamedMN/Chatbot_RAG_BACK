from __future__ import annotations

import logging
from typing import Any, Dict
from pathlib import Path

from fastapi import APIRouter
from .workspaces import router as ws_router

from config.settings import settings
try:
    from config.settings import check_ollama_status, get_deepseek_coder_config, create_ollama_llm
    _OLLAMA_HELPERS = True
except Exception:
    _OLLAMA_HELPERS = False

# Sub-routers
from .auth import router as auth_router     # public
from .ask import router as ask_router      # public
from .admin import router as admin_router   # protected via x-admin-key

api_router = APIRouter()
log = logging.getLogger("chatbot_rag.api")

api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(ask_router,  prefix="/ask",  tags=["ask"])
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
api_router.include_router(ws_router)

# ---------- /api/warmup ----------


@api_router.get(
    "/warmup",
    tags=["meta"],
    summary="Manual warmup: FAISS + embeddings (+ Ollama if available)"
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
                out["dimension"] = int(index.d)
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

    def _warmup_ollama() -> Dict[str, Any]:
        if not _OLLAMA_HELPERS:
            return {"ok": False, "error": "Ollama helpers not available in config.settings"}
        try:
            cfg = get_deepseek_coder_config()
            llm = create_ollama_llm(**cfg)
            _ = llm.invoke("ping")  # tiny prompt
            return {"ok": True, "model": cfg.get("model")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    result: Dict[str, Any] = {
        "faiss": _check_faiss(),
        "embeddings": _warmup_embeddings(),
    }
    if _OLLAMA_HELPERS:
        try:
            result["ollama"] = _warmup_ollama()
            result["ollama_tags"] = check_ollama_status()
        except Exception as e:
            result["ollama"] = {"ok": False, "error": str(e)}
    return result
