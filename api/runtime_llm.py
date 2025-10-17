# api/runtime_llm.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# Charge .env (sans override par défaut)
load_dotenv(override=False)

router = APIRouter(prefix="/llm", tags=["llm"])

# -------------------- Imports conditionnels --------------------

# Groq (cloud)
_GROQ_OK = False
try:
    from langchain_groq import ChatGroq
    _GROQ_OK = True
except Exception:
    pass

# Ollama (local)
_OLLAMA_OK = False
try:
    from langchain_ollama import ChatOllama as _OllamaChat
    _OLLAMA_OK = True
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama as _OllamaChat
        _OLLAMA_OK = True
    except Exception:
        pass


# -------------------- État runtime --------------------

class _RuntimeLLMState:
    provider: Optional[str] = None  # "groq" | "ollama"
    llm: Any = None
    ready: bool = False
    last_error: Optional[str] = None


STATE = _RuntimeLLMState()


# -------------------- Envs requis --------------------

REQUIRED_ENV: Dict[str, List[str]] = {
    "groq": ["GROQ_API_KEY", "GROQ_MODEL"],
    "ollama": ["OLLAMA_MODEL", "OLLAMA_HOST", "OLLAMA_NUM_CTX"],
}


def _missing_env(provider: str) -> List[str]:
    return [k for k in REQUIRED_ENV.get(provider, []) if not os.getenv(k)]


# -------------------- Builders --------------------

def _build_groq_llm() -> Any:
    """
    Groq natif via langchain-groq. Lit GROQ_API_KEY depuis l'env.
    """
    if not _GROQ_OK:
        raise RuntimeError("Installer Groq: pip install langchain-groq")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model, temperature=0.1, max_tokens=1200, timeout=60)
    _ = llm.invoke("ping")  # mini test
    return llm


def _build_ollama_llm() -> Any:
    """
    Ollama local (ChatOllama). Nécessite le daemon Ollama en cours d'exécution.
    """
    if not _OLLAMA_OK:
        raise RuntimeError(
            "Installer Ollama client Python: pip install langchain-ollama")
    model = os.getenv("OLLAMA_MODEL", "deepseek-coder:instruct")
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "6144"))
    llm = _OllamaChat(
        model=model,
        base_url=base_url,
        num_ctx=num_ctx,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        system="Tu es un expert en intégration et Software AG webMethods",
        stop=["```", "Human:", "User:", "\n\nHuman:", "\n\nUser:"],
    )
    _ = llm.invoke("ping")  # mini test
    return llm


# -------------------- Orchestrateur --------------------

def _start_provider(provider: str) -> None:
    provider = (provider or "").lower()
    if provider not in ("groq", "ollama"):
        raise HTTPException(
            status_code=400, detail="provider must be 'groq' or 'ollama'")

    missing = _missing_env(provider)
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing env for {provider}: {missing}")

    try:
        llm = _build_groq_llm() if provider == "groq" else _build_ollama_llm()

        # Sanity check
        test = llm.invoke("Démarrage: répondre 'OK' si prêt.")
        _ = getattr(test, "content", str(test))

        STATE.llm = llm
        STATE.provider = provider
        STATE.ready = True
        STATE.last_error = None
    except Exception as e:
        STATE.llm = None
        STATE.ready = False
        STATE.last_error = str(e)
        raise HTTPException(
            status_code=500, detail=f"LLM start failed ({provider}): {e}")


# -------------------- Schemas --------------------

class SelectBody(BaseModel):
    provider: str  # "groq" | "ollama"


# -------------------- Routes --------------------

@router.get("/status")
def status():
    env_default = os.getenv("LLM_PROVIDER", "").lower() or None
    env_status = None
    if env_default in ("groq", "ollama"):
        env_status = {"provider": env_default,
                      "missing": _missing_env(env_default)}
    return {
        "ready": STATE.ready,
        "active_provider": STATE.provider,
        "env_default": env_default,
        "env_status": env_status,
        "last_error": STATE.last_error,
        "hint": "POST /api/llm/select {\"provider\":\"groq\"|\"ollama\"} pour (re)lancer.",
    }


@router.post("/select")
def select_provider(body: SelectBody):
    """
    Client choisit cloud (groq) ou local (ollama) → on vérifie les envs et on instancie le LLM.
    """
    _start_provider(body.provider)
    return {"ok": True, "provider": STATE.provider, "ready": STATE.ready}


@router.post("/complete")
def complete(prompt: str = Query(..., description="Texte de la requête")):
    """
    Exécute une complétion via le provider actif.
    """
    if not STATE.ready or not STATE.llm:
        raise HTTPException(
            status_code=400, detail="LLM non initialisé. Appelle d'abord POST /api/llm/select.")
    try:
        msg = STATE.llm.invoke(prompt)
        text = getattr(msg, "content", str(msg))
        return {"provider": STATE.provider, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {e}")


def get_active_llm():
    """Returns the currently active LLM instance (or None)."""
    return STATE.llm
