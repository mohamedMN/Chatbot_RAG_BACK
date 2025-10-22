# api/runtime_llm.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# ✅ Import du wrapper Langchain
from utils.lmstudio_chat import LMStudioChat

load_dotenv(override=False)

router = APIRouter(prefix="/llm", tags=["llm"])

# -------------------- Imports conditionnels --------------------

_GROQ_OK = False
try:
    from langchain_groq import ChatGroq
    _GROQ_OK = True
except Exception:
    pass

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
    provider: Optional[str] = None
    llm: Any = None  # Runnable Langchain
    ready: bool = False
    last_error: Optional[str] = None


STATE = _RuntimeLLMState()


# -------------------- Envs requis --------------------

REQUIRED_ENV: Dict[str, List[str]] = {
    "groq": ["GROQ_API_KEY", "GROQ_MODEL"],
    # lmstudio n'a pas d'envs obligatoires (defaults OK)
}


def _missing_env(provider: str) -> List[str]:
    return [k for k in REQUIRED_ENV.get(provider, []) if not os.getenv(k)]


# -------------------- Builders --------------------

def _build_groq_llm() -> Any:
    """Groq cloud via langchain-groq."""
    if not _GROQ_OK:
        raise RuntimeError("Installer Groq: pip install langchain-groq")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1200,
        timeout=60
    )

    # Test de connexion
    from langchain_core.messages import HumanMessage
    _ = llm.invoke([HumanMessage(content="ping")])

    return llm


def _build_lmstudio_llm() -> Any:
    """LM Studio local via wrapper Langchain."""
    model = os.getenv(
        "LMSTUDIO_MODEL",
        "lmstudio-community/Meta-Llama-3-8B-Instruct"
    )

    llm = LMStudioChat(
        model=model,
        temperature=0.2,
        max_tokens=500,
        timeout=60.0,
    )

    # Test de connexion
    from langchain_core.messages import HumanMessage
    try:
        response = llm.invoke([HumanMessage(content="test")])
        if not response or not response.content:
            raise RuntimeError("LM Studio retourné réponse vide")
    except Exception as e:
        raise RuntimeError(
            f"LM Studio non accessible. Vérifiez que le serveur tourne sur "
            f"{os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234')}. "
            f"Erreur: {e}"
        )

    return llm


def _build_ollama_llm() -> Any:
    """Ollama local (optionnel)."""
    if not _OLLAMA_OK:
        raise RuntimeError("Installer Ollama: pip install langchain-ollama")

    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

    llm = _OllamaChat(
        model=model,
        base_url=base_url,
        num_ctx=num_ctx,
        temperature=0.2,
        timeout=60.0,
    )

    # Test
    from langchain_core.messages import HumanMessage
    _ = llm.invoke([HumanMessage(content="ping")])

    return llm


# -------------------- Orchestrateur --------------------

def _start_provider(provider: str) -> None:
    """Démarre le provider LLM choisi."""
    prov = (provider or "").lower()

    # ✅ Validation
    if prov not in ("groq", "lmstudio", "ollama"):
        raise HTTPException(
            status_code=400,
            detail=f"Provider invalide: '{prov}'. Doit être 'groq', 'lmstudio' ou 'ollama'"
        )

    # Vérifier les envs requis
    missing = _missing_env(prov)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Variables d'environnement manquantes pour {prov}: {missing}"
        )

    try:
        # ✅ Builder selon le provider
        if prov == "groq":
            llm = _build_groq_llm()
        elif prov == "lmstudio":
            llm = _build_lmstudio_llm()
        elif prov == "ollama":
            llm = _build_ollama_llm()
        else:
            raise ValueError(f"Provider non supporté: {prov}")

        # ✅ Succès
        STATE.llm = llm
        STATE.provider = prov
        STATE.ready = True
        STATE.last_error = None

        print(f"✅ LLM initialisé: {prov} ({type(llm).__name__})")

    except HTTPException:
        raise
    except Exception as e:
        STATE.llm = None
        STATE.ready = False
        STATE.last_error = str(e)

        print(f"❌ Échec initialisation {prov}: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Échec démarrage LLM ({prov}): {e}"
        )


# -------------------- Schemas --------------------

class SelectBody(BaseModel):
    provider: str  # "groq" | "lmstudio" | "ollama"


# -------------------- Routes --------------------

@router.get("/status")
def status():
    """Status du LLM actif et configuration disponible."""
    env_default = os.getenv("LLM_PROVIDER", "").lower() or None

    # Statut pour chaque provider
    providers_status = {
        "groq": {
            "available": _GROQ_OK,
            "missing_env": _missing_env("groq") if _GROQ_OK else ["langchain-groq non installé"],
            "configured": _GROQ_OK and not _missing_env("groq"),
        },
        "lmstudio": {
            "available": True,  # Toujours disponible (wrapper custom)
            "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            "model": os.getenv("LMSTUDIO_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct"),
            "configured": True,
        },
        "ollama": {
            "available": _OLLAMA_OK,
            "missing_env": [] if _OLLAMA_OK else ["langchain-ollama non installé"],
            "configured": _OLLAMA_OK,
        },
    }

    return {
        "ready": STATE.ready,
        "active_provider": STATE.provider,
        "env_default": env_default,
        "last_error": STATE.last_error,
        "providers": providers_status,
        "hint": "POST /api/llm/select {\"provider\":\"groq\"|\"lmstudio\"|\"ollama\"} pour (re)lancer.",
    }


@router.post("/select")
def select_provider(body: SelectBody):
    """
    ✅ Sélectionne et démarre le provider LLM.
    
    Exemples:
    - {"provider": "lmstudio"} → LM Studio local
    - {"provider": "groq"} → Groq cloud (nécessite GROQ_API_KEY)
    - {"provider": "ollama"} → Ollama local
    """
    provider=body.provider.lower().strip()
    if( provider== "ollama"):
        provider = "lmstudio"
    _start_provider(provider)

    return {
        "ok": True,
        "provider": STATE.provider,
        "ready": STATE.ready,
        "llm_type": type(STATE.llm).__name__ if STATE.llm else None,
    }


@router.post("/complete")
def complete(prompt: str = Query(..., description="Texte de la requête")):
    """Exécute une complétion avec le provider actif."""
    if not STATE.ready or not STATE.llm:
        raise HTTPException(
            status_code=400,
            detail="LLM non initialisé. Appelez d'abord POST /api/llm/select"
        )

    try:
        from langchain_core.messages import HumanMessage
        msg = STATE.llm.invoke([HumanMessage(content=prompt)])
        text = getattr(msg, "content", str(msg))

        return {
            "provider": STATE.provider,
            "text": text,
            "length": len(text),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Complétion échouée: {e}"
        )


# -------------------- Helper pour RAG --------------------

def get_active_llm():
    """
    ✅ Retourne le LLM actif (Runnable Langchain).
    
    Auto-initialise LMStudio si rien n'est configuré.
    Utilisé par RAGGenerator.
    """
    if not STATE.ready or not STATE.llm:
        # Auto-init avec LMStudio par défaut
        try:
            print("⚠️  Aucun LLM actif, initialisation auto de LMStudio...")
            _start_provider("lmstudio")
        except Exception as e:
            print(f"⚠️  Auto-init LMStudio échoué: {e}")
            print("   Le système fonctionnera en mode extractif")
            return None

    return STATE.llm
