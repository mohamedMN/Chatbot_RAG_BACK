# api/ask.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException

from .models import AskRequest, AskResponse, Hit

# --- Ton RAG existant ---
from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator
from rag.helpers import format_context_for_llm

# --- LLM (Ollama) optionnel via settings ---
try:
    from config.settings import create_ollama_llm, get_deepseek_coder_config
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

router = APIRouter()

# Singletons (chargés une seule fois par process)
_retriever = RAGRetriever()
_llm = None
_generator: Optional[RAGGenerator] = None


def _get_generator() -> RAGGenerator:
    global _generator, _llm
    if _generator is not None:
        return _generator

    # Essaie d’instancier un LLM Ollama si disponible
    if _HAS_OLLAMA:
        try:
            cfg = get_deepseek_coder_config()
            _llm = create_ollama_llm(**cfg)
        except Exception:
            # fallback sans LLM (le générateur gère la clarification/extractif)
            _llm = None

    _generator = RAGGenerator(llm=_llm)
    return _generator


@router.post(
    "",
    response_model=AskResponse,
    summary="Ask the chatbot (public)",
    tags=["ask"],
)
def ask(payload: AskRequest) -> AskResponse:
    """
    Récupération + génération RAG :
    - utilise ton RAGRetriever (FAISS local)
    - formate le contexte (numéroté) pour citations [#n]
    - génère la réponse via ton RAGGenerator (Ollama si dispo, sinon clarification/fallback)
    """
    try:
        # 1) Retrieval
        hits_raw: List[Dict[str, Any]] = _retriever.retrieve(
            query=payload.q,
            top_k=payload.k,
            min_score=payload.min_score,
        )
        hits: List[Hit] = [Hit(**h) for h in hits_raw]

        # 2) Contexte formaté en extraits numérotés [#n] (optionnel dans la réponse)
        context_str: Optional[str] = None
        if payload.include_context:
            context_str = format_context_for_llm(
                [h.model_dump() for h in hits], max_chars=1500
            )

        # 3) Génération
        generator = _get_generator()
        answer: str = generator.generate_answer(
            question=payload.q,
            context_hits=[h.model_dump() for h in hits],
            max_chars=1800,
        )

        return AskResponse(
            query=payload.q,
            top_k=payload.k,
            min_score=payload.min_score,
            hits=hits,
            context=context_str,
            answer=answer,
            debug={"hits_count": len(hits)},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ask failed: {e}")
