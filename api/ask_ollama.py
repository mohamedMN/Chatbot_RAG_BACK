# backend/api/ask_ollama.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from rag.retriever_ollama import RAGOllamaRetriever
from llm.lmstudio_client import chat_once  # ← swap

router = APIRouter(prefix="/llm/ask", tags=["llm-ollama"])


class AskBodyOllama(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = 5
    min_score: float = 0.30
    ctx_chars: int = 1000
    chunk_chars: int = 320
    answer_words: int = 80
    include_context: bool = True
    workspace_id: Optional[str] = None    # ← NEW


def ask_with_ollama(body: AskBodyOllama) -> Dict[str, Any]:
    retr = RAGOllamaRetriever(ws_id=body.workspace_id)   # ← use ws
    hits = retr.retrieve(body.query, top_k=body.top_k,
                         min_score=body.min_score)
    context, used = retr.build_context(
        hits, chunk_chars=body.chunk_chars, ctx_chars=body.ctx_chars)

    instruction = (
        "Réponds de manière brève et précise (max {w} mots). "
        "Si tu n'es pas sûr, dis-le clairement. "
        "Langue: même que la question.\n\n"
        "Contexte:\n{ctx}\n\nQuestion:\n{q}\n\nRéponse courte:"
    ).format(w=body.answer_words, ctx=context if body.include_context else "", q=body.query)

    try:
        answer = chat_once(
            prompt=instruction,
            temperature=0.2,
            num_ctx=2048,
            max_tokens=128,   # LM Studio -> max_tokens
            timeout=60.0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"LM Studio call failed: {e}")

    return {
        "answer": answer,
        "hits": used,
        "used_chars": len(context),
        "total_hits": len(hits),
    }
