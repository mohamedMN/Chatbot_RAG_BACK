# api/ask_groq.py
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.runtime_llm import get_active_llm
from rag.retriever_groq import RAGGroqRetriever

router = APIRouter(prefix="/llm/ask", tags=["ask-llm"])


class AskBody(BaseModel):
    # Accept legacy keys: "question" or "prompt" → coerce to "query"
    model_config = ConfigDict(extra="ignore")
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = 6
    min_score: Optional[float] = 0.30
    ctx_chars: Optional[int] = 1600
    answer_words: Optional[int] = 140

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, v):
        if isinstance(v, dict) and "query" not in v:
            if "question" in v:
                v["query"] = v.pop("question")
            elif "prompt" in v:
                v["query"] = v.pop("prompt")
        return v


def _pack_context(hits: List[Dict[str, Any]], budget: int) -> str:
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        b = f"[{i}] {h.get('subject','')}\n{h.get('content','')}\n"
        if used + len(b) > budget:
            break
        parts.append(b)
        used += len(b)
    return "".join(parts).strip()


@router.post("/groq")
def ask_groq(body: AskBody):
    llm = get_active_llm()
    if not llm:
        raise HTTPException(
            400, "Groq not selected. POST /api/llm/select {'provider':'groq'} first."
        )

    q = (body.query or "").strip()
    if not q:
        raise HTTPException(422, "empty query")

    top_k = int(body.top_k or 6)
    min_score = float(body.min_score or 0.30)
    ctx_chars = int(body.ctx_chars or 1600)
    answer_words = int(body.answer_words or 140)

    retr = RAGGroqRetriever()
    hits = retr.retrieve(q, top_k=top_k, min_score=min_score) or []
    context = _pack_context(hits, ctx_chars)

    prompt = (
        "Answer using ONLY the following context. If missing, say so.\n"
        f"Limit to ~{answer_words} words.\n\n"
        f"Question: {q}\n\nContext:\n{context}\n\nAnswer:"
    )

    t0 = time.perf_counter()
    msg = llm.invoke(prompt)
    text = getattr(msg, "content", str(msg))
    tgen = (time.perf_counter() - t0) * 1000

    return {
        "ok": True,
        "provider": "groq",
        "answer": text,
        "hits": hits,
        "timing_ms": {"generation": round(tgen, 1)},
    }
