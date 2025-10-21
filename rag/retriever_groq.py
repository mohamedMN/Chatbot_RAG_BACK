# rag/retriever_groq.py
from __future__ import annotations
import time
import re
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from config.settings import settings
from core.embedder_selector import get_embedder
from indexing.faiss_indexer import FAISSIndexer

log = logging.getLogger("rag.retriever.groq")

# ----------------- small local helpers -----------------


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n


def _extract_keywords(text: str) -> List[str]:
    import re
    toks = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_]+", (text or "").lower())
    stop = {"et", "ou", "les", "des", "pour", "avec", "dans", "de", "du", "la", "le",
            "un", "une", "en", "sur", "a", "au", "aux", "the", "and", "of", "to", "in"}
    toks = [t for t in toks if len(t) > 2 and t not in stop]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:14]


def _kw_overlap(keywords: List[str], text: str) -> float:
    if not keywords:
        return 0.0
    t = (text or "").lower()
    hits = sum(1 for k in keywords if k in t)
    return hits / max(1, len(set(keywords)))


def _trim(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return (cut or s[:n]).rstrip() + "…"


def _dedup(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for h in hits:
        key = (h.get("source"), str(h.get("ordinal")))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
# -------------------------------------------------------


class RAGGroqRetriever:
    def __init__(self, faiss_indexer: Optional[FAISSIndexer] = None):
        self.faiss = faiss_indexer or FAISSIndexer()  # default to global if not provided
        self.embedder = get_embedder()
        self.runtime: Optional[Dict[str, Any]] = None
        self.w_sim = 0.7
        self.w_kw = 0.3
        self.min_sim = 0.25

    def load_runtime(self) -> bool:
        if not self.faiss.load_index():
            return False
        self.runtime = self.faiss.get_runtime_data()
        return True

    def update_config(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _llm_rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enable_llm_rerank or not hits:
            return hits
        try:
            from api.runtime_llm import STATE as RUNTIME
            llm = RUNTIME.llm if (
                RUNTIME.ready and RUNTIME.provider == "groq") else None
        except Exception:
            llm = None
        if not llm:
            return hits

        bullets = []
        cut = min(len(hits), self.llm_rerank_k)
        for i in range(cut):
            txt = _trim(hits[i].get("content", "").replace("\n", " "), 600)
            bullets.append(f"{i+1}. {txt}")

        prompt = (
            "Note la pertinence de chaque extrait par rapport à la question (0 à 4).\n"
            "Forme stricte: 1:3, 2:1, 3:4 ...\n\n"
            f"Question: {query}\n\nExtraits:\n" +
            "\n".join(bullets) + "\n\nNotes:"
        )
        try:
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
        except Exception:
            return hits

        scores: Dict[int, int] = {}
        for m in re.finditer(r"(\d+)\s*[:\-]\s*([0-4])", text):
            idx = int(m.group(1)) - 1
            val = int(m.group(2))
            if 0 <= idx < cut:
                scores[idx] = val

        out = []
        for i, h in enumerate(hits[:cut]):
            llm_s = scores.get(i, 0)
            comb = 0.65*float(h.get("final_score", 0.0)) + 0.35*(llm_s/4.0)
            hh = dict(h)
            hh["llm_score"] = llm_s
            hh["final_score"] = float(comb)
            out.append(hh)
        out += hits[cut:]
        out.sort(key=lambda x: (-x["final_score"], x["ordinal"]))
        return out

    def retrieve(
        self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        if not (self.runtime_data or self.load_runtime()):
            return []

        k_user = int(top_k) if top_k is not None else self.top_k_default
        k_search = max(6, k_user * 3)
        thr = float(
            min_score) if min_score is not None else self.min_score_default

        rd = self.runtime_data
        index, idmap, meta = rd["index"], rd["idmap"], rd.get("meta", {})
        use_ip = str(meta.get("metric", "")).lower() == "ip"

        t0 = time.perf_counter()
        qv = np.asarray(self.embedding_model.embed([query])[
                        0], dtype="float32").reshape(1, -1)
        if use_ip and self.normalize:
            qv = _norm(qv)

        scores, ids = index.search(qv, int(k_search))
        scores, ids = scores[0].tolist(), ids[0].tolist()
        pos_by_id = {int(fid): int(i) for i, fid in enumerate(idmap["ids"])}

        kws = _extract_keywords(query)
        cands: List[Dict[str, Any]] = []

        for s, fid in zip(scores, ids):
            if fid == -1:
                continue
            pos = pos_by_id.get(int(fid))
            if pos is None:
                continue
            subject = str(idmap["subject"][pos])
            content = str(idmap["content"][pos])
            src = str(idmap["source"][pos])
            ord_ = int(idmap["ordinal"][pos])

            kw_s = _kw_overlap(kws, f"{subject}\n{content}")
            final = 0.75 * float(s) + 0.25 * kw_s
            if final < thr:
                continue

            cands.append({
                "id": int(fid),
                "score": float(s),
                "final_score": float(final),
                "ordinal": ord_,
                "subject": subject,
                "source": src,
                "content": _trim(content, self.chunk_chars),
                "kw_score": kw_s,
            })

        cands = _dedup(cands)
        cands.sort(key=lambda h: (-h["final_score"], h["ordinal"]))

        # optional Groq re-rank
        reranked = self._llm_rerank(query, cands[: max(k_user*2, 10)])
        out = reranked[:k_user]
        log.debug("Groq retrieve: k=%d -> %d (%.1f ms)", k_user,
                  len(out), (time.perf_counter()-t0)*1000)
        return out
