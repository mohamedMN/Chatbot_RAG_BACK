from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re

from indexing.faiss_indexer import FAISSIndexer
from core.embedder_selector import get_embedder

_WORD = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_]+")


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _kw_overlap_score(query: str, text: str) -> float:
    qk = set(_tokens(query))
    if not qk:
        return 0.0
    tk = set(_tokens(text))
    if not tk:
        return 0.0
    inter = len(qk.intersection(tk))
    return inter / max(1, len(qk))


class RAGOllamaRetriever:
    """
    Fast retriever:
      - FAISS vector search
      - light hybrid score (vector + keyword overlap)
      - no LLM re-rank
    """

    def __init__(self, faiss_indexer: Optional[FAISSIndexer] = None):
        self.faiss = faiss_indexer or FAISSIndexer()
        self.embedder = get_embedder()  # LM Studio
        self.runtime: Optional[Dict[str, Any]] = None
        # weights
        self.w_sim = 0.7
        self.w_kw = 0.3
        self.min_sim = 0.25

    def load_runtime(self) -> bool:
        if not self.faiss.load_index():
            return False
        self.runtime = self.faiss.get_runtime_data()
        return True

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.30) -> List[Dict[str, Any]]:
        if not self.runtime:
            if not self.load_runtime():
                return []

        rd = self.runtime
        index = rd["index"]
        idmap = rd["idmap"]
        meta = rd.get("meta", {})

        q_vec = self.embedder.embed([query])[0]
        qv = np.asarray(q_vec, dtype="float32").reshape(1, -1)

        # Dimension safety (prevents faiss AssertionError)
        faiss_d = getattr(index, "d", None)
        if faiss_d is not None and int(qv.shape[1]) != int(faiss_d):
            raise ValueError(
                f"Embedding dimension mismatch: query={int(qv.shape[1])} vs index={faiss_d}. "
                "Rebuild the FAISS index with the current embedding model."
            )

        # Normalize if index metric is IP and vectors are expected normalized
        use_ip = str(meta.get("metric", "")).lower() == "ip"
        if use_ip and bool(meta.get("normalized", True)):
            n = float(np.linalg.norm(qv))
            if n > 0:
                qv = qv / n

        # widen then re-rank
        widen = max(top_k * 3, top_k)
        scores, idxs = index.search(qv, int(widen))
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        id_to_row = {int(i): r for r, i in enumerate(idmap["ids"])}
        cands: List[Dict[str, Any]] = []
        for s, fid in zip(scores, idxs):
            if fid == -1:
                continue
            row = id_to_row.get(int(fid))
            if row is None or s < self.min_sim:
                continue
            content = str(idmap["content"][row])
            subject = str(idmap["subject"][row])
            source = str(idmap["source"][row])

            kw = _kw_overlap_score(query, subject + " " + content)
            final = self.w_sim * float(s) + self.w_kw * float(kw)

            cands.append({
                "score": float(s),
                "kw": float(kw),
                "final_score": float(final),
                "id": int(fid),
                "ordinal": int(idmap["ordinal"][row]),
                "content": content,
                "subject": subject,
                "source": source,
            })

        cands.sort(key=lambda h: (-h["final_score"], h["ordinal"]))
        out = [h for h in cands if h["final_score"] >= min_score]
        return out[:top_k]

    @staticmethod
    def build_context(hits: List[Dict[str, Any]], chunk_chars: int, ctx_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
        ctx_parts: List[str] = []
        used: List[Dict[str, Any]] = []
        budget = max(0, int(ctx_chars))
        for h in hits:
            if budget <= 0:
                break
            text = (h.get("content") or "")[: max(0, int(chunk_chars))]
            if not text:
                continue
            snippet = f"[{h.get('source','?')}] {text}".strip()
            if len(snippet) > budget:
                snippet = snippet[:budget]
            ctx_parts.append(snippet)
            used.append(h)
            budget -= len(snippet) + 1
        return "\n\n".join(ctx_parts), used
