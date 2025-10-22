# rag/retriever.py
from __future__ import annotations

from typing import Dict, List, Any, Optional
import re
import numpy as np

from config.settings import settings
from core.embedder import EmbeddingModel
from indexing.faiss_indexer import FAISSIndexer
from rag.helpers import norm, extract_keywords, deduplicate_hits

# --- petits utilitaires locaux ---
_WORD = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_]+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _keyword_overlap_score(query_kws: List[str], text: str) -> float:
    if not query_kws:
        return 0.0
    toks = set(_tokenize(text))
    if not toks:
        return 0.0
    inter = sum(1 for k in query_kws if k in toks)
    return inter / max(1, len(set(query_kws)))


def _title_boost(subject: str, query: str) -> float:
    s = (subject or "").lower()
    q = (query or "").lower()
    boost = 0.0
    for kw in ("webmethods", "esb", "integration server", "api gateway", "universal messaging"):
        if kw in s and kw in q:
            boost += 0.12
    if any(p in q for p in ["qu'est-ce", "c'est quoi", "what is", "définition", "definition"]):
        if any(p in s for p in ["introduction", "overview", "présentation", "qu'est-ce", "définition"]):
            boost += 0.15
    return boost


def _policy_penalty(text: str, query: str) -> float:
    """Pénalise les extraits 'policy/safety' si la question ne demande PAS ça."""
    t = (text or "").lower()
    q = (query or "").lower()
    if any(w in q for w in ["policy", "sécurité", "compliance", "confidentiel", "sécuriser"]):
        return 0.0
    bad_markers = ("refuserai", "dangereuses", "illégales",
                   "assistant", "je ne peux pas", "mes règles")
    return 0.25 if any(m in t for m in bad_markers) else 0.0


def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_: float = 0.7) -> List[int]:
    """MMR classique sur vecteurs **déjà normalisés**."""
    n = cand_vecs.shape[0]
    if n == 0:
        return []
    sim_q = (cand_vecs @ query_vec.T).reshape(-1)  # (n,)
    selected: List[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        if not selected:
            # indice relatif à remaining → on convertit ensuite
            pos = int(np.argmax(sim_q[remaining]))
            selected.append(remaining.pop(pos))
            continue
        # max similarité à l'ensemble sélectionné
        max_sim_to_sel = []
        for idx in remaining:
            sims = cand_vecs[idx:idx+1] @ cand_vecs[selected].T
            max_sim_to_sel.append(float(np.max(sims)))
        mmr_score = lambda_ * sim_q[remaining] - \
            (1 - lambda_) * np.array(max_sim_to_sel)
        pick_pos = int(np.argmax(mmr_score))
        selected.append(remaining.pop(pick_pos))
    return selected


class RAGRetriever:
    """Retriever robuste : vector search élargi + hybrid score + MMR + LLM re-rank (optionnel)."""

    def __init__(self, faiss_indexer: Optional[FAISSIndexer] = None):
        self.faiss_indexer = faiss_indexer or FAISSIndexer()
        self.embedding_model = EmbeddingModel()
        self.runtime_data: Optional[Dict[str, Any]] = None
        self.cfg = settings.get_retrieval_config()

        # pondérations / seuils
        self.w_sim = float(self.cfg.get("w_sim", 0.6))
        self.w_kw = float(self.cfg.get("w_kw", 0.25))
        self.w_tit = float(self.cfg.get("w_title", 0.15))
        self.min_final = 0.0
        self.min_cosine = 0.0
        self.enable_llm_rerank = bool(self.cfg.get("llm_rerank", True))
        self.llm_rerank_k = int(self.cfg.get("llm_rerank_k", 10))
        self.simple_mode = bool(self.cfg.get("simple_mode", False))
    # -------- lifecycle --------
    def load_runtime(self) -> bool:
        try:
            if not self.faiss_indexer.load_index():
                print("Failed to load FAISS index")
                return False
            self.runtime_data = self.faiss_indexer.get_runtime_data() or {}
            # sanity check minimal keys
            for key in ("index", "idmap", "meta"):
                if key not in self.runtime_data:
                    self.runtime_data[key] = {}  # évite KeyError
            return True
        except Exception as e:
            print(f"Error loading runtime data: {e}")
            return False

    # -------- main API --------
    def retrieve(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self.runtime_data and not self.load_runtime():
            return []

        k = int(top_k or self.cfg.get("top_k", 5))
        threshold = float(min_score or self.cfg.get("min_score", 0.30))
        threshold = max(threshold, self.min_final)

        # 1) Vector search élargi (x3)
        initial = self._vector_search(query, top_k=k)
        if not initial:
            return []

        # 2) Hybrid scoring
        scored = self._hybrid_score(initial, query)
        if not scored:
            return []

       

        
        

        # 5) Filtre final
        for h in initial:
            h["final_score"] = float(h.get("score", 0.0))

        # 6) Dé-duplication douce
        initial.sort(key=lambda x: x["final_score"], reverse=True)
        return initial[:k]

    # -------- steps --------
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        rd = self.runtime_data or {}
        index = rd.get("index")
        idmap = rd.get("idmap") or {}
        meta = rd.get("meta") or {}

        if index is None or not idmap:
            return []

        ids = list(map(int, idmap.get("ids", [])))
        ordinal = idmap.get("ordinal", [])
        content = idmap.get("content", [])
        subject = idmap.get("subject", [])
        source = idmap.get("source", [])

        if not ids:
            return []

        qv = np.array(self.embedding_model.embed([query])[
                      0], dtype="float32").reshape(1, -1)
        use_ip = str(meta.get("metric", "")).lower() == "ip"
        if use_ip and self.cfg.get("normalize", True):
            qv = norm(qv)

        scores, indices = index.search(qv, int(top_k))
        scores, indices = scores[0], indices[0]

        hits: List[Dict[str, Any]] = []
        id_to_row = {int(i): r for r, i in enumerate(ids)}
        for sc, idx in zip(scores, indices):
            if int(idx) == -1:
                continue
            row = id_to_row.get(int(idx))
            if row is None:
                continue
            hits.append({
                "score": float(sc),
                "id": int(idx),
                "ordinal": int(ordinal[row]) if row < len(ordinal) else row,
                "content": str(content[row]) if row < len(content) else "",
                "subject": str(subject[row]) if row < len(subject) else "",
                "source": str(source[row]) if row < len(source) else "",
            })
        return hits

    def _hybrid_score(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if self.simple_mode:
            out = []
            for h in candidates:
                hh = dict(h)
                hh["kw_score"] = 0.0
                hh["title_boost"] = 0.0
                hh["penalty"] = 0.0
                hh["final_score"] = float(h.get("score", 0.0))
                out.append(hh)
            out.sort(key=lambda x: x["final_score"], reverse=True)
            return out
        kws = [k.lower() for k in extract_keywords(query)]
        out: List[Dict[str, Any]] = []

        for h in candidates:
            content = h.get("content", "") or ""
            subject = h.get("subject", "") or ""
            sim = float(h.get("score", 0.0))

            # Si pas de mots-clés, on filtre très bas sur la cosine
            if sim < self.min_cosine and not kws:
                continue

            kw_s = _keyword_overlap_score(kws, f"{subject} {content}")
            tit_b = _title_boost(subject, query)
            pen = _policy_penalty(f"{subject} {content}", query)

            final = self.w_sim * sim + self.w_kw * kw_s + self.w_tit * tit_b
            final = max(0.0, final - pen)

            hh = dict(h)
            hh.update({
                "kw_score": float(kw_s),
                "title_boost": float(tit_b),
                "penalty": float(pen),
                "final_score": float(final),
            })
            out.append(hh)

        out.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return out

    def _mmr_select(self, query: str, ranked: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        texts = [
            f"{h.get('subject','')} {h.get('content','')}" for h in ranked]
        if not texts:
            return []
        M = np.array(self.embedding_model.embed(texts), dtype="float32")
        qv = np.array(self.embedding_model.embed([query])[
                      0], dtype="float32").reshape(1, -1)
        M = norm(M)
        qv = norm(qv)
        pick_idxs = _mmr(qv, M, k=k, lambda_=0.7)
        return [ranked[i] for i in pick_idxs if 0 <= i < len(ranked)]

    def _llm_rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Réordonne **jusqu’à** llm_rerank_k premiers éléments avec l’LLM, puis recolle la queue telle quelle."""
        try:
            from api.runtime_llm import STATE as RUNTIME_LLM_STATE
            llm = RUNTIME_LLM_STATE.llm if getattr(
                RUNTIME_LLM_STATE, "ready", False) else None
        except Exception:
            llm = None

        if not llm or not hits:
            return hits

        k = max(1, min(int(getattr(self, "llm_rerank_k", 10) or 10), len(hits)))
        head = hits[:k]
        tail = hits[k:]

        bullets = []
        for i, h in enumerate(head, 1):
            txt = (h.get("content") or "").replace("\n", " ").strip()[:600]
            bullets.append(f"{i}. {txt}")

        prompt = (
            "Tu es un juge de pertinence. Note chaque extrait entre 0 et 4 par rapport à la question.\n"
            "0 = hors-sujet, 1 = faible, 2 = moyen, 3 = bon, 4 = très pertinent.\n"
            "Réponds UNIQUEMENT sous la forme: 1:3, 2:0, 3:2, ...\n\n"
            f"Question: {query}\n\nExtraits:\n" +
            "\n".join(bullets) + "\n\nNotes:"
        )

        try:
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(
                resp)) if resp is not None else ""
        except Exception:
            text = ""

        # Parsing tolérant
        scores: Dict[int, int] = {}
        for m in re.finditer(r"(\d+)\s*[:\-]\s*([0-4])", text):
            idx = int(m.group(1))
            val = int(m.group(2))
            if 1 <= idx <= len(head):
                scores[idx - 1] = val

        # fallback: suite de chiffres "3 2 0 ..."
        if not scores:
            raw_nums = re.findall(r"\b[0-4]\b", text)
            for pos, s in enumerate(raw_nums[:len(head)]):
                scores[pos] = int(s)

        # combinaison pour le head
        new_head: List[Dict[str, Any]] = []
        for i, h in enumerate(head):
            llm_s = int(scores.get(i, 0))
            base = float(h.get("final_score", h.get("score", 0.0)) or 0.0)
            comb = 0.6 * base + 0.4 * (llm_s / 4.0)
            hh = dict(h)
            hh["llm_score"] = llm_s
            hh["final_score"] = float(comb)
            new_head.append(hh)

        new_head.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return new_head + tail

    # -------- config & stats --------
    def update_config(self, **kwargs) -> None:
        """
        Update runtime weights/thresholds without restarting.
        Allowed keys: w_sim, w_kw, w_tit, min_cosine, min_final, llm_rerank, llm_rerank_k
        """
        allowed = {"w_sim", "w_kw", "w_tit", "min_cosine",
                   "min_final", "llm_rerank", "llm_rerank_k", "simple_mode"}
        
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            elif k == "simple_mode":
                self.simple_mode = bool(v)
            if k == "llm_rerank":
                self.enable_llm_rerank = bool(v)
            elif k == "llm_rerank_k":
                try:
                    self.llm_rerank_k = max(1, int(v))
                except Exception:
                    pass
            elif k in {"w_sim", "w_kw", "w_tit", "min_cosine", "min_final"}:
                try:
                    setattr(self, k if k != "w_title" else "w_tit", float(v))
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        rd = getattr(self, "runtime_data", None) or {}
        meta = rd.get("meta", {}) or {}
        return {
            "index_loaded": bool(self.runtime_data),
            "metric": meta.get("metric"),
            "vectors": int(meta.get("vectors", 0) or 0),
            "dim": int(meta.get("dim", 0) or 0),
            "normalize": bool(self.cfg.get("normalize", True)),
            "weights": {
                "w_sim": float(self.w_sim),
                "w_kw": float(self.w_kw),
                "w_title": float(self.w_tit),
            },
            "thresholds": {
                "min_cosine": float(self.min_cosine),
                "min_final": float(self.min_final),
            },
            "llm_rerank": {
                "enabled": bool(self.enable_llm_rerank),
                "k": int(self.llm_rerank_k),
            },
        }
