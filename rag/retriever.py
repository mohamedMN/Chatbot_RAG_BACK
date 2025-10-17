from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import re
import numpy as np

from config.settings import settings
from core.embedder import EmbeddingModel
from indexing.faiss_indexer import FAISSIndexer
from rag.helpers import (
    norm,
    extract_keywords,
    deduplicate_hits,
)

# --- petits utilitaires ---
_WORD = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_]+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _keyword_overlap_score(query_kws: List[str], text: str) -> float:
    if not query_kws:
        return 0.0
    toks = set(_tokenize(text))
    if not toks:
        return 0.0
    inter = len([k for k in query_kws if k in toks])
    return inter / max(1, len(set(query_kws)))


def _title_boost(subject: str, query: str) -> float:
    s = (subject or "").lower()
    q = (query or "").lower()
    boost = 0.0
    # exact terms likely in “overview/definition” questions
    for kw in ("webmethods", "esb", "integration server", "api gateway", "universal messaging"):
        if kw in s and kw in q:
            boost += 0.12
    # definition patterns
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
    # heuristique : passages qui ressemblent à des règles d’assistant
    bad_markers = (
        "refuserai", "dangereuses", "illégales", "assistant", "je ne peux pas", "mes règles"
    )
    return 0.25 if any(m in t for m in bad_markers) else 0.0


def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_: float = 0.7) -> List[int]:
    """Classic MMR to promote diversity on already normalized vectors."""
    n = cand_vecs.shape[0]
    if n == 0:
        return []
    # similarity to query (cosine = dot since we normalize)
    sim_q = (cand_vecs @ query_vec.T).reshape(-1)  # shape (n,)

    selected: List[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        if not selected:
            # pick best to query
            i = int(np.argmax(sim_q[remaining]))
            selected.append(remaining.pop(i))
            continue

        # compute max similarity to already selected (diversity term)
        max_sim_to_sel = []
        for idx in remaining:
            sims = cand_vecs[idx:idx+1] @ cand_vecs[selected].T
            max_sim_to_sel.append(float(np.max(sims)))
        max_sim_to_sel = np.array(max_sim_to_sel)

        mmr_score = lambda_ * sim_q[remaining] - (1 - lambda_) * max_sim_to_sel
        pick = int(np.argmax(mmr_score))
        selected.append(remaining.pop(pick))
    return selected


class RAGRetriever:
    """Retriever robuste : vector search élargi + hybrid score + MMR + LLM re-rank (optionnel)."""

    def __init__(self, faiss_indexer: Optional[FAISSIndexer] = None):
        self.faiss_indexer = faiss_indexer or FAISSIndexer()
        self.embedding_model = EmbeddingModel()
        self.runtime_data: Optional[Dict[str, Any]] = None
        self.cfg = settings.get_retrieval_config()
        # pondérations par défaut (ajustables via settings)
        self.w_sim = float(self.cfg.get("w_sim", 0.6))
        self.w_kw = float(self.cfg.get("w_kw", 0.25))
        self.w_tit = float(self.cfg.get("w_title", 0.15))
        self.min_cosine = float(self.cfg.get("min_cosine", 0.28))  # garde-fou
        self.min_final = float(self.cfg.get("min_final", 0.35))
        self.enable_llm_rerank = bool(self.cfg.get("llm_rerank", True))
        self.llm_rerank_k = int(self.cfg.get("llm_rerank_k", 10))

    # -------- lifecycle --------
    def load_runtime(self) -> bool:
        try:
            if not self.faiss_indexer.load_index():
                print("Failed to load FAISS index")
                return False
            self.runtime_data = self.faiss_indexer.get_runtime_data()
            return True
        except Exception as e:
            print(f"Error loading runtime data: {e}")
            return False

    # -------- main API --------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        if not self.runtime_data and not self.load_runtime():
            return []

        k = int(top_k or self.cfg.get("top_k", 5))
        threshold = float(min_score or self.cfg.get("min_score", 0.30))
        threshold = max(threshold, self.min_final)

        # 1) Vector search élargi (x3) + normalisation si IP
        initial = self._vector_search(query, top_k=k * 3)

        if not initial:
            return []

        # 2) Hybrid scoring + garde-fous (cosine + mots-clés + boosts + pénalités)
        scored = self._hybrid_score(initial, query)

        # 3) Diversité (MMR) sur les meilleurs candidats
        reranked = self._mmr_select(query, scored, k=max(k * 2, 10))

        # 4) LLM re-rank (optionnel) pour le top N, puis coupe finale
        if self.enable_llm_rerank:
            reranked = self._llm_rerank(query, reranked[: self.llm_rerank_k])

        # 5) Filtre final par score
        final = [h for h in reranked if h.get("final_score", 0.0) >= threshold]
        if not final:
            # pas de match fort → retourne les 1-2 meilleurs (utile pour transparence)
            final = reranked[: min(2, len(reranked))]

        # 6) Dé-duplication douce (évite des passages quasi identiques)
        final = deduplicate_hits(final, similarity_threshold=0.90)

        # tronque à k
        return final[:k]

    # -------- steps --------
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        rd = self.runtime_data
        index = rd["index"]
        idmap = rd["idmap"]
        meta = rd.get("meta", {})

        qv = np.array(self.embedding_model.embed([query])[
                      0], dtype="float32").reshape(1, -1)
        use_ip = str(meta.get("metric", "")).lower() == "ip"
        if use_ip and self.cfg.get("normalize", True):
            qv = norm(qv)
        scores, indices = index.search(qv, top_k)
        scores = scores[0]
        indices = indices[0]

        hits: List[Dict[str, Any]] = []
        # map global id -> row in idmap
        id_to_row = {int(i): r for r, i in enumerate(idmap["ids"])}
        for sc, idx in zip(scores, indices):
            if idx == -1:
                continue
            row = id_to_row.get(int(idx))
            if row is None:
                continue
            hit = {
                "score": float(sc),                 # raw sim (IP/L2)
                "id": int(idx),
                "ordinal": int(idmap["ordinal"][row]),
                "content": str(idmap["content"][row]),
                "subject": str(idmap["subject"][row]),
                "source": str(idmap["source"][row]),
            }
            hits.append(hit)
        return hits

    def _hybrid_score(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        kws = [k.lower() for k in extract_keywords(query)]
        out: List[Dict[str, Any]] = []

        for h in candidates:
            content = h.get("content", "")
            subject = h.get("subject", "")
            sim = float(h.get("score", 0.0))

            # garde-fou cosine (évite du bruit total)
            if sim < self.min_cosine and not kws:
                continue

            kw_s = _keyword_overlap_score(kws, f"{subject} {content}")
            tit_b = _title_boost(subject, query)
            pen = _policy_penalty(f"{subject} {content}", query)

            final = self.w_sim * sim + self.w_kw * kw_s + self.w_tit * tit_b
            final = max(0.0, final - pen)

            hh = dict(h)
            hh["kw_score"] = kw_s
            hh["title_boost"] = tit_b
            hh["penalty"] = pen
            hh["final_score"] = float(final)
            out.append(hh)

        out.sort(key=lambda x: x["final_score"], reverse=True)
        return out

    def _mmr_select(self, query: str, ranked: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        # prends les vecteurs des candidats (on les re-calcule via embedder)
        texts = [
            f"{h.get('subject','')} {h.get('content','')}" for h in ranked]
        embs = self.embedding_model.embed(texts)  # list[list[float]]
        M = np.array(embs, dtype="float32")
        qv = np.array(self.embedding_model.embed([query])[
                      0], dtype="float32").reshape(1, -1)

        # normaliser pour cosine
        M = norm(M)
        qv = norm(qv)

        pick_idxs = _mmr(qv, M, k=k, lambda_=0.7)
        return [ranked[i] for i in pick_idxs]

    def _llm_rerank(self, query: str, topN: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fait scorer par le LLM la pertinence de chaque snippet (0..4),
        puis combine : final_score = 0.6*final_score + 0.4*(llm_score/4)
        S’il n’y a pas de LLM actif, on renvoie tel quel.
        """
        # on récupère le LLM actif
        try:
            from api.runtime_llm import STATE as RUNTIME_LLM_STATE
            llm = RUNTIME_LLM_STATE.llm if RUNTIME_LLM_STATE.ready else None
        except Exception:
            llm = None

        if not llm or not topN:
            return topN

        # construire un prompt concis et robuste
        bullets = []
        for i, h in enumerate(topN, 1):
            txt = (h.get("content") or "").replace("\n", " ")
            txt = txt[:600]  # borne pour éviter réponses trop longues
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
            text = getattr(resp, "content", str(resp))
        except Exception:
            text = ""

        # parsing très tolérant
        scores = {}
        for m in re.finditer(r"(\d+)\s*[:\-]\s*([0-4])", text):
            idx = int(m.group(1))
            val = int(m.group(2))
            if 1 <= idx <= len(topN):
                scores[idx - 1] = val

        # repondère
        out = []
        for i, h in enumerate(topN):
            llm_s = scores.get(i, 0)
            comb = 0.6 * float(h.get("final_score", 0.0)) + 0.4 * (llm_s / 4.0)
            hh = dict(h)
            hh["llm_score"] = llm_s
            hh["final_score"] = float(comb)
            out.append(hh)

        out.sort(key=lambda x: x["final_score"], reverse=True)
        return out
