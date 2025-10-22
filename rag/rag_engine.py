from __future__ import annotations

import logging
import time
import unicodedata
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

from config.settings import settings
from indexing.faiss_indexer import FAISSIndexer
from rag.generator import RAGGenerator
from rag.helpers import format_context_for_llm
from rag.retriever import RAGRetriever

log = logging.getLogger(__name__)

# ---------- heuristics / helpers ----------

ALIASES = {
    "webmethods": "webMethods",
    "web methods": "webMethods",
    "wm": "webMethods",
}

DEF_PAT = re.compile(
    r"\b(c['’]est\s*quoi|qu['’]est-ce\s*que|definition|définition|what\s+is)\b",
    re.I,
)
OVERVIEW_HINTS = (
    "overview", "introduction", "qu’est-ce que", "qu est ce que",
    "présentation", "vue d’ensemble", "concepts clés", "basics", "getting started"
)


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_query(q: str) -> str:
    """Lower, strip accents, trim spaces (for detection only)."""
    return _strip_accents(q).lower().strip()


def expand_aliases(q: str) -> str:
    """Replace common aliases in a user-visible way (preserve casing of targets)."""
    res = q
    for k, v in ALIASES.items():
        res = re.sub(rf"\b{k}\b", v, res, flags=re.IGNORECASE)
    return res


def is_definitional(q_raw: str) -> bool:
    return bool(DEF_PAT.search(normalize_query(q_raw)))


def rerank_hits_for_overview(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Boost hits that look like definitions/overviews."""
    def score(h: Dict[str, Any]) -> float:
        base = float(h.get("score") or 0.0)
        txt = f"{h.get('subject','')} {h.get('content','')}".lower()
        boost = 0.0
        if "webmethods" in txt:
            boost += 0.10
        if any(hint in txt for hint in OVERVIEW_HINTS):
            boost += 0.15
        if len(h.get("content", "")) >= 120:  # avoid tiny snippets
            boost += 0.05
        return base + boost

    return sorted(hits, key=score, reverse=True)


def merge_dedup_hits(
    a: List[Dict[str, Any]],
    b: List[Dict[str, Any]],
    key: Callable[[Dict[str, Any]], Tuple[str, int]]
    = lambda h: (str(h.get("source")), int(h.get("ordinal") or -1)),
) -> List[Dict[str, Any]]:
    """Merge two hit lists, dedupe by (source, ordinal)."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for lst in (a, b):
        for h in lst or []:
            k = key(h)
            if k in seen:
                continue
            seen.add(k)
            out.append(h)
    return out


class RAGEngine:
    """RAG engine combining vector retrieval and LLM generation."""

    def __init__(self, llm: Any = None, faiss_indexer: Optional[FAISSIndexer] = None) -> None:
        self.llm = llm
        self.retriever = RAGRetriever(faiss_indexer)
        self.generator = RAGGenerator(llm)
        self.simple_mode = bool(settings.get_retrieval_config().get("simple_mode", False))

        # rolling stats
        self.session_stats: Dict[str, Any] = {
            "queries_processed": 0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_total_time": 0.0,
        }

    # ---------------- lifecycle ----------------
    def initialize(self) -> bool:
        log.info("Initialisation du moteur RAG…")
        if not self.retriever.load_runtime():
            log.error("Index FAISS introuvable: %s", settings.faiss_index_path)
            return False
        if not self.llm:
            log.warning("LLM non configuré : fallback extractif uniquement.")
        else:
            log.info("LLM détecté — génération activée.")
        return True

    def set_llm(self, llm: Any) -> None:
        self.llm = llm
        self.generator.llm = llm

    # ---------------- query paths ----------------
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline: retrieval (+hybrid fallback) → re-ranking → generation.
        """
        t0 = time.time()
        try:
            # 0) preproc
            q_raw = (question or "").strip()
            if not q_raw:
                return {"success": False, "error": "Empty question", "question": question}

            definitional = False if self.simple_mode else is_definitional(
                q_raw)
            q_normed = q_raw if self.simple_mode else expand_aliases(q_raw)

            # Better defaults for definitional questions
            tk = max(6, int(top_k or 6)) if definitional else int(top_k or 4)
            ms = float(min_score) if min_score is not None else (
                0.25 if definitional else 0.30)

            # 1) dense retrieval
            t1 = time.time()
            dense_hits = self.retriever.retrieve(
                q_normed, top_k=tk, min_score=ms) or []
            dense_time = time.time() - t1

            # average score (for hybrid trigger)
            if dense_hits:
                avg_dense = sum(float(h.get("score") or 0.0)
                                for h in dense_hits) / len(dense_hits)
            else:
                avg_dense = 0.0

            # 2) hybrid lexical fallback (if available in your retriever)
            #    — only if results are few or low-confidence
            hybrid_hits = dense_hits
            if (len(dense_hits) < 2 or avg_dense < 0.35) and hasattr(self.retriever, "retrieve_lexical"):
                try:
                    lex_hits = self.retriever.retrieve_lexical(
                        q_normed, top_k=tk) or []
                    hybrid_hits = merge_dedup_hits(dense_hits, lex_hits)
                except Exception:
                    # lexical not wired or failed — keep dense only
                    hybrid_hits = dense_hits

            # 3) re-rank to surface overview/definition chunks
            hits = hybrid_hits if self.simple_mode else rerank_hits_for_overview(
                hybrid_hits)


            # 4) generation
            t2 = time.time()
            answer = self.generator.generate_answer(q_raw, hits)
            gen_time = time.time() - t2

            total = time.time() - t0
            self._accum_stats(dense_time, gen_time, total)

            result: Dict[str, Any] = {
                "question": q_raw,
                "answer": answer,
                "context_found": bool(hits),
                "context_count": len(hits),
                "timing": {
                    "retrieval_time": round(dense_time, 3),
                    "generation_time": round(gen_time, 3),
                    "total_time": round(total, 3),
                },
                "success": True,
            }
            if return_context:
                result["context_hits"] = hits
                result["formatted_context"] = format_context_for_llm(hits)
            return result

        except Exception as e:  # pragma: no cover
            log.exception("Erreur dans query()")
            return {
                "question": question,
                "answer": f"Erreur: {e}",
                "success": False,
                "error": str(e),
            }

    def retrieve_only(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            t0 = time.time()
            hits = self.retriever.retrieve(question, top_k, min_score) or []
            tt = time.time() - t0
            return {
                "question": question,
                "context_hits": hits,
                "context_count": len(hits),
                "formatted_context": format_context_for_llm(hits),
                "retrieval_time": round(tt, 3),
                "success": True,
            }
        except Exception as e:  # pragma: no cover
            log.exception("Erreur retrieve_only()")
            return {"question": question, "success": False, "error": str(e)}

    def generate_only(self, question: str, context_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            t0 = time.time()
            answer = self.generator.generate_answer(question, context_hits)
            return {
                "question": question,
                "answer": answer,
                "context_count": len(context_hits),
                "generation_time": round(time.time() - t0, 3),
                "success": True,
            }
        except Exception as e:  # pragma: no cover
            log.exception("Erreur generate_only()")
            return {"question": question, "answer": f"Erreur: {e}", "success": False, "error": str(e)}

    # ---------------- batch / streaming ----------------
    def batch_query(self, questions: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for q in questions:
            out.append(self.query(q, **kwargs))
        return out

    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Iterable[str]:
        try:
            hits = self.retriever.retrieve(question, top_k, min_score) or []
            yield from self.generator.generate_streaming_answer(question, hits)
        except Exception as e:  # pragma: no cover
            log.exception("Erreur stream_query()")
            yield f"Erreur de streaming: {e}"

    # ---------------- evaluation & health ----------------
    def evaluate_query(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        **query_kwargs: Any,
    ) -> Dict[str, Any]:
        res = self.query(question, return_context=True, **query_kwargs)
        if not res.get("success"):
            return {"success": False, "error": res.get("error")}

        metrics: Dict[str, Any] = {
            "question": question,
            "context_found": res["context_found"],
            "context_count": res["context_count"],
            "retrieval_time": res["timing"]["retrieval_time"],
            "generation_time": res["timing"]["generation_time"],
            "total_time": res["timing"]["total_time"],
        }
        if "context_hits" in res:
            q = self.generator.evaluate_response_quality(
                question, res["context_hits"], res["answer"]
            )
            metrics.update(q)
        if expected_answer:
            metrics["expected_match"] = self._compare_answers(
                res["answer"], expected_answer)
        metrics["success"] = True
        return metrics

    def get_stats(self) -> Dict[str, Any]:
        return {
            "session": self.session_stats.copy(),
            "retriever": self.retriever.get_stats(),
            "initialized": self.retriever.runtime_data is not None,
            "llm_available": self.llm is not None,
        }

    def update_config(
        self,
        retriever_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if retriever_config:
            self.retriever.update_config(**retriever_config)
        if generator_config and "max_chars" in generator_config:
            self.generator.max_chars = int(generator_config["max_chars"])

    def health_check(self) -> Dict[str, Any]:
        health = {"overall": "healthy", "components": {}, "issues": []}
        if not self.retriever.runtime_data:
            health["components"]["retriever"] = "error"
            health["issues"].append("Index FAISS non chargé")
            health["overall"] = "error"
        else:
            health["components"]["retriever"] = "healthy"

        if not self.llm:
            health["components"]["generator"] = "warning"
            health["issues"].append("LLM non configuré")
            if health["overall"] == "healthy":
                health["overall"] = "warning"
        else:
            health["components"]["generator"] = "healthy"
        return health

    # ---------------- internals ----------------
    def _accum_stats(self, rt: float, gt: float, tt: float) -> None:
        self.session_stats["queries_processed"] += 1
        n = self.session_stats["queries_processed"]
        self.session_stats["avg_retrieval_time"] = (
            (self.session_stats["avg_retrieval_time"] * (n - 1)) + rt) / n
        self.session_stats["avg_generation_time"] = (
            (self.session_stats["avg_generation_time"] * (n - 1)) + gt) / n
        self.session_stats["avg_total_time"] = (
            (self.session_stats["avg_total_time"] * (n - 1)) + tt) / n

    def _compare_answers(self, generated: str, expected: str) -> Dict[str, float]:
        from rag.helpers import extract_keywords
        gk = set(extract_keywords((generated or "").lower()))
        ek = set(extract_keywords((expected or "").lower()))
        if not ek:
            return {"keyword_overlap": 0.0, "generated_keywords": len(gk), "expected_keywords": 0}
        overlap = len(gk & ek) / max(1, len(ek))
        return {"keyword_overlap": overlap, "generated_keywords": len(gk), "expected_keywords": len(ek)}
