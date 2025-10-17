# rag/generator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

from rag.helpers import format_context_for_llm, sources_list, extract_keywords

_MIN_CONTEXT_CHARS = 180           # ignorer les hits trop courts
_MIN_SOURCES_IN_ANSWER = 1         # exiger au moins 1 [#n]
_MIN_ANSWER_CHARS = 120            # refuser les réponses trop courtes
_REQUIRE_CONTEXT = True            # mode strict: pas de “connaissance générale”

_CITE_RE = re.compile(r"\[#\d+\]")


class RAGGenerator:
    """
    Générateur STRICTEMENT ancré dans le contexte.
    - Si le contexte est vide/faible => refuse poliment (pas d'hallucination).
    - Chaque claim doit citer [#n]; sinon on refait un fallback extractif.
    """

    def __init__(self, llm: Any = None, max_chars: Optional[int] = None) -> None:
        self.llm = llm
        self.max_chars = int(max_chars or 1800)
        self._setup_prompts()

    def _setup_prompts(self) -> None:
        self.SYSTEM_PROMPT = (
            "Tu es un assistant expert webMethods. Tu DOIS répondre UNIQUEMENT avec les informations "
            "du contexte fourni. Chaque assertion factuelle doit référencer au moins une source au format [#n]. "
            "Si le contexte est insuffisant, dis explicitement 'Contexte insuffisant' et propose des pistes (sections/doc à ajouter), "
            "sans inventer de contenu ni utiliser de connaissances générales."
        )

        self.HUMAN_TEMPLATE = (
            "Question:\n{question}\n\n"
            "Contexte (avec références [#n]):\n{context}\n\n"
            "Consignes STRICTES:\n"
            "1) Réponds en français, de façon claire et concise.\n"
            "2) Utilise UNIQUEMENT le contexte; ne complète PAS par des connaissances générales.\n"
            "3) Chaque phrase importante doit citer au moins une référence [#n].\n"
            "4) Si le contexte ne permet pas de répondre, écris:\n"
            "   - 'Contexte insuffisant pour répondre précisément.'\n"
            "   - Puis suggère quels documents/sections ajouter.\n\n"
            "Réponse:"
        )

    def generate_answer(
        self,
        question: str,
        context_hits: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
    ) -> str:
        max_len = max_chars or self.max_chars

        # 0) filtrer un contexte “utile”
        filtered = []
        for h in context_hits or []:
            txt = (h.get("content") or "").strip()
            if len(txt) >= _MIN_CONTEXT_CHARS:
                filtered.append(h)

        has_ctx = bool(filtered)
        ctx = format_context_for_llm(
            filtered, max_chars=1500) if has_ctx else ""

        # 1) si pas de LLM → extractif strict
        if not self.llm:
            return self._extractive_strict(question, filtered)[:max_len]

        # 2) si contexte vide et mode strict → refuser poliment
        if _REQUIRE_CONTEXT and not has_ctx:
            return self._no_context_reply(question)[:max_len]

        # 3) génération LLM ancrée
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", self.HUMAN_TEMPLATE),
            ])
            chain = prompt | self.llm | StrOutputParser()
            out = (chain.invoke(
                {"question": question, "context": ctx}) or "").strip()
        except Exception:
            out = ""

        # 4) validation : doit citer [#n], longueur minimale
        if not self._is_valid_anchored_answer(out):
            # fallback extractif très strict (et court)
            out = self._extractive_strict(question, filtered)

        # 5) ajouter bloc Sources (lisible)
        if has_ctx:
            out = f"{out}\n\n**Sources**\n{sources_list(filtered)}"

        return out[:max_len]

    def generate_streaming_answer(self, question: str, context_hits: List[Dict[str, Any]]):
        text = self.generate_answer(question, context_hits)
        step = 50
        for i in range(0, len(text), step):
            yield text[i:i+step]

    # -------- internals --------

    def _is_valid_anchored_answer(self, text: str) -> bool:
        if not text or len(text) < _MIN_ANSWER_CHARS:
            return False
        cites = _CITE_RE.findall(text)
        return len(cites) >= _MIN_SOURCES_IN_ANSWER
    

    def _no_context_reply(self, question: str) -> str:
        kws= extract_keywords(question)
        topic= kws[0] if kws else "le sujet"
        return (
            f"Contexte insuffisant pour répondre précisément sur « {topic} ».\n"
            "Ajoutez/ingérez une fiche d’overview ou la section concernée (ex: introduction webMethods, "
            "composant visé, guide d’exploitation) puis relancez la question."
        )

    def _extractive_strict(self, question: str, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return self._no_context_reply(question)
        parts= ["## Informations issues de la base :\n"]
        for i, h in enumerate(hits[:3], 1):
            subject= h.get("subject") or "Extrait"
            content= (h.get("content") or "").strip()
            source= (h.get("source") or "document").split("\\")[-1].split("/")[-1]
            if not content:
                continue
            snippet= content[:600] + ("…" if len(content) > 600 else "")
            parts.append(
                f"### [#{i}] {subject}\n{snippet}\n*Source: {source}*")
        parts.append(
            "\n*Réponse extraite automatiquement du contexte disponible.*")
        return "\n".join(parts)
