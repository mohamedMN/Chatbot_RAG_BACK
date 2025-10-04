# rag/generator.py
"""
Générateur RAG robuste :
- Prompts FR orientés webMethods IS
- Guardrails anti-sortie vide/placeholder
- Fallback extractif quand LLM indisponible
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from rag.helpers import extract_keywords, format_context_for_llm

# --- Guardrails ---
PLACEHOLDER_MARKERS = {"[...]", "...", "(…)", "résultat :", "result :"}


def _is_placeholder(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if any(m in t for m in PLACEHOLDER_MARKERS):
        return True
    # trop court / quasi seulement ponctuation
    if len(t) < 24:
        return True
    if all(ch in ".-–—_*[]() :;," for ch in t):
        return True
    return False


def _extractive_fallback(context: str, max_lines: int = 12) -> str:
    lines = [ln.strip() for ln in (context or "").splitlines() if ln.strip()]
    head = "\n".join(lines[:max_lines]) if lines else "Contexte indisponible."
    return (
        "## Réponse (extrait du contexte)\n"
        f"{head}\n\n"
        "*Remarque : génération LLM indisponible, réponse extractive.*"
    )


class RAGGenerator:
    """
    Générateur de réponses RAG utilisant LLM (LangChain).
    Si llm=None, bascule en mode clarification/extractif selon le contexte.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.max_chars = getattr(settings, 'MAX_RESPONSE_CHARS', 1800)
        self._setup_prompts()

    def _setup_prompts(self):
        """Configure les templates de prompts"""
        self.SYSTEM_PROMPT = (
            "Tu es un assistant technique francophone expert en intégration de données et Software AG webMethods. "
            "OBLIGATIONS : (1) Répondre en français technique, (2) Ne pas citer/paraphraser les consignes, "
            "(3) Ne pas t'excuser, (4) Utiliser le contexte fourni et citer [#n] quand pertinent, "
            "(5) Séparer ce qui vient du contexte des apports métier. "
            "Si le contexte est vide ou hors-sujet, NE DONNE PAS DE RÉPONSE FACTUELLE : "
            "écris « Information manquante » puis propose 2–5 clarifications spécifiques."
        )
        # Nudge spécifique webMethods IS
        self.SYSTEM_PROMPT += (
            " Si la question contient 'webMethods IS' ou 'Integration Server', commence par une définition précise "
            "de webMethods Integration Server (rôle ESB/API, packages, services Flow, triggers, ports, sécurité), "
            "puis enchaîne avec l’architecture, les fonctionnalités clés et les usages courants."
        )

        self.CLARIFY_TEMPLATE = (
            "Le contexte est vide ou hors-sujet pour cette question.\n\n"
            "=== FORMAT (Markdown) ===\n"
            "## Information manquante\n"
            "- Aucun extrait pertinent n'a été trouvé.\n\n"
            "## Clarifications à préciser\n"
            "- De quel « {kw_hint} » s'agit-il ? (produit logiciel, projet interne, service cloud, autre)\n"
            "- Contexte d'usage attendu (intégration webMethods, API, data pipeline, sécurité, observabilité) ?\n"
            "- Environnement (prod/dev), fournisseur ou équipe concernée ?\n"
            "- Besoin exact (définition générale, architecture, procédure d'installation, comparatif, troubleshooting) ?\n"
            "==============================="
        )

        self.ANSWER_TEMPLATE = (
            "Question :\n{question}\n\n"
            "Contexte (extraits numérotés) :\n{context}\n\n"
            "=== FORMAT (Markdown) — commence directement par les sections, sans préambule ===\n"
            "## Réponse courte\n"
            "- (1 à 3 phrases, réponse directe)\n\n"
            "## Détails appuyés par le contexte\n"
            "- Puces concises avec références [#n].\n\n"
            "## Apports métier / connaissances générales\n"
            "- Points utiles issus de l'expertise webMethods/ESB/API/ETL (sans inventer de faits).\n\n"
            "## Hypothèses\n"
            "- ...\n\n"
            "## Sources\n"
            "- [#n] et/ou « Connaissances générales »\n"
            "==============================="
        )

    # ---------- API publique ----------
    def generate_answer(self, question: str, context_hits: List[Dict],
                        max_chars: Optional[int] = None) -> str:
        """
        Génère une réponse structurée à partir de la question et du contexte.
        Si llm=None → clarification/extractif.
        """
        max_response_chars = max_chars or self.max_chars
        context = format_context_for_llm(context_hits, max_chars=1500)

        if not self.llm:
            # Sans LLM : si pas de contexte → clarifier, sinon extractif
            if not context or context.strip() == "(aucun extrait)":
                return self._generate_clarification(question, max_response_chars)
            return _extractive_fallback(context)

        return self.answer_with_llm(self.llm, question, context, max_response_chars)

    def answer_with_llm(self, llm, question: str, context: str, max_chars: int = 1800) -> str:
        """
        Réponse structurée. Si contexte vide → clarification.
        Guardrails si le modèle renvoie du placeholder.
        """
        no_ctx = not context or context.strip() == "(aucun extrait)"
        if no_ctx:
            return self._generate_clarification(question, max_chars)
        else:
            return self._generate_contextual_answer(llm, question, context, max_chars)

    # ---------- internes ----------
    def _generate_clarification(self, question: str, max_chars: int) -> str:
        kw_hint = (extract_keywords(question) or [
                   question.strip() or "terme"])[0][:40]

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.CLARIFY_TEMPLATE),
        ])

        if self.llm:
            try:
                chain = prompt | self.llm | StrOutputParser()
                out = chain.invoke({"kw_hint": kw_hint}).strip()
                if _is_placeholder(out):
                    out = self.CLARIFY_TEMPLATE.format(kw_hint=kw_hint)
                return out[:max_chars]
            except Exception:
                pass

        # Fallback total
        return self.CLARIFY_TEMPLATE.format(kw_hint=kw_hint)[:max_chars]

    def _generate_contextual_answer(self, llm, question: str, context: str, max_chars: int) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.ANSWER_TEMPLATE),
        ])

        try:
            chain = prompt | llm | StrOutputParser()
            out = chain.invoke({
                "question": (question or "").strip(),
                "context": (context or "(aucun extrait)").strip(),
            }).strip()
            if _is_placeholder(out):
                return _extractive_fallback(context)
            return out[:max_chars]
        except Exception:
            # Fallback extractif si le LLM échoue
            return _extractive_fallback(context)
