# rag/generator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import logging
import os
from rag.helpers import format_context_for_llm, sources_list, extract_keywords

log = logging.getLogger(__name__)

# Configuration
_MIN_CONTEXT_CHARS = 50           # Minimum chars par chunk pour être considéré
_MIN_SOURCES_IN_ANSWER = 2        # Minimum de citations [#n] requises
_MIN_ANSWER_CHARS = 100           # Longueur minimale de réponse acceptable
_REQUIRE_CONTEXT = True           # Mode strict: refuse si pas de contexte

# NEW: env toggle (default: NO fallback)
_ALLOW_FALLBACK = os.getenv("RAG_ALLOW_FALLBACK", "0") == "1"

# Regex pour détecter les citations
_CITE_RE = re.compile(r"\[#\d+\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def _split_sentences(text: str) -> List[str]:
    """Découpe le texte en phrases."""
    s = (text or "").strip()
    if not s:
        return []
    return [t.strip() for t in _SENT_SPLIT.split(s) if t.strip()]


def _valid_cites_in(text: str, max_idx: int) -> List[int]:
    """Extrait les numéros de citation valides [#n] dans le texte."""
    out = []
    for m in re.finditer(r"\[#(\d+)\]", text):
        n = int(m.group(1))
        if 1 <= n <= max_idx:
            out.append(n)
    return out


def _drop_invalid_cites(text: str, max_idx: int) -> str:
    """Supprime les citations [#n] invalides (n > max_idx)."""
    return re.sub(
        r"\[#(\d+)\]",
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= max_idx else "",
        text
    )


def _keyword_overlap(a: str, b: str) -> float:
    """Calcule le chevauchement de mots-clés entre deux textes."""
    A = set(extract_keywords((a or "").lower()))
    B = set(extract_keywords((b or "").lower()))
    return 0.0 if not A else len(A & B) / max(1, len(A))


def _attach_best_cite(sentence: str, hits: List[Dict[str, Any]]) -> str:
    """
    Si une phrase n'a pas de citation [#n], attache automatiquement
    la source la plus pertinente par chevauchement de mots-clés.
    """
    if re.search(r"\[#\d+\]", sentence):
        return sentence

    best_idx, best_score = None, -1.0
    for i, h in enumerate(hits, 1):
        txt = f"{h.get('subject','')} {h.get('content','')}"
        sc = _keyword_overlap(sentence, txt)
        if sc > best_score:
            best_idx, best_score = i, sc

    if best_idx is not None and best_score > 0.05:
        return sentence.rstrip() + f" [#{best_idx}]"
    return sentence


def _enforce_anchoring(text: str, hits: List[Dict[str, Any]], min_ratio: float = 0.7) -> str:
    """
    S'assure qu'au moins `min_ratio` des phrases ont une citation [#n].
    Si pas assez, attache automatiquement des citations.
    """
    if not hits:
        return text

    max_idx = len(hits)
    text = _drop_invalid_cites(text, max_idx)

    lines = text.splitlines()
    new_lines = []

    for line in lines:
        # Ignorer les en-têtes et lignes vides
        if not line.strip() or line.strip().startswith(("**Sources", "Sources:", "##", "# ")):
            new_lines.append(line)
            continue

        sents = _split_sentences(line) or [line]

        # Calculer le ratio de phrases citées
        with_cite = sum(1 for s in sents if re.search(r"\[#\d+\]", s))
        ratio = with_cite / max(1, len(sents))

        if ratio >= min_ratio:
            new_lines.append(line)
            continue

        # Fixer les phrases sans citation
        fixed = []
        for s in sents:
            # Ne pas citer les fragments trop courts
            if len(s) < 20:
                fixed.append(s)
                continue
            fixed.append(_attach_best_cite(s, hits))

        new_lines.append(" ".join(fixed))

    return "\n".join(new_lines)


def _count_content_sentences(text: str) -> int:
    """Compte le nombre de phrases de contenu (hors en-têtes)."""
    c = 0
    for line in text.splitlines():
        if line.strip().startswith(("**Sources", "Sources:", "##", "# ")):
            continue
        c += len(_split_sentences(line))
    return c


class RAGGenerator:
    """
    Générateur STRICTEMENT ancré dans le contexte fourni.
    
    Principes:
    - Refuse de répondre si contexte insuffisant (pas d'hallucination)
    - Chaque affirmation doit citer au moins une source [#n]
    - Réponses concises et précises (3-5 phrases max)
    - Fallback extractif si le LLM échoue
    """

    def __init__(self, llm: Any = None, max_chars: Optional[int] = None) -> None:
        """
        Args:
            llm: Instance LLM compatible Langchain (avec .invoke())
            max_chars: Limite de caractères pour la réponse finale
        """
        self.llm = llm
        self.max_chars = int(max_chars or 1800)
        self._setup_prompts()

        log.info(
            f"RAGGenerator initialisé - LLM: {type(llm).__name__ if llm else 'None'}")

    def _detect_small_model(self, llm: Any) -> bool:
        """Détecte si c'est un petit modèle (<3B paramètres)."""
        if not llm:
            return False
        
        llm_type = type(llm).__name__.lower()
        
        # Pour LMStudioChat, vérifier le nom du modèle
        if hasattr(llm, 'model'):
            model_name = str(llm.model).lower()
            
            # Modèles petits explicites
            small_markers = [
                "1b", "1.5b", "2b", "3b",  # Taille
                "tiny", "small", "mini",    # Qualificatifs
                "llama-3.2-1b", "llama3.2:1b",  # Modèles spécifiques
            ]
            
            if any(marker in model_name for marker in small_markers):
                return True
            
            # Modèles grands explicites
            large_markers = ["7b", "8b", "13b", "70b", "mistral", "mixtral"]
            if any(marker in model_name for marker in large_markers):
                return False
        
        # Par défaut: considérer comme grand modèle
        return False

    def _setup_prompts(self) -> None:
        """Configure les prompts système et utilisateur."""
        if self._detect_small_model:
                # ✅ PROMPT ULTRA-SIMPLIFIÉ pour llama-3.2-1b
            self.SYSTEM_PROMPT = "Tu es un assistant technique."

            self.HUMAN_TEMPLATE = (
                "Info:\n{context}\n\n"
                "Q: {question}\n"
                "R:"  # Très court !
            )
        else:
            self.SYSTEM_PROMPT = (
                "Tu es un assistant technique expert en webMethods et architecture d'intégration. "
                "Tu réponds UNIQUEMENT avec les informations extraites du contexte fourni.\n\n"
                "RÈGLES STRICTES:\n"
                "1. Chaque affirmation factuelle DOIT citer au moins une source [#n]\n"
                "2. Ne JAMAIS utiliser de connaissances générales ou externes\n"
                "3. Si le contexte est insuffisant, dis explicitement 'Contexte insuffisant'\n"
                "4. Réponds de manière CONCISE (3-5 phrases maximum)\n"
                "5. Structure: réponse directe + citations, pas de reformulation du contexte"
            )

            self.HUMAN_TEMPLATE = (
                "CONTEXTE DOCUMENTAIRE (avec références [#n]):\n"
                "{context}\n\n"
                "QUESTION DE L'UTILISATEUR:\n"
                "{question}\n\n"
                "INSTRUCTIONS DE RÉPONSE:\n"
                "• Réponds en français, de façon claire et directe\n"
                "• Utilise UNIQUEMENT les informations du contexte ci-dessus\n"
                "• Chaque phrase importante doit inclure [#n] pour citer la source\n"
                "• Maximum 5 phrases courtes\n"
                "• Si le contexte ne permet pas de répondre précisément:\n"
                "  - Écris: 'Contexte insuffisant pour répondre à cette question.'\n"
                "  - Suggère quels documents/sections seraient nécessaires\n\n"
                "RÉPONSE:"
            )
        

    def generate_answer(
        self,
        question: str,
        context_hits: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Génère une réponse ancrée dans le contexte fourni.

        Args:
            question: Question de l'utilisateur
            context_hits: Liste des chunks pertinents avec leurs métadonnées
            max_chars: Limite de caractères (override)

        Returns:
            Réponse générée avec citations [#n] et bloc Sources
        """
        max_len = max_chars or self.max_chars
        raw_hits = context_hits or []

        log.info(f"📝 Génération réponse - Question: '{question[:60]}...'")
        log.info(f"📚 Chunks reçus: {len(raw_hits)}")

        # 1) Filtrer les chunks avec contenu significatif
        filtered = []
        for h in raw_hits:
            txt = (h.get("content") or "").strip()
            if len(txt) >= _MIN_CONTEXT_CHARS:
                filtered.append(h)

        # Fallback: garder au moins les 3-5 premiers même s'ils sont courts
        if not filtered and raw_hits:
            filtered = raw_hits[:min(5, len(raw_hits))]
            log.warning(
                "⚠️  Aucun chunk avec contenu suffisant, utilisation des premiers chunks")

        has_ctx = bool(filtered)
        log.info(f"✅ Chunks filtrés: {len(filtered)}")

        # 2) Formater le contexte pour le LLM
        if self._detect_small_model:
            # ✅ Pour petits modèles: MAX 800 chars de contexte
            # ✅ Format simple pour petits modèles) if has_ctx else ""
            ctx = format_context_for_llm(
                filtered, max_chars=800, simple_format=self._detect_small_model)
        else:
            ctx = format_context_for_llm(filtered, max_chars=1500) if has_ctx else ""

        # 3) Pas de LLM → Mode extractif strict
        if not self.llm:
            log.warning("⚠️  Pas de LLM configuré, mode extractif")
            return self._extractive_strict(question, filtered)[:max_len]

        # 4) Pas de contexte et mode strict → Refus poli
        if _REQUIRE_CONTEXT and not has_ctx:
            log.warning("⚠️  Pas de contexte, refus de répondre")
            return self._no_context_reply(question)[:max_len]

        # 5) Génération LLM ancrée
        try:
            log.debug(f"🤖 Construction du prompt - Contexte: {len(ctx)} chars")

            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", self.HUMAN_TEMPLATE),
            ])

            # Configuration LLM avec paramètres stricts
            llm = getattr(self.llm, "bind", lambda **kw: self.llm)(
                temperature=0.1,  # Très déterministe
                max_tokens=500,   # Réponses plus complètes
                # Arrêter avant bloc sources
                stop=["\n**Sources", "\nSources:", "\n\n---"]
            )

            chain = prompt | llm | StrOutputParser()

            log.info("🔄 Invocation du LLM...")
            out = (chain.invoke({
                "question": question,
                "context": ctx
            }) or "").strip()

            log.info(f"✅ Réponse LLM reçue: {len(out)} chars")

        except Exception as e:
            log.error(f"❌ Erreur LLM: {e}", exc_info=True)
            out = ""

        # 6) Validation et ancrage forcé
        if has_ctx:
            out = _drop_invalid_cites(out, max_idx=len(filtered))
            # 60% des phrases citées
            out = _enforce_anchoring(out, filtered, min_ratio=0.6)

            citations = _valid_cites_in(out, len(filtered))
            log.info(
                f"📎 Citations trouvées: {len(set(citations))} sources uniques")

        # 7) Validation finale: réponse acceptable ?
        if not self._is_valid_anchored_answer(out):
            log.warning(
                "⚠️  Réponse LLM invalide (trop courte ou pas assez de citations)")
            if _ALLOW_FALLBACK and has_ctx:
                log.warning(
                    "    → Fallback extractif (activé via RAG_ALLOW_FALLBACK=1)")
                out = self._extractive_strict(question, filtered)
            else:
                log.info(
                    "    → Pas de fallback (RAG_ALLOW_FALLBACK=0). On retourne la réponse LLM telle quelle.")

        # 8) Ajouter le bloc Sources (lisible et cliquable)
        if has_ctx:
            out = f"{out}\n\n**Sources utilisées:**\n{sources_list(filtered)}"

        final = out[:max_len]
        log.info(f"✓ Réponse finale: {len(final)} chars")
        return final

    def generate_streaming_answer(
        self,
        question: str,
        context_hits: List[Dict[str, Any]]
    ):
        """
        Version streaming de generate_answer (pour affichage progressif).
        
        Yields:
            Chunks de texte progressifs
        """
        text = self.generate_answer(question, context_hits)
        step = 50  # Envoyer par paquets de 50 chars

        for i in range(0, len(text), step):
            yield text[i:i+step]

    def evaluate_response_quality(
        self,
        question: str,
        hits: List[Dict[str, Any]],
        answer: str
    ) -> Dict[str, Any]:
        """
        Évalue la qualité de la réponse (pour dashboards/monitoring).

        Returns:
            Métriques de qualité: citations, longueur, couverture keywords, etc.
        """
        max_idx = len(hits)
        answer = (answer or "").strip()

        # Citations
        cited = _valid_cites_in(answer, max_idx)
        unique_citations = len(set(cited))

        # Phrases avec/sans citations
        sent_total = _count_content_sentences(answer)
        sent_with_cite = 0
        for line in answer.splitlines():
            if line.strip().startswith(("**Sources", "Sources:", "##", "# ")):
                continue
            for s in _split_sentences(line):
                if re.search(r"\[#\d+\]", s):
                    sent_with_cite += 1

        cite_ratio = (sent_with_cite / max(1, sent_total)
                      ) if sent_total else 0.0

        # Couverture keywords question ↔ réponse
        qk = set(extract_keywords((question or "").lower()))
        ak = set(extract_keywords((answer or "").lower()))
        kw_overlap_q = (len(qk & ak) / max(1, len(qk))) if qk else 0.0

        # Support depuis les chunks cités
        cited_txt = " ".join(
            f"{hits[i-1].get('subject','')} {hits[i-1].get('content','')}"
            for i in cited if 1 <= i <= max_idx
        )
        ck = set(extract_keywords(cited_txt.lower()))
        kw_support = (len(ak & ck) / max(1, len(ak))) if ak else 0.0

        # Verdict global
        passes = bool(
            len(answer) >= _MIN_ANSWER_CHARS
            and cite_ratio >= 0.5
            and unique_citations >= _MIN_SOURCES_IN_ANSWER
        )

        return {
            "length_chars": len(answer),
            "sentences_total": sent_total,
            "sentences_with_citation": sent_with_cite,
            "citation_sentence_ratio": round(cite_ratio, 3),
            "unique_citations": unique_citations,
            "keyword_overlap_with_question": round(kw_overlap_q, 3),
            "keyword_support_from_cited_chunks": round(kw_support, 3),
            "passes_min_requirements": passes,
        }

    # ==================== Méthodes internes ====================

    def _is_valid_anchored_answer(self, text: str) -> bool:
        """Vérifie qu'une réponse a assez de contenu et de citations."""
        if not text or len(text) < _MIN_ANSWER_CHARS:
            return False

        cites = _CITE_RE.findall(text)
        return len(cites) >= _MIN_SOURCES_IN_ANSWER

    def _no_context_reply(self, question: str) -> str:
        """Réponse polie quand le contexte est vide/insuffisant."""
        kws = extract_keywords(question)
        topic = kws[0] if kws else "cette question"

        return (
            f"**Contexte insuffisant**\n\n"
            f"Je n'ai pas trouvé d'information pertinente dans la documentation "
            f"pour répondre précisément à votre question sur « {topic} ».\n\n"
            f"**Suggestions:**\n"
            f"• Vérifiez que les documents pertinents sont bien indexés\n"
            f"• Ajoutez des sections d'introduction ou guides d'overview\n"
            f"• Reformulez la question avec des termes plus spécifiques"
        )

    def _extractive_strict(self, question: str, hits: List[Dict[str, Any]]) -> str:
        """
        Fallback extractif: retourne directement les chunks les plus pertinents
        sans génération LLM, avec formatage propre.
        """
        if not hits:
            return self._no_context_reply(question)

        parts = ["**Informations extraites de la documentation:**\n"]

        for i, h in enumerate(hits[:3], 1):  # Max 3 premiers chunks
            subject = h.get("subject") or "Information"
            content = (h.get("content") or "").strip()
            source = (h.get("source") or "document").split(
                "\\")[-1].split("/")[-1]

            if not content:
                continue

            # Tronquer si trop long
            snippet = content[:600] + ("…" if len(content) > 600 else "")

            parts.append(f"\n**[#{i}] {subject}**")
            parts.append(f"{snippet}")
            parts.append(f"*Source: {source}*\n")

        parts.append(
            "\n---\n"
            "*Note: Réponse extraite automatiquement du contexte disponible. "
            "Pour une synthèse, assurez-vous que le LLM est correctement configuré.*"
        )

        return "\n".join(parts)
