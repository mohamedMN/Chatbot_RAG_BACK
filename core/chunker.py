# core/chunker.py
from __future__ import annotations
import math
import hashlib
import re
from typing import List, Dict, Any, Set, Tuple

# ---- Config fallbacks (use your real settings if present) ----
try:
    from config.settings import TARGET_TOKENS, MAX_TOKENS, MIN_TOKENS, TOKENS_PER_WORD
except Exception:
    TOKENS_PER_WORD = 0.75      # rough heuristic
    TARGET_TOKENS = 48          # ~64 words
    MIN_TOKENS = 16
    MAX_TOKENS = 64

# ---- Optional SectionDetector (use yours if present) ----
try:
    from core.section_detector import SectionDetector
except Exception:
    class SectionDetector:
        def detect_sections(self, text: str, source: str, predefined_sections: List[Dict[str, Any]] = None):
            return [{"title": "Section Principale", "start": 0, "end": len(text)}]

        def identify_section(self, position: int, sections: List[Dict[str, Any]]) -> str:
            for s in sections:
                if s["start"] <= position <= s["end"]:
                    return s["title"]
            return "Section Principale"

# ---- Minimal text utils (use yours if present) ----


def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive spaces
    s = re.sub(r"[ \t]+", " ", s)
    # collapse 3+ blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def _split_into_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # preserve newlines as sentence boundaries too
    lines = [ln.strip() for ln in text.split("\n")]
    out: List[str] = []
    for line in lines:
        if not line:
            continue
        parts = _SENT_SPLIT.split(line) or [line]
        out.extend([p.strip() for p in parts if p.strip()])
    return out


def _estimate_tokens(s: str) -> int:
    if not s:
        return 0
    # simple heuristic instead of a tokenizer
    words = len(s.split())
    return max(1, int(words * TOKENS_PER_WORD))


def _create_text_hash(s: str) -> str:
    return hashlib.sha1((s or "").strip().encode("utf-8")).hexdigest()


def _find_text_overlap(a: str, b: str, min_len: int = 12) -> str:
    """
    Return prefix of b that overlaps suffix of a (simple O(n) scan).
    """
    a, b = a or "", b or ""
    max_overlap = min(len(a), len(b))
    best = ""
    for k in range(max_overlap, min_len - 1, -1):
        if a.endswith(b[:k]):
            best = b[:k]
            break
    return best

# ----------------------------------------------------------------


class DocumentChunker:
    """
    Deterministic chunker with GLOBAL unique IDs across all documents.
    """

    def __init__(self):
        self.section_detector = SectionDetector()
        self._global_chunk_counter = 0  # ← Compteur global

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Input doc must contain at least: {'content': str, 'source': str}
        Optional: 'predefined_sections': List[{'title','start','end'}]
        """
        text = _normalize_text(doc.get("content", ""))
        source = doc.get("source", "")

        predefined_sections = doc.get("predefined_sections", []) or []
        if predefined_sections:
            sections = self.section_detector.detect_sections(
                text, source, predefined_sections)
        else:
            sections = self.section_detector.detect_sections(text, source)

        chunks = self._create_chunks(text, source, sections)
        chunks = self._remove_duplicates_and_trim_overlaps(chunks)

        # Assign GLOBAL sequential IDs (incremental across all documents)
        for ch in chunks:
            self._global_chunk_counter += 1
            ch["id"] = self._global_chunk_counter  # ← ID global unique
            # strip debug tokens if present
            if "tokens" in ch:
                del ch["tokens"]

        return chunks

    # ------------ internals ------------

    def _create_chunks(self, text: str, source: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sentences = _split_into_sentences(text)
        chunks: List[Dict[str, Any]] = []
        cur_sents: List[str] = []
        cur_tokens = 0
        used_hashes: Set[str] = set()

        for sent in sentences:
            if not sent.strip():
                continue
            sh = _create_text_hash(sent)
            if sh in used_hashes:
                continue

            t = _estimate_tokens(sent)

            # split extra-long sentences
            if t > MAX_TOKENS:
                if cur_sents:
                    self._finalize_chunk(
                        cur_sents, cur_tokens, text, source, sections, chunks, used_hashes)
                    cur_sents, cur_tokens = [], 0
                self._split_long_sentence(
                    sent, text, source, sections, chunks, used_hashes)
                continue

            # would overflow window?
            if cur_tokens > 0 and cur_tokens + t > MAX_TOKENS:
                self._finalize_chunk(
                    cur_sents, cur_tokens, text, source, sections, chunks, used_hashes)
                cur_sents, cur_tokens = [sent], t
            else:
                cur_sents.append(sent)
                cur_tokens += t
                if cur_tokens >= TARGET_TOKENS:
                    self._finalize_chunk(
                        cur_sents, cur_tokens, text, source, sections, chunks, used_hashes)
                    cur_sents, cur_tokens = [], 0

        if cur_sents:
            self._finalize_chunk(cur_sents, cur_tokens,
                                 text, source, sections, chunks, used_hashes)

        return chunks

    def _split_long_sentence(
        self, sentence: str, text: str, source: str,
        sections: List[Dict[str, Any]], chunks: List[Dict[str, Any]],
        used_hashes: Set[str]
    ) -> None:
        words = sentence.split()
        target_words = max(4, math.floor(
            TARGET_TOKENS / max(1e-6, TOKENS_PER_WORD)))
        for i in range(0, len(words), target_words):
            sub = " ".join(words[i:i+target_words]).strip()
            if not sub:
                continue
            tk = _estimate_tokens(sub)
            # keep last piece even if tiny to preserve content
            if tk >= MIN_TOKENS or (i + target_words) >= len(words):
                pos = text.find(sub)
                if pos < 0:
                    pos = 0
                section = self.section_detector.identify_section(pos, sections)
                chunks.append({
                    "content": sub,
                    "source": source,
                    "section": section,
                    "tokens": tk
                })
        used_hashes.add(_create_text_hash(sentence))

    def _finalize_chunk(
        self, sentences: List[str], token_count: int, text: str,
        source: str, sections: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]], used_hashes: Set[str]
    ) -> None:
        if not sentences:
            return
        chunk_text = " ".join(sentences).strip()
        if not chunk_text:
            return
        pos = text.find(sentences[0])
        if pos < 0:
            pos = 0
        section = self.section_detector.identify_section(pos, sections)
        chunks.append({
            "content": chunk_text,
            "source": source,
            "section": section,
            "tokens": int(token_count)
        })
        for s in sentences:
            used_hashes.add(_create_text_hash(s))

    def _remove_duplicates_and_trim_overlaps(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"DEBUG: Chunks avant déduplication: {len(chunks)}")

        # remove exact duplicates
        unique: Dict[str, Dict[str, Any]] = {}
        for ch in chunks:
            content = (ch.get("content") or "").strip()
            if not content:
                continue
            h = _create_text_hash(content)
            if h not in unique:
                unique[h] = dict(ch)
        clean = list(unique.values())

        # trim overlaps between adjacent chunks
        result: List[Dict[str, Any]] = []
        for i, cur in enumerate(clean):
            cur_txt = (cur.get("content") or "").strip()
            if not cur_txt:
                continue
            if result:
                prev_txt = result[-1]["content"]
                ov = _find_text_overlap(prev_txt, cur_txt, min_len=12)
                if ov:
                    cur_txt = cur_txt[len(ov):].strip()
                    cur["content"] = cur_txt
                    cur["tokens"] = _estimate_tokens(cur_txt)
            if cur_txt:
                result.append(cur)

        print(f"DEBUG: Chunks après déduplication: {len(result)}")
        return result

    # Optional stats API
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {"total_chunks": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0,
                    "target_range": f"{MIN_TOKENS}-{MAX_TOKENS}"}
        toks = [_estimate_tokens(ch.get("content", "")) for ch in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(toks) / len(toks),
            "min_tokens": min(toks),
            "max_tokens": max(toks),
            "target_range": f"{MIN_TOKENS}-{MAX_TOKENS}",
        }
