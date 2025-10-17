from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import re
from difflib import SequenceMatcher


# rag/helpers.py



def norm(vec: np.ndarray) -> np.ndarray:
    """
    L2-normalize. Accepts 1D or 2D input and always returns a 2D array.
    - 1D (D,)   -> (1, D)
    - 2D (N, D) -> (N, D)
    """
    a = np.asarray(vec, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    # guard against zero norm
    n[n == 0] = 1.0
    return a / n




def sources_list(hits: List[Dict], max_items: int = 8) -> str:
    """
    Returns a compact “Sources” list (one line per unique file).
    """
    seen = set()
    lines: List[str] = []
    for i, h in enumerate(hits, 1):
        src = str(h.get("source", "")).replace("\\", "/").split("/")[-1]
        if not src or src in seen:
            continue
        seen.add(src)
        lines.append(f"[#{i}] {src}")
        if len(lines) >= max_items:
            break
    return "\n".join(lines)

def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text"""
    if not text:
        return []

    # Convert to lowercase and extract words
    words = re.findall(r'\b\w+\b', text.lower())

    # French and English stopwords
    stopwords = {
        # French
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou',
        'à', 'pour', 'dans', 'sur', 'avec', 'par', 'ce', 'ces', 'qui',
        'que', 'est', 'sont', 'être', 'avoir', 'faire', 'dit', 'fait',
        'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
        'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
        'cette', 'cet', 'celui', 'celle', 'ceux', 'celles',
        # English
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
        'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'them', 'their', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'some',
        'any', 'few', 'more', 'most', 'other', 'such', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'now'
    }

    # Common question words to filter out
    question_words = {
        "qu'est", "qu'est-ce", "quoi", "comment", "pourquoi",
        "quand", "où", "qui", "quel", "quelle", "quels", "quelles",
        "what", "how", "why", "when", "where", "who", "which"
    }

    # Filter words
    keywords = []
    for word in words:
        if (len(word) > 2 and
            word not in stopwords and
            word not in question_words and
                not word.isdigit()):
            keywords.append(word)

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords


def filter_hits_by_keywords(
    hits: List[Dict[str, Any]],
    keywords: List[str],
    min_matches: int = 1
) -> List[Dict[str, Any]]:
    """Filter hits based on keyword presence"""
    if not keywords or not hits:
        return hits

    filtered = []
    for hit in hits:
        # Combine all searchable text
        searchable = f"{hit.get('subject', '')} {hit.get('content', '')} {hit.get('source', '')}".lower(
        )

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw.lower() in searchable)

        if matches >= min_matches:
            hit['keyword_matches'] = matches
            filtered.append(hit)

    # If too few results, return original hits
    if len(filtered) < 2 and len(hits) > 0:
        return hits[:5]  # Return top 5 original hits

    return filtered


def deduplicate_hits(
    hits: List[Dict[str, Any]],
    similarity_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """Remove duplicate or very similar hits"""
    if len(hits) <= 1:
        return hits

    unique_hits = []

    for hit in hits:
        is_duplicate = False
        hit_content = hit.get('content', '').lower()

        for unique_hit in unique_hits:
            unique_content = unique_hit.get('content', '').lower()

            # Calculate similarity
            similarity = SequenceMatcher(
                None, hit_content, unique_content).ratio()

            if similarity > similarity_threshold:
                is_duplicate = True
                # Keep the one with higher score
                if hit.get('score', 0) > unique_hit.get('score', 0):
                    unique_hits.remove(unique_hit)
                    unique_hits.append(hit)
                break

        if not is_duplicate:
            unique_hits.append(hit)

    return unique_hits


def calculate_relevance_score(
    hit: Dict[str, Any],
    keywords: List[str]
) -> float:
    """Calculate relevance score based on multiple factors"""
    score = 0.0

    content = f"{hit.get('subject', '')} {hit.get('content', '')}".lower()

    if not keywords:
        return hit.get('score', 0.0)

    # Keyword presence (weighted by position)
    for i, kw in enumerate(keywords):
        kw_lower = kw.lower()
        if kw_lower in content:
            # Earlier keywords are more important
            weight = 1.0 - (i * 0.1)
            score += 0.3 * max(0.5, weight)

    # Exact phrase matches
    if len(keywords) > 1:
        phrase = " ".join(keywords[:3]).lower()
        if phrase in content:
            score += 0.4

    # Length penalty (too short or too long)
    content_length = len(hit.get('content', ''))
    if 100 <= content_length <= 1000:
        score += 0.1
    elif content_length < 50:
        score -= 0.2

    # Combine with original score
    original_score = hit.get('score', 0.0)
    combined_score = (original_score * 0.6) + (score * 0.4)

    return min(1.0, max(0.0, combined_score))


def validate_hit(hit: Dict[str, Any]) -> bool:
    """Validate that a hit has required fields and reasonable content"""
    required_fields = ['score', 'id', 'content', 'subject', 'source']

    # Check required fields
    for field in required_fields:
        if field not in hit:
            return False

    # Check content is not empty
    if not hit.get('content', '').strip():
        return False

    # Check score is reasonable
    score = hit.get('score', 0)
    if not isinstance(score, (int, float)) or score < 0 or score > 10:
        return False

    return True


def format_context_for_llm(
    hits: List[Dict[str, Any]],
    max_chars: int = 1500
) -> str:
    """Format context hits for LLM consumption"""
    if not hits:
        return "Aucun contexte pertinent trouvé."

    context_parts = []
    current_length = 0

    for i, hit in enumerate(hits, 1):
        subject = hit.get('subject', 'Information')
        content = hit.get('content', '').strip()
        source = hit.get('source', 'Document')

        # Extract just filename from path
        if '\\' in source:
            source = source.split('\\')[-1]
        elif '/' in source:
            source = source.split('/')[-1]

        if not content:
            continue

        # Format hit
        formatted = f"[#{i}] {subject}\n{content}\n(Source: {source})\n"

        # Check length
        if current_length + len(formatted) > max_chars:
            if i == 1:
                # Truncate first hit if it's too long
                available = max_chars - current_length - 50
                content = content[:available] + "..."
                formatted = f"[#{i}] {subject}\n{content}\n(Source: {source})\n"
            else:
                break

        context_parts.append(formatted)
        current_length += len(formatted)

    if not context_parts:
        return "Aucun contexte pertinent trouvé."

    return "\n---\n".join(context_parts)
