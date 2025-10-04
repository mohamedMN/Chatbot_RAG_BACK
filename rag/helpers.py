"""
Fonctions utilitaires pour le système RAG
Compatible avec votre code existant
"""
import re
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extrait les mots-clés d'un texte pour le filtrage lexical
    Compatible avec votre code existant
    
    Args:
        text: Texte à analyser
        min_length: Longueur minimale des mots-clés
        
    Returns:
        Liste des mots-clés importants
    """
    if not text:
        return []

    # Nettoie le texte
    text = text.lower().strip()

    # Supprime la ponctuation mais garde les espaces
    text = re.sub(r'[^\w\sàâäéèêëïîôöùûüÿç-]', ' ', text)

    # Mots vides français et anglais (étendus pour webMethods/Software AG)
    stop_words = {
        # Français
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais',
        'pour', 'par', 'avec', 'dans', 'sur', 'sous', 'vers', 'chez', 'sans',
        'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous',
        'que', 'qui', 'quoi', 'dont', 'où', 'comment', 'pourquoi', 'quand',
        'est', 'sont', 'être', 'avoir', 'faire', 'dit', 'peut', 'doit',
        # Anglais
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        # Mots techniques génériques
        'configuration', 'installation', 'setup', 'guide', 'documentation',
        'example', 'exemple', 'note', 'important', 'attention'
    }

    # Extrait les mots
    words = re.findall(r'\b\w+\b', text)

    # Filtre et nettoie
    keywords = []
    for word in words:
        word = word.strip()
        if (len(word) >= min_length and
            word not in stop_words and
            not word.isdigit() and
                not re.match(r'^\d+\.\d+$', word)):  # Évite les versions comme 1.2
            keywords.append(word)

    # Priorise les termes spécifiques webMethods/Software AG
    technical_terms = [
        'webmethods', 'software', 'ag', 'integration', 'server', 'broker',
        'adapters', 'trading', 'networks', 'designer', 'developer',
        'flow', 'service', 'package', 'namespace', 'pipeline', 'document',
        'api', 'rest', 'soap', 'http', 'jms', 'jdbc', 'xml', 'json',
        'environment', 'prod', 'dev', 'test', 'staging'
    ]

    # Compte les occurrences
    word_counts = Counter(keywords)

    # Priorise les termes techniques et fréquents
    prioritized = []
    for word, count in word_counts.most_common():
        if word in technical_terms:
            prioritized.insert(0, word)  # Met en tête
        else:
            prioritized.append(word)

    return prioritized[:20]  # Limite à 20 mots-clés


def norm(vectors: np.ndarray) -> np.ndarray:
    """
    Normalise des vecteurs pour cosine similarity
    Compatible avec votre code existant
    
    Args:
        vectors: Matrice de vecteurs
        
    Returns:
        Vecteurs normalisés
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Évite la division par zéro
    return vectors / norms


def filter_hits_by_keywords(hits: List[Dict], keywords: List[str],
                            min_keyword_matches: int = 1) -> List[Dict]:
    """
    Filtre les résultats par correspondance lexicale
    Compatible avec votre code existant
    
    Args:
        hits: Liste des résultats de recherche
        keywords: Mots-clés à rechercher
        min_keyword_matches: Nombre minimum de correspondances
        
    Returns:
        Résultats filtrés
    """
    if not keywords or not hits:
        return hits

    filtered_hits = []

    for hit in hits:
        # Combine le contenu, sujet et source pour la recherche
        searchable_text = " ".join([
            hit.get("content", ""),
            hit.get("subject", ""),
            hit.get("source", "")
        ]).lower()

        # Compte les correspondances de mots-clés
        matches = 0
        for keyword in keywords:
            if keyword.lower() in searchable_text:
                matches += 1

        # Garde si assez de correspondances ou score très élevé
        if matches >= min_keyword_matches or hit.get("score", 0) > 0.85:
            hit["keyword_matches"] = matches
            filtered_hits.append(hit)

    # Trie par nombre de correspondances puis par score
    filtered_hits.sort(
        key=lambda h: (-h.get("keyword_matches", 0), -h.get("score", 0)))

    return filtered_hits


def format_context_for_llm(hits: List[Dict], max_chars: int = 1500) -> str:
    """
    Formate le contexte récupéré pour le LLM
    
    Args:
        hits: Résultats de la recherche
        max_chars: Nombre maximum de caractères
        
    Returns:
        Contexte formaté avec numérotation
    """
    if not hits:
        return "(aucun extrait)"

    context_parts = []
    current_length = 0

    for i, hit in enumerate(hits, 1):
        content = hit.get("content", "").strip()
        source = hit.get("source", "").split(
            "/")[-1]  # Nom du fichier seulement
        score = hit.get("score", 0)

        if not content:
            continue

        # Format: [#n] contenu (source, score)
        formatted = f"[#{i}] {content}"
        if source:
            formatted += f" (source: {source}, score: {score:.2f})"

        # Vérifie la limite de caractères
        if current_length + len(formatted) > max_chars:
            if not context_parts:  # Assure au moins un extrait
                context_parts.append(formatted[:max_chars-50] + "...")
            break

        context_parts.append(formatted)
        current_length += len(formatted) + 2  # +2 pour \n\n

        # Limite le nombre d'extraits
        if i >= 5:
            break

    if not context_parts:
        return "(aucun extrait)"

    return "\n\n".join(context_parts)


def calculate_relevance_score(hit: Dict, keywords: List[str]) -> float:
    """
    Calcule un score de pertinence combiné
    
    Args:
        hit: Résultat de recherche
        keywords: Mots-clés de la requête
        
    Returns:
        Score de pertinence combiné (0-1)
    """
    vector_score = hit.get("score", 0.0)
    keyword_matches = hit.get("keyword_matches", 0)

    # Bonus pour les correspondances de mots-clés
    keyword_bonus = min(keyword_matches * 0.1, 0.3)

    # Bonus pour les sources techniques (webMethods, etc.)
    source = hit.get("source", "").lower()
    source_bonus = 0.0
    if any(term in source for term in ['webmethods', 'software', 'ag', 'integration']):
        source_bonus = 0.05

    # Score final (plafonné à 1.0)
    final_score = min(vector_score + keyword_bonus + source_bonus, 1.0)

    return final_score


def deduplicate_hits(hits: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
    """
    Supprime les doublons basés sur la similarité de contenu
    
    Args:
        hits: Liste des résultats
        similarity_threshold: Seuil de similarité pour considérer comme doublon
        
    Returns:
        Liste dédupliquée
    """
    if len(hits) <= 1:
        return hits

    unique_hits = []

    for current_hit in hits:
        current_content = current_hit.get("content", "").lower()
        is_duplicate = False

        for unique_hit in unique_hits:
            unique_content = unique_hit.get("content", "").lower()

            # Calcul de similarité simple basé sur les mots communs
            current_words = set(current_content.split())
            unique_words = set(unique_content.split())

            if not current_words or not unique_words:
                continue

            intersection = len(current_words & unique_words)
            union = len(current_words | unique_words)
            similarity = intersection / union if union > 0 else 0

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_hits.append(current_hit)

    return unique_hits


def validate_hit(hit: Dict) -> bool:
    """
    Valide qu'un résultat de recherche est utilisable
    
    Args:
        hit: Résultat à valider
        
    Returns:
        True si valide
    """
    # Vérifie les champs obligatoires
    required_fields = ["content", "score"]
    if not all(field in hit for field in required_fields):
        return False

    # Vérifie que le contenu n'est pas vide
    content = hit.get("content", "").strip()
    if len(content) < 10:  # Minimum 10 caractères
        return False

    # Vérifie que le score est valide
    score = hit.get("score", 0)
    if not isinstance(score, (int, float)) or score < 0:
        return False

    return True
