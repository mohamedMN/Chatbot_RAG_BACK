"""
Utilitaires pour le traitement de texte
"""
import re
import hashlib
from typing import List
from config.settings import TOKENS_PER_WORD


def normalize_text(text: str) -> str:
    """
    Normalise et nettoie le texte
    
    Args:
        text: Texte à normaliser
        
    Returns:
        Texte normalisé
    """
    if not text:
        return ""

    # Remplace les espaces multiples par un seul espace
    text = re.sub(r'\s+', ' ', text)
    # Supprime les caractères spéciaux mais garde la ponctuation importante
    text = re.sub(r'[^\w\s.,;:!?()"-]', '', text)
    # Supprime les espaces en début et fin
    return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Estime le nombre de tokens dans un texte
    
    Args:
        text: Texte d'entrée
        
    Returns:
        Nombre estimé de tokens
    """
    if not text:
        return 0

    words = text.split()
    return int(len(words) * TOKENS_PER_WORD)


def split_into_sentences(text: str) -> List[str]:
    """
    Divise le texte en phrases de manière précise
    
    Args:
        text: Texte à diviser
        
    Returns:
        Liste des phrases
    """
    if not text:
        return []

    # Abréviations courantes pour éviter les mauvaises coupures
    abbreviations = ['M.', 'Mme.', 'Dr.', 'Prof.',
                     'etc.', 'ex.', 'Fig.', 'fig.', 'e.g.', 'i.e.']

    # Remplace temporairement les abréviations pour éviter les fausses coupures
    for abbr in abbreviations:
        text = text.replace(abbr, abbr.replace('.', '<PD>'))

    # Divise sur la ponctuation de fin de phrase
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)

    # Restaure les abréviations
    sentences = [s.replace('<PD>', '.') for s in sentences]

    # Filtre les phrases vides
    return [s.strip() for s in sentences if s.strip()]


def create_text_hash(text: str) -> str:
    """
    Crée un hash unique pour un texte
    
    Args:
        text: Texte à hasher
        
    Returns:
        Hash MD5 du texte
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def find_text_overlap(text1: str, text2: str, min_length: int = 10) -> str:
    """
    Trouve le chevauchement entre deux textes
    
    Args:
        text1: Premier texte
        text2: Deuxième texte
        min_length: Longueur minimale pour considérer un chevauchement
        
    Returns:
        Texte en chevauchement ou chaîne vide
    """
    if not text1 or not text2:
        return ""

    max_overlap_len = min(len(text1), len(text2)) // 2

    for overlap_size in range(max_overlap_len, min_length - 1, -1):
        if text1[-overlap_size:] == text2[:overlap_size]:
            return text2[:overlap_size]

    return ""


def clean_section_title(title: str) -> str:
    """
    Nettoie le titre d'une section
    
    Args:
        title: Titre brut
        
    Returns:
        Titre nettoyé
    """
    if not title:
        return "Section Sans Titre"

    # Supprime les caractères de formatage
    title = re.sub(r'^#+\s*', '', title)  # Markdown headers
    title = re.sub(r'^\d+\.\s*', '', title)  # Numbered headers
    title = title.strip()

    # Limite la longueur
    if len(title) > 80:
        title = title[:77] + "..."

    return title or "Section Sans Titre"
