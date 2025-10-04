"""
Module de chunking (découpage) des documents
"""
import math
from typing import List, Dict, Any, Set
from config.settings import TARGET_TOKENS, MAX_TOKENS, MIN_TOKENS, TOKENS_PER_WORD
from core.section_detector import SectionDetector
from utils.text_utils import (
    normalize_text,
    split_into_sentences,
    estimate_tokens,
    create_text_hash,
    find_text_overlap
)


class DocumentChunker:
    """Classe pour découper les documents en chunks sans duplication"""

    def __init__(self):
        self.section_detector = SectionDetector()

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Découpe un document en chunks de taille optimale
        
        Args:
            doc: Document avec id, content, source et predefined_sections
            
        Returns:
            Liste des chunks générés
        """
        # Nettoie le texte du document
        text = normalize_text(doc['content'])
        source = doc['source']

        # Détecte les sections
        predefined_sections = doc.get('predefined_sections', [])
        sections = self.section_detector.detect_sections(
            text, source, predefined_sections)

        # Crée les chunks
        chunks = self._create_small_chunks(text, source, sections)

        # Vérifie et supprime les duplications
        clean_chunks = self._remove_duplicates(chunks)

        return clean_chunks

    def _create_small_chunks(self, text: str, source: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Crée des petits chunks (40-60 tokens) sans duplication
        
        Args:
            text: Texte à découper
            source: Source du document
            sections: Sections du document
            
        Returns:
            Liste des chunks
        """
        # Divise le texte en phrases
        sentences = split_into_sentences(text)

        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        used_sentences = set()

        for sentence in sentences:
            # Ignore les phrases vides
            if not sentence.strip():
                continue

            # Évite les phrases déjà utilisées
            sentence_hash = create_text_hash(sentence)
            if sentence_hash in used_sentences:
                continue

            sentence_tokens = estimate_tokens(sentence)

            # Gère les phrases trop longues
            if sentence_tokens > MAX_TOKENS:
                # Finalise le chunk current s'il n'est pas vide
                if current_chunk_sentences:
                    self._finalize_chunk(current_chunk_sentences, current_token_count,
                                         text, source, sections, chunks, used_sentences)
                    current_chunk_sentences = []
                    current_token_count = 0

                # Divise la phrase longue
                self._split_long_sentence(
                    sentence, text, source, sections, chunks, used_sentences)
                continue

            # Vérifie si ajouter cette phrase dépasse la limite
            if current_token_count > 0 and current_token_count + sentence_tokens > MAX_TOKENS:
                # Finalise le chunk current
                self._finalize_chunk(current_chunk_sentences, current_token_count,
                                     text, source, sections, chunks, used_sentences)

                # Commence un nouveau chunk avec cette phrase
                current_chunk_sentences = [sentence]
                current_token_count = sentence_tokens
            else:
                # Ajoute la phrase au chunk current
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens

                # Finalise le chunk si on atteint la taille cible
                if current_token_count >= TARGET_TOKENS:
                    self._finalize_chunk(current_chunk_sentences, current_token_count,
                                         text, source, sections, chunks, used_sentences)
                    current_chunk_sentences = []
                    current_token_count = 0

        # Finalise le dernier chunk s'il reste des phrases
        if current_chunk_sentences:
            self._finalize_chunk(current_chunk_sentences, current_token_count,
                                 text, source, sections, chunks, used_sentences)

        # Assigne des IDs uniques
        for i, chunk in enumerate(chunks):
            chunk['id'] = i + 1

        return chunks

    def _split_long_sentence(self, sentence: str, text: str, source: str,
                             sections: List[Dict[str, Any]], chunks: List[Dict[str, Any]],
                             used_sentences: Set[str]) -> None:
        """
        Divise une phrase trop longue en morceaux plus petits
        
        Args:
            sentence: Phrase à diviser
            text: Texte complet (pour identifier les sections)
            source: Source du document
            sections: Sections du document
            chunks: Liste des chunks (modifiée)
            used_sentences: Phrases déjà utilisées (modifiée)
        """
        words = sentence.split()
        target_word_count = math.floor(TARGET_TOKENS / TOKENS_PER_WORD)

        for i in range(0, len(words), target_word_count):
            sub_words = words[i:min(i + target_word_count, len(words))]
            if sub_words:
                sub_text = ' '.join(sub_words)
                sub_tokens = estimate_tokens(sub_text)

                # Ajoute seulement si assez de tokens ou dernier morceau
                if sub_tokens >= MIN_TOKENS or i + target_word_count >= len(words):
                    # Trouve la position pour identifier la section
                    pos = text.find(sub_text)
                    if pos == -1:
                        pos = 0

                    section = self.section_detector.identify_section(
                        pos, sections)

                    chunks.append({
                        'content': sub_text,
                        'source': source,
                        'section': section,
                        'tokens': sub_tokens
                    })

        # Marque la phrase entière comme utilisée
        used_sentences.add(create_text_hash(sentence))

    def _finalize_chunk(self, sentences: List[str], token_count: int, text: str,
                        source: str, sections: List[Dict[str, Any]],
                        chunks: List[Dict[str, Any]], used_sentences: Set[str]) -> None:
        """
        Finalise un chunk et l'ajoute à la liste
        
        Args:
            sentences: Phrases du chunk
            token_count: Nombre de tokens
            text: Texte complet
            source: Source du document
            sections: Sections du document
            chunks: Liste des chunks (modifiée)
            used_sentences: Phrases utilisées (modifiée)
        """
        if not sentences:
            return

        chunk_text = ' '.join(sentences)

        # Trouve la position pour identifier la section
        pos = text.find(sentences[0])
        if pos == -1:
            pos = 0

        section = self.section_detector.identify_section(pos, sections)

        chunks.append({
            'content': chunk_text,
            'source': source,
            'section': section,
            'tokens': token_count
        })

        # Marque les phrases comme utilisées
        for sentence in sentences:
            used_sentences.add(create_text_hash(sentence))

    def _remove_duplicates(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Supprime les duplications entre chunks
        
        Args:
            chunks: Liste des chunks à nettoyer
            
        Returns:
            Liste des chunks sans duplication
        """
        # Supprime d'abord les chunks complètement identiques
        unique_chunks = {}

        for chunk in chunks:
            content = chunk.get('content', '').strip()

            if not content:
                continue

            content_hash = create_text_hash(content)

            if content_hash not in unique_chunks:
                unique_chunks[content_hash] = chunk

        clean_chunks = list(unique_chunks.values())

        # Supprime les chevauchements partiels entre chunks adjacents
        result = []
        clean_chunks.sort(key=lambda x: x.get('id', 0))

        for i in range(len(clean_chunks)):
            current = clean_chunks[i]
            current_content = current.get('content', '')

            # Vérifie le chevauchement avec le chunk précédent
            if i > 0:
                previous = clean_chunks[i-1]
                previous_content = previous.get('content', '')

                # Trouve le chevauchement
                overlap_text = find_text_overlap(
                    previous_content, current_content)

                # Supprime le chevauchement significatif
                if overlap_text and len(overlap_text) > 10:
                    current_content = current_content[len(
                        overlap_text):].strip()
                    current['content'] = current_content
                    current['tokens'] = estimate_tokens(current_content)

            # Ajoute le chunk s'il n'est pas vide
            if current_content.strip():
                result.append(current)

        # Réassigne des IDs séquentiels
        for i, chunk in enumerate(result):
            chunk['id'] = i + 1
            # Supprime les informations de debug
            if 'tokens' in chunk:
                del chunk['tokens']

        return result

    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcule les statistiques du chunking
        
        Args:
            chunks: Liste des chunks
            
        Returns:
            Statistiques du chunking
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_tokens': 0,
                'min_tokens': 0,
                'max_tokens': 0,
                'target_range': f"{MIN_TOKENS}-{MAX_TOKENS}"
            }

        tokens = [estimate_tokens(chunk.get('content', ''))
                  for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_tokens': sum(tokens) / len(tokens),
            'min_tokens': min(tokens),
            'max_tokens': max(tokens),
            'target_range': f"{MIN_TOKENS}-{MAX_TOKENS}"
        }
