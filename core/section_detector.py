"""
Module de détection des sections dans les documents
"""
import re
from typing import List, Dict, Any
from pathlib import Path
from config.settings import HEADING_PATTERNS, SECTION_MAX_TITLE_LENGTH
from utils.text_utils import clean_section_title


class SectionDetector:
    """Détecteur de sections dans les documents"""

    def __init__(self):
        self.heading_patterns = [re.compile(
            pattern, re.IGNORECASE) for pattern in HEADING_PATTERNS]

    def detect_sections(self, text: str, source: str, predefined_sections: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Détecte les sections dans un texte
        
        Args:
            text: Texte du document
            source: Source du document
            predefined_sections: Sections prédéfinies (ex: depuis DOCX/PDF)
            
        Returns:
            Liste des sections avec positions de début et fin
        """
        # Si on a des sections prédéfinies, on les utilise
        if predefined_sections and len(predefined_sections) > 0:
            return self._validate_predefined_sections(predefined_sections, text)

        # Sinon, on détecte depuis le texte brut
        return self._detect_from_text(text, source)

    def _validate_predefined_sections(self, sections: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Valide et nettoie les sections prédéfinies
        
        Args:
            sections: Sections prédéfinies
            text: Texte complet
            
        Returns:
            Sections validées
        """
        validated_sections = []
        text_length = len(text)

        for section in sections:
            # Valide les positions
            start = max(0, section.get('start', 0))
            end = min(text_length, section.get('end', text_length))

            if start < end:
                validated_sections.append({
                    'title': clean_section_title(section.get('title', 'Section')),
                    'start': start,
                    'end': end
                })

        # Assure-t-on que les sections couvrent tout le texte
        if not validated_sections:
            return self._create_default_section(text, "Document")

        return validated_sections

    def _detect_from_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Détecte les sections depuis le texte brut
        
        Args:
            text: Texte à analyser
            source: Source du document
            
        Returns:
            Sections détectées
        """
        sections = []
        lines = text.split('\n')
        current_position = 0
        section_start = 0
        current_section_title = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if line_stripped:
                is_heading, title = self._is_heading(line_stripped, lines, i)

                if is_heading:
                    # Finalise la section précédente
                    if current_position > section_start:
                        sections.append({
                            'title': clean_section_title(current_section_title or "Section"),
                            'start': section_start,
                            'end': current_position
                        })

                    # Commence une nouvelle section
                    section_start = current_position
                    current_section_title = title

            current_position += len(line) + 1  # +1 pour le retour à la ligne

        # Finalise la dernière section
        if current_position > section_start:
            sections.append({
                'title': clean_section_title(current_section_title or "Section Principale"),
                'start': section_start,
                'end': current_position
            })

        # Si aucune section trouvée, crée une section par défaut
        if not sections:
            return self._create_default_section(text, Path(source).stem)

        return sections

    def _is_heading(self, line: str, all_lines: List[str], line_index: int) -> tuple[bool, str]:
        """
        Détermine si une ligne est un titre
        
        Args:
            line: Ligne à analyser
            all_lines: Toutes les lignes du document
            line_index: Index de la ligne courante
            
        Returns:
            Tuple (is_heading, title_text)
        """
        # Vérifie les patterns d'environnement spécifiques
        env_match = re.match(
            r'^(Environnement\s+[A-Z0-9]+(?:\s*\([^)]+\))?\s*:)', line)
        if env_match:
            return True, env_match.group(1).strip()

        # Vérifie "Description Technique"
        if re.search(r'description\s+technique\s*:', line.lower()):
            job_match = re.match(
                r'^(.*?)(?:description\s+technique\s*:)', line.lower())
            if job_match:
                return True, job_match.group(1).strip()
            else:
                return True, line

        # Vérifie les patterns génériques
        for pattern in self.heading_patterns:
            if pattern.match(line):
                return True, line

        # Vérifie si c'est une ligne courte isolée (potentiel titre)
        if (len(line) < SECTION_MAX_TITLE_LENGTH and
                self._is_isolated_line(all_lines, line_index)):
            return True, line

        return False, line

    def _is_isolated_line(self, lines: List[str], index: int) -> bool:
        """
        Vérifie si une ligne est isolée (entourée de lignes vides)
        
        Args:
            lines: Toutes les lignes
            index: Index de la ligne
            
        Returns:
            True si la ligne est isolée
        """
        # Vérifie la ligne précédente
        prev_empty = (index == 0 or not lines[index-1].strip())
        # Vérifie la ligne suivante
        next_empty = (index == len(lines)-1 or not lines[index+1].strip())

        return prev_empty and next_empty

    def _create_default_section(self, text: str, title: str) -> List[Dict[str, Any]]:
        """
        Crée une section par défaut pour tout le document
        
        Args:
            text: Texte complet
            title: Titre par défaut
            
        Returns:
            Liste avec une section unique
        """
        return [{
            'title': clean_section_title(title.replace('_', ' ').replace('-', ' ').title()),
            'start': 0,
            'end': len(text)
        }]

    def identify_section(self, position: int, sections: List[Dict[str, Any]]) -> str:
        """
        Identifie la section à une position donnée
        
        Args:
            position: Position dans le texte
            sections: Liste des sections
            
        Returns:
            Titre de la section
        """
        # Cherche la section qui contient cette position
        for section in sections:
            if section['start'] <= position <= section['end']:
                return section['title']

        # Si pas trouvé, cherche la section précédente la plus proche
        closest_section = None
        closest_distance = float('inf')

        for section in sections:
            if position > section['end']:
                distance = position - section['end']
                if distance < closest_distance:
                    closest_distance = distance
                    closest_section = section

        if closest_section:
            return closest_section['title']

        return "Contenu Principal"
