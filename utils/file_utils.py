"""
Utilitaires pour la gestion des fichiers
"""
import json
import os
from pathlib import Path
from typing import Union, Dict, List, Any


def save_json(data: Union[Dict, List], filepath: Union[str, Path]) -> None:
    """
    Sauvegarde des données au format JSON
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin du fichier de destination
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """
    Charge des données depuis un fichier JSON
    
    Args:
        filepath: Chemin du fichier JSON
        
    Returns:
        Données chargées
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_supported_files(directory: Union[str, Path], extensions: List[str]) -> List[Path]:
    """
    Récupère tous les fichiers supportés d'un dossier
    
    Args:
        directory: Dossier à scanner
        extensions: Extensions supportées
        
    Returns:
        Liste des fichiers trouvés
    """
    directory = Path(directory)
    supported_files = []

    if not directory.exists():
        return supported_files

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            supported_files.append(file_path)

    return supported_files


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    S'assure qu'un dossier existe
    
    Args:
        path: Chemin du dossier
        
    Returns:
        Path object du dossier
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Récupère les informations d'un fichier
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Informations du fichier
    """
    filepath = Path(filepath)

    return {
        'name': filepath.name,
        'stem': filepath.stem,
        'suffix': filepath.suffix.lower(),
        'size': filepath.stat().st_size if filepath.exists() else 0,
        'exists': filepath.exists()
    }
