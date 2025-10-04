"""
Gestionnaire d'index FAISS pour la recherche vectorielle
"""
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS non disponible. Installez avec: pip install faiss-cpu")

from config.settings import settings
from utils.file_utils import save_json, load_json


class FAISSIndexer:
    """Gestionnaire d'index FAISS local pour RAG"""

    def __init__(self):
        self.index = None
        self.idmap = None
        self.metadata = None
        self.dimension = settings.get_faiss_config()["dimension"]

        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS requis pour l'indexing. Installez: pip install faiss-cpu")

    def create_index(self, embeddings_dict: Dict[str, Dict[str, Any]],
                     index_type: str = "IndexFlatIP") -> None:
        """
        Crée un nouvel index FAISS
        
        Args:
            embeddings_dict: Dictionnaire des embeddings avec métadonnées
            index_type: Type d'index FAISS (IndexFlatIP, IndexIVFFlat, etc.)
        """
        if not embeddings_dict:
            raise ValueError("Aucun embedding fourni pour créer l'index")

        print(f"Création de l'index FAISS {index_type}...")

        # Prépare les données
        embeddings_matrix, idmap_data = self._prepare_data(embeddings_dict)

        # Crée l'index FAISS selon le type
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type.startswith("IndexIVF"):
            # Index avec quantification pour de gros datasets
            nlist = min(100, len(embeddings_matrix) //
                        10)  # Nombre de clusters
            if index_type == "IndexIVFFlat":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist)
            else:
                raise ValueError(f"Type d'index non supporté: {index_type}")

            # Entraîne l'index IVF
            self.index.train(embeddings_matrix)
        else:
            raise ValueError(f"Type d'index non supporté: {index_type}")

        # Normalise les vecteurs si nécessaire (pour cosine similarity avec IP)
        if settings.get_faiss_config()["normalize"] and "IP" in index_type:
            faiss.normalize_L2(embeddings_matrix)

        # Ajoute les embeddings à l'index
        self.index.add(embeddings_matrix)

        # Stocke les métadonnées
        self.idmap = idmap_data
        self.metadata = {
            "index_type": index_type,
            "dimension": self.dimension,
            "total_vectors": len(embeddings_matrix),
            "metric": "ip" if "IP" in index_type else "l2",
            "normalized": settings.get_faiss_config()["normalize"] and "IP" in index_type
        }

        print(f"Index créé avec {len(embeddings_matrix)} vecteurs")

    def _prepare_data(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Dict]:
        """
        Prépare les données pour FAISS
        
        Args:
            embeddings_dict: Dictionnaire des embeddings
            
        Returns:
            Tuple (matrice embeddings, données idmap)
        """
        # Trie par ID pour ordre consistant
        sorted_items = sorted(embeddings_dict.items(), key=lambda x: int(x[0]))

        embeddings_list = []
        ids = []
        content = []
        subject = []
        source = []
        ordinal = []

        for i, (chunk_id, data) in enumerate(sorted_items):
            embeddings_list.append(data["embedding"])
            ids.append(int(chunk_id))
            content.append(data.get("content", ""))
            subject.append(data.get("section", ""))  # section devient subject
            source.append(data.get("source", ""))
            ordinal.append(i)  # Ordre naturel

        # Convertit en matrice numpy
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        # Structure idmap compatible avec votre code
        idmap_data = {
            "ids": ids,
            "content": content,
            "subject": subject,
            "source": source,
            "ordinal": ordinal
        }

        return embeddings_matrix, idmap_data

    def save_index(self, index_path: Optional[Path] = None,
                   idmap_path: Optional[Path] = None,
                   metadata_path: Optional[Path] = None) -> None:
        """
        Sauvegarde l'index et les métadonnées
        
        Args:
            index_path: Chemin de l'index FAISS (optionnel)
            idmap_path: Chemin des métadonnées (optionnel)
            metadata_path: Chemin des métadonnées d'index (optionnel)
        """
        if self.index is None:
            raise ValueError("Aucun index à sauvegarder")

        # Utilise les chemins par défaut si non spécifiés
        idx_path = index_path or settings.faiss_index_path
        id_path = idmap_path or settings.idmap_path
        meta_path = metadata_path or settings.metadata_path

        # Sauvegarde l'index FAISS
        faiss.write_index(self.index, str(idx_path))
        print(f"Index FAISS sauvegardé: {idx_path}")

        # Sauvegarde l'idmap
        save_json(self.idmap, id_path)
        print(f"Idmap sauvegardé: {id_path}")

        # Sauvegarde les métadonnées
        save_json(self.metadata, meta_path)
        print(f"Métadonnées sauvegardées: {meta_path}")

    def load_index(self, index_path: Optional[Path] = None,
                   idmap_path: Optional[Path] = None,
                   metadata_path: Optional[Path] = None) -> bool:
        """
        Charge un index existant
        
        Args:
            index_path: Chemin de l'index FAISS
            idmap_path: Chemin de l'idmap
            metadata_path: Chemin des métadonnées
            
        Returns:
            True si chargé avec succès
        """
        # Utilise les chemins par défaut si non spécifiés
        idx_path = index_path or settings.faiss_index_path
        id_path = idmap_path or settings.idmap_path
        meta_path = metadata_path or settings.metadata_path

        try:
            # Vérifie que tous les fichiers existent
            if not all(p.exists() for p in [idx_path, id_path, meta_path]):
                return False

            # Charge l'index FAISS
            self.index = faiss.read_index(str(idx_path))

            # Charge l'idmap
            self.idmap = load_json(id_path)

            # Charge les métadonnées
            self.metadata = load_json(meta_path)

            print(f"Index chargé: {self.metadata['total_vectors']} vecteurs")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement de l'index: {e}")
            return False

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[List[float], List[int]]:
        """
        Recherche dans l'index FAISS
        
        Args:
            query_vector: Vecteur de requête
            top_k: Nombre de résultats
            
        Returns:
            Tuple (scores, ids)
        """
        if self.index is None:
            raise ValueError("Aucun index chargé")

        # S'assure que le vecteur est au bon format
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        # Normalise si nécessaire
        if self.metadata.get("normalized", False):
            faiss.normalize_L2(query)

        # Configure nprobe pour les index IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = settings.get_faiss_config()["nprobe"]

        # Recherche
        scores, ids = self.index.search(query, top_k)

        return scores[0].tolist(), ids[0].tolist()

    def get_runtime_data(self) -> Dict[str, Any]:
        """
        Prépare les données runtime pour le retrieveur
        Compatible avec votre code existant
        
        Returns:
            Dictionnaire runtime avec index, idmap, meta
        """
        if self.index is None or self.idmap is None:
            raise ValueError("Index non chargé")

        return {
            "index": self.index,
            "idmap": self.idmap,
            "meta": self.metadata
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistiques de l'index
        
        Returns:
            Dictionnaire des statistiques
        """
        if self.index is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "total_vectors": self.metadata.get("total_vectors", 0),
            "dimension": self.metadata.get("dimension", 0),
            "index_type": self.metadata.get("index_type", "unknown"),
            "metric": self.metadata.get("metric", "unknown"),
            "normalized": self.metadata.get("normalized", False)
        }
