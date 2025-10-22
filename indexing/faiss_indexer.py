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

    def create_index(self, embeddings_dict: Dict[str, Dict[str, Any]], index_type: str = "IndexFlatIP") -> None:
        if not embeddings_dict:
            raise ValueError("Aucun embedding fourni pour créer l'index")

        print(f"Création de l'index FAISS {index_type}...")

        X, idmap_data = self._prepare_data(embeddings_dict)
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension embedding inattendue: got {X.shape}, expected (*,{self.dimension})")

        n = X.shape[0]
        faiss_cfg = settings.get_faiss_config()
        use_ip = ("IP" in index_type)
        do_norm = bool(faiss_cfg.get("normalize", True) and use_ip)

        # Normalize BEFORE training and adding if we want cosine with IP
        if do_norm:
            faiss.normalize_L2(X)

        # Build base index (with safety for small datasets if IVF requested)
        if index_type == "IndexFlatIP":
            base = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexFlatL2":
            base = faiss.IndexFlatL2(self.dimension)
        elif index_type.startswith("IndexIVF"):
            # rough rule: need enough vectors to train
            nlist = min(max(1, faiss_cfg.get("nlist", 100)), max(1, n // 10))
            quantizer = faiss.IndexFlatIP(
                self.dimension) if use_ip else faiss.IndexFlatL2(self.dimension)
            base = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            if n < max(1000, nlist * 5):
                # not enough to train IVF reliably → fallback to flat
                print("Dataset trop petit pour IVF ⇒ fallback vers IndexFlat")
                base = faiss.IndexFlatIP(
                    self.dimension) if use_ip else faiss.IndexFlatL2(self.dimension)
            else:
                base.train(X)
        else:
            raise ValueError(f"Type d'index non supporté: {index_type}")

        # Wrap with IDMap to preserve your external IDs
        self.index = faiss.IndexIDMap2(base)
        ids_np = np.asarray(idmap_data["ids"], dtype=np.int64)
        self.index.add_with_ids(X, ids_np)

        # Sidecar data
        self.idmap = idmap_data
        self.metadata = {
            "index_type": index_type,
            "dimension": self.dimension,
            "total_vectors": int(n),
            "metric": "ip" if use_ip else "l2",
            "normalized": bool(do_norm),
            # optional for visibility:
            "normalize": bool(faiss_cfg.get("normalize", True)),
        }

        print(f"Index créé avec {n} vecteurs")

    def _prepare_data(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Dict]:
        sorted_items = sorted(embeddings_dict.items(), key=lambda x: int(x[0]))

        embeddings_list, ids = [], []
        content, subject, source, ordinal = [], [], [], []

        for i, (chunk_id, data) in enumerate(sorted_items):
            # If your embedder uses "vector", change to data["vector"]
            vec = data["embedding"]
            embeddings_list.append(vec)
            ids.append(int(chunk_id))
            content.append(data.get("content", ""))
            subject.append(data.get("section", ""))  # section -> subject
            source.append(data.get("source", ""))
            ordinal.append(i)

        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        idmap_data = {
            "ids": ids,
            "content": content,
            "subject": subject,
            "source": source,
            "ordinal": ordinal,
        }
        return embeddings_matrix, idmap_data


    def save_index(self) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index not built")
        if self.idmap is None:
            raise RuntimeError("idmap is not set")
        if self.metadata is None:
            raise RuntimeError("metadata is not set")

        idx_path = Path(settings.faiss_index_path)
        idmap_path = Path(settings.idmap_path)
        meta_path = Path(settings.metadata_path)

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idmap_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_path))
        save_json(self.idmap, idmap_path)
        save_json(self.metadata, meta_path)
        
    def load_index(self, index_path: Optional[Path] = None,
                    idmap_path: Optional[Path] = None,
                    metadata_path: Optional[Path] = None) -> bool:
        idx_path = Path(index_path or settings.faiss_index_path)
        id_path = Path(idmap_path or settings.idmap_path)
        meta_path = Path(metadata_path or settings.metadata_path)

        try:
            if not (idx_path.exists() and id_path.exists() and meta_path.exists()):
                return False

            self.index = faiss.read_index(str(idx_path))
            self.idmap = load_json(id_path)
            self.metadata = load_json(meta_path)

            # Minimal validation
            if not isinstance(self.idmap, dict) or "ids" not in self.idmap:
                raise RuntimeError("Corrupt idmap")
            if not isinstance(self.metadata, dict) or "total_vectors" not in self.metadata:
                raise RuntimeError("Corrupt metadata")

            print(
                f"Index chargé: {self.metadata.get('total_vectors', '?')} vecteurs")
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
