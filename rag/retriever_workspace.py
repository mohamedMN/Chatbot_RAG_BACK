# rag/retriever_workspace.py
from __future__ import annotations
import numpy as np
import faiss
from typing import Dict, List, Any, Optional
import logging

log = logging.getLogger(__name__)


class WorkspaceRetriever:
    """
    Retriever optimisé pour les workspaces individuels.
    Effectue un scoring proper avec FAISS.
    """

    def __init__(self):
        self.embedding_model = None
        self.runtime_data: Optional[Dict[str, Any]] = None

    def load_runtime(self, runtime_data: Dict[str, Any]) -> bool:
        """
        Charge les données runtime depuis un workspace.
        
        Args:
            runtime_data: Dict avec 'index', 'idmap', 'meta'
        """
        if not runtime_data:
            log.error("Runtime data vide")
            return False

        required = ['index', 'idmap']
        missing = [k for k in required if k not in runtime_data]
        if missing:
            log.error(f"Runtime data incomplet: manque {missing}")
            return False

        self.runtime_data = runtime_data

        # Lazy load embedder
        if not self.embedding_model:
            try:
                from core.embedder_selector import get_embedder
                self.embedding_model = get_embedder()
                log.info("✓ Embedder chargé")
            except Exception as e:
                log.error(f"Échec chargement embedder: {e}")
                return False

        index = runtime_data['index']
        ntotal = getattr(index, 'ntotal', 0)
        log.info(f"✓ Retriever chargé: {ntotal} vecteurs")

        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Récupère les chunks les plus pertinents.
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de résultats max
            min_score: Score minimum (0-1)
        
        Returns:
            Liste de hits triés par score décroissant
        """
        if not self.runtime_data:
            log.error("Runtime data non chargé")
            return []

        try:
            # 1. Embed query
            query_embedding = self._embed_query(query)
            if query_embedding is None:
                return []

            # 2. Search FAISS
            hits = self._search_faiss(query_embedding, top_k * 2)  # Widen

            # 3. Filter by min_score
            filtered = [h for h in hits if h['score'] >= min_score]

            # 4. Re-rank with hybrid scoring
            scored = self._hybrid_scoring(query, filtered)

            # 5. Return top_k
            result = scored[:top_k]

            log.info(
                f"✓ Retrieve: {len(result)}/{len(hits)} hits (min_score={min_score})")
            return result

        except Exception as e:
            log.error(f"Erreur retrieve: {e}", exc_info=True)
            return []

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Génère l'embedding de la query."""
        if not self.embedding_model:
            log.error("Embedding model non initialisé")
            return None

        try:
            embeddings = self.embedding_model.embed([query])
            vec = np.array(embeddings[0], dtype=np.float32)

            # Normalize for cosine similarity (if using IP)
            meta = self.runtime_data.get('meta', {})
            if meta.get('metric', 'ip') == 'ip':
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

            return vec.reshape(1, -1)

        except Exception as e:
            log.error(f"Erreur embedding query: {e}")
            return None

    def _search_faiss(self, query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Recherche dans l'index FAISS."""
        index = self.runtime_data['index']
        idmap = self.runtime_data['idmap']

        # Vérifier dimension
        faiss_dim = getattr(index, 'd', None)
        if faiss_dim and query_vec.shape[1] != faiss_dim:
            log.error(
                f"Dimension mismatch: query={query_vec.shape[1]} vs index={faiss_dim}")
            return []

        # Search
        ntotal = getattr(index, 'ntotal', 0)
        k_actual = min(k, ntotal)

        if k_actual == 0:
            log.warning("Index vide")
            return []

        scores, indices = index.search(query_vec, k_actual)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        # Build hits
        hits = []
        id_to_row = {int(i): r for r, i in enumerate(idmap['ids'])}

        for score, idx in zip(scores, indices):
            if idx == -1:  # FAISS invalid index
                continue

            row = id_to_row.get(int(idx))
            if row is None:
                log.warning(f"Index {idx} non trouvé dans idmap")
                continue

            hits.append({
                'id': int(idx),
                'ordinal': int(idmap['ordinal'][row]),
                'score': float(score),
                'content': str(idmap['content'][row]),
                'subject': str(idmap['subject'][row]),
                'source': str(idmap['source'][row]),
            })

        return hits

    def _hybrid_scoring(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank avec scoring hybride (vector + keywords).
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for hit in hits:
            # Keyword overlap
            text = f"{hit['subject']} {hit['content']}".lower()
            text_words = set(text.split())

            if query_words:
                overlap = len(query_words & text_words) / len(query_words)
            else:
                overlap = 0.0

            # Hybrid score (70% vector, 30% keywords)
            hit['keyword_score'] = overlap
            hit['final_score'] = 0.7 * hit['score'] + 0.3 * overlap

        # Sort by final_score
        hits.sort(key=lambda h: (-h['final_score'], h['ordinal']))

        return hits
