"""
Module de récupération de contexte pour RAG
Compatible avec votre code existant
"""
from typing import Dict, List, Any, Optional
import numpy as np

from config.settings import settings
from core.embedder import EmbeddingModel
from rag.helpers import norm, filter_hits_by_keywords, extract_keywords, deduplicate_hits, calculate_relevance_score, validate_hit
from indexing.faiss_indexer import FAISSIndexer


class RAGRetriever:
    """
    Récupérateur de contexte pour RAG utilisant FAISS
    Compatible avec votre fonction retrieve() existante
    """

    def __init__(self, faiss_indexer: Optional[FAISSIndexer] = None):
        self.faiss_indexer = faiss_indexer or FAISSIndexer()
        self.embedding_model = EmbeddingModel()
        self.runtime_data = None

        # Configuration par défaut
        self.config = settings.get_retrieval_config()

    def load_runtime(self) -> bool:
        """
        Charge les données runtime (index, idmap, métadonnées)
        
        Returns:
            True si chargé avec succès
        """
        try:
            # Charge l'index FAISS s'il n'est pas déjà chargé
            if not self.faiss_indexer.load_index():
                print("Impossible de charger l'index FAISS")
                return False

            # Prépare les données runtime compatibles avec votre code
            self.runtime_data = self.faiss_indexer.get_runtime_data()
            print("Données runtime chargées avec succès")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement des données runtime: {e}")
            return False

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Récupère le contexte pertinent pour une requête
        Interface compatible avec votre code existant
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre maximum de résultats (optionnel)
            min_score: Score minimum (optionnel)
            
        Returns:
            Liste des chunks pertinents avec scores et métadonnées
        """
        if not self.runtime_data:
            if not self.load_runtime():
                return []

        # Utilise les paramètres fournis ou la configuration par défaut
        k = top_k or self.config["top_k"]
        min_score_threshold = min_score or self.config["min_score"]

        return self._retrieve_with_runtime(self.runtime_data, query, k, min_score_threshold)

    def _retrieve_with_runtime(self, runtime: Dict, query: str,
                               top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """
        Version interne qui reproduit exactement votre logique retrieve()
        
        Args:
            runtime: Données runtime (index, idmap, meta)
            query: Requête
            top_k: Nombre de résultats
            min_score: Score minimum
            
        Returns:
            Liste des résultats filtrés et triés
        """
        index, idmap, meta = runtime["index"], runtime["idmap"], runtime.get("meta", {
        })
        use_ip = str(meta.get("metric", "")).lower() == "ip"
        normalize = bool(self.config.get("normalize", True))

        # Génère l'embedding de la requête
        query_embedding = self.embedding_model.embed([query])[0]
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)

        # Normalise si nécessaire pour inner product
        if use_ip and normalize:
            q = norm(q)

        # Recherche vectorielle
        scores, ids = index.search(q, int(top_k))
        scores, ids = scores[0].tolist(), ids[0].tolist()

        # Création du mapping ID -> position
        id_arr = idmap["ids"]
        pos_by_id = {int(fid): int(i) for i, fid in enumerate(id_arr)}

        # Construction des résultats
        hits: List[Dict] = []
        for s, fid in zip(scores, ids):
            if fid == -1 or s < min_score:
                continue

            pos = pos_by_id.get(int(fid))
            if pos is None:
                continue

            try:
                ordinal_val = int(idmap["ordinal"][pos])
            except Exception:
                ordinal_val = int(pos)

            hit = {
                "score": float(s),
                "id": int(fid),
                "ordinal": ordinal_val,
                "content": str(idmap["content"][pos]),
                "subject": str(idmap["subject"][pos]),
                "source": str(idmap["source"][pos]),
            }

            # Valide le résultat
            if validate_hit(hit):
                hits.append(hit)

        # Tri stable : meilleurs scores, puis ordre naturel
        hits.sort(key=lambda h: (-h["score"], h["ordinal"]))

        # Filtrage lexical avec vos helpers
        kws = extract_keywords(query)
        hits = filter_hits_by_keywords(hits, kws)

        # Durcissement du seuil pour questions courtes/ambiguës (votre logique)
        thr = float(self.config.get("min_score", 0.30))
        if len(kws) <= 1 and hits and thr < 0.45:
            hits = [h for h in hits if h.get("score", 0.0) >= 0.45]

        # Post-traitement supplémentaire
        hits = self._post_process_hits(hits, kws)

        return hits

    def _post_process_hits(self, hits: List[Dict], keywords: List[str]) -> List[Dict]:
        """
        Post-traitement des résultats (dédoublonnage, scoring)
        
        Args:
            hits: Résultats bruts
            keywords: Mots-clés extraits
            
        Returns:
            Résultats post-traités
        """
        if not hits:
            return hits

        # Calcule un score de pertinence amélioré
        for hit in hits:
            hit["relevance_score"] = calculate_relevance_score(hit, keywords)

        # Dédoublonne les résultats similaires
        hits = deduplicate_hits(hits, similarity_threshold=0.85)

        # Tri final par score de pertinence puis score vectoriel
        hits.sort(key=lambda h: (-h.get("relevance_score", 0), -h.get("score", 0)))

        return hits

    def search_by_keywords(self, keywords: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche basée uniquement sur des mots-clés (fallback)
        
        Args:
            keywords: Liste de mots-clés
            top_k: Nombre de résultats
            
        Returns:
            Résultats basés sur correspondance lexicale
        """
        if not self.runtime_data or not keywords:
            return []

        idmap = self.runtime_data["idmap"]
        hits = []

        # Recherche lexicale simple dans le contenu
        for i, content in enumerate(idmap["content"]):
            content_lower = content.lower()
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)

            if matches > 0:
                hit = {
                    # Score basé sur le pourcentage de correspondance
                    "score": matches / len(keywords),
                    "id": idmap["ids"][i],
                    "ordinal": idmap["ordinal"][i],
                    "content": content,
                    "subject": idmap["subject"][i],
                    "source": idmap["source"][i],
                    "keyword_matches": matches
                }
                hits.append(hit)

        # Trie par nombre de correspondances
        hits.sort(key=lambda h: -h["keyword_matches"])

        return hits[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistiques du retrieveur
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            "runtime_loaded": self.runtime_data is not None,
            "config": self.config.copy()
        }

        if self.faiss_indexer:
            stats.update(self.faiss_indexer.get_stats())

        return stats

    def update_config(self, **kwargs) -> None:
        """
        Met à jour la configuration du retrieveur
        
        Args:
            **kwargs: Nouveaux paramètres de configuration
        """
        self.config.update(kwargs)
        print(f"Configuration mise à jour: {kwargs}")

# Fonction utilitaire compatible avec votre code existant


def retrieve(runtime: Dict, query: str, top_k: int, min_score: float) -> List[Dict]:
    """
    Fonction de récupération compatible avec votre code existant
    
    Args:
        runtime: Données runtime (index, idmap, meta)
        query: Requête utilisateur
        top_k: Nombre de résultats
        min_score: Score minimum
        
    Returns:
        Liste des résultats
    """
    retriever = RAGRetriever()
    retriever.runtime_data = runtime
    return retriever._retrieve_with_runtime(runtime, query, top_k, min_score)
