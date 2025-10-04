"""
Module de génération d'embeddings pour les chunks
"""
import random
import math
from typing import List, Dict, Any
from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION

# Import conditionnel de sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Attention: sentence-transformers non installé. Utilisation d'embeddings factices.")
    print("Pour installer: pip install sentence-transformers")


class EmbeddingModel:
    """Modèle d'embeddings pour les chunks de texte"""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.embedding_dim = EMBEDDING_DIMENSION

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                # Récupère la dimension réelle du modèle
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(
                    f"Modèle d'embedding chargé: {model_name} (dimension: {self.embedding_dim})")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")
                self.model = None
        else:
            self.model = None
            print(
                f"Utilisation d'embeddings factices (dimension: {self.embedding_dim})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour une liste de textes
        
        Args:
            texts: Liste des textes à encoder
            
        Returns:
            Liste des embeddings (vecteurs de flottants)
        """
        if not texts:
            return []

        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            return self._embed_real(texts)
        else:
            return self._embed_dummy(texts)

    def _embed_real(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings réels avec sentence-transformers
        
        Args:
            texts: Liste des textes
            
        Returns:
            Embeddings réels
        """
        try:
            embeddings = self.model.encode(
                texts, convert_to_tensor=False, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Erreur lors de la génération d'embeddings: {e}")
            return self._embed_dummy(texts)

    def _embed_dummy(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings factices pour les tests
        
        Args:
            texts: Liste des textes
            
        Returns:
            Embeddings factices normalisés
        """
        embeddings = []

        for text in texts:
            # Utilise le hash du texte comme seed pour la reproductibilité
            random.seed(hash(text) % (2**32))

            # Génère un vecteur aléatoire
            embedding = [random.uniform(-1, 1)
                         for _ in range(self.embedding_dim)]

            # Normalise le vecteur
            magnitude = math.sqrt(sum(x**2 for x in embedding))
            if magnitude > 0:
                normalized = [x / magnitude for x in embedding]
            else:
                normalized = [0.0] * self.embedding_dim

            embeddings.append(normalized)

        return embeddings

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle d'embedding
        
        Returns:
            Informations sur le modèle
        """
        return {
            'model_name': self.model_name,
            'dimension': self.embedding_dim,
            'is_real': SENTENCE_TRANSFORMERS_AVAILABLE and self.model is not None,
            'available': SENTENCE_TRANSFORMERS_AVAILABLE
        }


class ChunkEmbedder:
    """Classe pour générer des embeddings de chunks en lots"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.embedding_model = EmbeddingModel(model_name)
        self.batch_size = batch_size

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Crée des embeddings pour une liste de chunks
        
        Args:
            chunks: Liste des chunks à encoder
            
        Returns:
            Dictionnaire des embeddings avec métadonnées
        """
        if not chunks:
            return {}

        print(f"Génération d'embeddings pour {len(chunks)} chunks...")

        result = {}
        total_processed = 0

        # Traite par lots
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [chunk.get('content', '') for chunk in batch]

            # Filtre les textes vides
            non_empty_texts = [(j, text)
                               for j, text in enumerate(texts) if text.strip()]

            if not non_empty_texts:
                continue

            # Génère les embeddings pour les textes non vides
            indices, filtered_texts = zip(*non_empty_texts)
            embeddings = self.embedding_model.embed(list(filtered_texts))

            # Ajoute aux résultats
            for idx, embedding in zip(indices, embeddings):
                chunk = batch[idx]
                chunk_id = str(
                    chunk.get('id', f'chunk_{total_processed + idx}'))

                result[chunk_id] = {
                    'embedding': embedding,
                    'content': chunk.get('content', ''),
                    'source': chunk.get('source', ''),
                    'section': chunk.get('section', 'Section Principale')
                }

            total_processed += len(batch)

            # Affiche le progrès tous les 5 lots
            if total_processed % (self.batch_size * 5) == 0:
                print(f"Traité {total_processed}/{len(chunks)} chunks...")

        print(f"Embeddings générés avec succès pour {len(result)} chunks")
        return result

    def get_embedding_stats(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcule les statistiques des embeddings
        
        Args:
            embeddings_dict: Dictionnaire des embeddings
            
        Returns:
            Statistiques des embeddings
        """
        if not embeddings_dict:
            return {
                'total_embeddings': 0,
                'embedding_dimension': 0,
                'model_info': self.embedding_model.get_embedding_info()
            }

        # Prend le premier embedding pour vérifier la dimension
        first_embedding = next(iter(embeddings_dict.values()))['embedding']

        return {
            'total_embeddings': len(embeddings_dict),
            'embedding_dimension': len(first_embedding),
            'model_info': self.embedding_model.get_embedding_info(),
            'sample_embedding_norm': math.sqrt(sum(x**2 for x in first_embedding))
        }

    def validate_embeddings(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valide la qualité des embeddings générés
        
        Args:
            embeddings_dict: Dictionnaire des embeddings
            
        Returns:
            Rapport de validation
        """
        if not embeddings_dict:
            return {
                'valid': False,
                'error': 'Aucun embedding trouvé'
            }

        issues = []
        valid_count = 0
        expected_dim = self.embedding_model.embedding_dim

        for chunk_id, data in embeddings_dict.items():
            embedding = data.get('embedding', [])

            # Vérifie la dimension
            if len(embedding) != expected_dim:
                issues.append(
                    f"Chunk {chunk_id}: dimension incorrecte ({len(embedding)} != {expected_dim})")
                continue

            # Vérifie si l'embedding contient des valeurs valides
            if not embedding or all(x == 0 for x in embedding):
                issues.append(f"Chunk {chunk_id}: embedding vide ou nul")
                continue

            # Vérifie la norme (devrait être proche de 1 pour les embeddings normalisés)
            norm = math.sqrt(sum(x**2 for x in embedding))
            if norm < 0.5 or norm > 2.0:  # Plage de tolérance
                issues.append(f"Chunk {chunk_id}: norme suspecte ({norm:.3f})")

            valid_count += 1

        return {
            'valid': len(issues) == 0,
            'valid_count': valid_count,
            'total_count': len(embeddings_dict),
            'issues': issues[:10],  # Limite à 10 problèmes pour éviter le spam
            'success_rate': valid_count / len(embeddings_dict) if embeddings_dict else 0
        }
