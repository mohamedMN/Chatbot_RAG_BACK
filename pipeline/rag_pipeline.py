"""
Pipeline RAG complet avec indexing FAISS
"""
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import settings
from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from core.embedder import ChunkEmbedder
from indexing.faiss_indexer import FAISSIndexer
from rag.rag_engine import RAGEngine
from utils.file_utils import save_json


class RAGPipelineWithFAISS:
    """Pipeline RAG complet incluant la création d'index FAISS"""

    def __init__(self):
        self.document_loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.embedder = ChunkEmbedder()
        self.faiss_indexer = FAISSIndexer()

        # Statistiques du pipeline
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'faiss_index_created': False,
            'processing_time': 0,
            'errors': []
        }

    def run_full_pipeline(self, documents_path: Optional[Path] = None,
                          output_path: Optional[Path] = None,
                          rebuild_index: bool = False) -> Dict[str, Any]:
        """
        Exécute le pipeline complet incluant la création d'index FAISS
        
        Args:
            documents_path: Dossier des documents (optionnel)
            output_path: Dossier de sortie (optionnel)
            rebuild_index: Forcer la reconstruction de l'index
            
        Returns:
            Résultats et statistiques du pipeline
        """
        start_time = time.time()

        # Utilise les chemins par défaut
        docs_path = documents_path or settings.documents_path
        out_path = output_path or settings.output_path

        print("=" * 60)
        print("DÉMARRAGE DU PIPELINE RAG COMPLET AVEC FAISS")
        print("=" * 60)
        print(f"Documents: {docs_path}")
        print(f"Sortie: {out_path}")
        print(f"Index FAISS: {settings.index_path}")
        print()

        try:
            # Vérifie si l'index existe déjà
            if not rebuild_index and self._index_exists():
                print(
                    "Index FAISS existant trouvé. Utilisez rebuild_index=True pour le reconstruire.")
                return self._create_success_result(skip_build=True)

            # Étape 1: Chargement des documents
            print("ÉTAPE 1: Chargement des documents...")
            documents = self._load_documents(docs_path)

            if not documents:
                return self._create_error_result("Aucun document trouvé")

            # Étape 2: Chunking des documents
            print("\nÉTAPE 2: Découpage en chunks...")
            chunks = self._process_chunks(documents)

            if not chunks:
                return self._create_error_result("Aucun chunk généré")

            # Étape 3: Génération des embeddings
            print("\nÉTAPE 3: Génération des embeddings...")
            embeddings = self._generate_embeddings(chunks)

            # Étape 4: Création de l'index FAISS
            print("\nÉTAPE 4: Création de l'index FAISS...")
            self._create_faiss_index(embeddings)

            # Étape 5: Sauvegarde
            print("\nÉTAPE 5: Sauvegarde des résultats...")
            self._save_results(chunks, embeddings, out_path)

            # Calcul du temps total
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time

            # Affichage du résumé
            self._print_summary()

            return self._create_success_result()

        except Exception as e:
            error_msg = f"Erreur dans le pipeline: {str(e)}"
            print(f"\nERREUR: {error_msg}")
            self.stats['errors'].append(error_msg)
            return self._create_error_result(error_msg)

    def _index_exists(self) -> bool:
        """Vérifie si l'index FAISS existe déjà"""
        return (settings.faiss_index_path.exists() and
                settings.idmap_path.exists() and
                settings.metadata_path.exists())

    def _load_documents(self, documents_path: Path) -> List[Dict[str, Any]]:
        """Charge les documents"""
        documents = self.document_loader.load_documents(documents_path)
        self.stats['documents_processed'] = len(documents)

        print(f"  {len(documents)} document(s) chargé(s)")

        for doc in documents:
            doc_name = Path(doc['source']).name
            content_length = len(doc['content'])
            sections_count = len(doc.get('predefined_sections', []))
            print(
                f"    {doc_name}: {content_length} caractères, {sections_count} sections")

        return documents

    def _process_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Traite les documents en chunks"""
        all_chunks = []

        for i, document in enumerate(documents, 1):
            print(
                f"    Document {i}/{len(documents)}: {Path(document['source']).name}")

            document_chunks = self.chunker.chunk_document(document)
            all_chunks.extend(document_chunks)

            print(f"      {len(document_chunks)} chunks générés")

        # Statistiques
        stats = self.chunker.get_chunking_stats(all_chunks)
        self.stats['chunks_created'] = stats['total_chunks']

        print(f"\n  Statistiques de chunking:")
        print(f"    Total chunks: {stats['total_chunks']}")
        print(f"    Tokens moyens: {stats['avg_tokens']:.1f}")
        print(f"    Plage: {stats['min_tokens']}-{stats['max_tokens']} tokens")

        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Génère les embeddings"""
        embeddings = self.embedder.create_embeddings(chunks)
        self.stats['embeddings_generated'] = len(embeddings)

        # Validation
        validation = self.embedder.validate_embeddings(embeddings)

        print(f"\n  Statistiques des embeddings:")
        print(f"    Total: {len(embeddings)}")
        print(f"    Valides: {validation['valid_count']}")
        print(f"    Taux de succès: {validation['success_rate']:.1%}")

        if validation['issues']:
            print(f"    Problèmes détectés: {len(validation['issues'])}")

        return embeddings

    def _create_faiss_index(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> None:
        """Crée l'index FAISS"""
        faiss_config = settings.get_faiss_config()
        index_type = faiss_config["index_type"]

        print(f"  Création de l'index {index_type}...")

        # Crée l'index
        self.faiss_indexer.create_index(embeddings_dict, index_type)

        # Sauvegarde l'index
        self.faiss_indexer.save_index()

        self.stats['faiss_index_created'] = True

        # Statistiques de l'index
        index_stats = self.faiss_indexer.get_stats()
        print(f"  Index créé: {index_stats['total_vectors']} vecteurs")
        print(f"  Type: {index_stats['index_type']}")
        print(f"  Métrique: {index_stats['metric']}")

    def _save_results(self, chunks: List[Dict[str, Any]],
                      embeddings: Dict[str, Dict[str, Any]],
                      output_path: Path) -> None:
        """Sauvegarde les résultats (pour compatibilité)"""
        # Sauvegarde les chunks
        chunks_path = output_path / 'chunks.json'
        save_json(chunks, chunks_path)
        print(f"    Chunks sauvegardés: {chunks_path}")

        # Sauvegarde les embeddings (pour compatibilité ascendante)
        embeddings_path = output_path / 'embeddings.json'
        save_json(embeddings, embeddings_path)
        print(f"    Embeddings sauvegardés: {embeddings_path}")

        # Sauvegarde les statistiques
        stats_path = output_path / 'pipeline_stats.json'
        save_json(self.stats, stats_path)
        print(f"    Statistiques sauvegardées: {stats_path}")

    def _print_summary(self) -> None:
        """Affiche le résumé final"""
        print("\n" + "=" * 60)
        print("PIPELINE RAG TERMINÉ AVEC SUCCÈS")
        print("=" * 60)

        print(f"RÉSUMÉ:")
        print(f"  Documents traités: {self.stats['documents_processed']}")
        print(f"  Chunks créés: {self.stats['chunks_created']}")
        print(f"  Embeddings générés: {self.stats['embeddings_generated']}")
        print(
            f"  Index FAISS: {'Créé' if self.stats['faiss_index_created'] else 'Échec'}")
        print(f"  Temps d'exécution: {self.stats['processing_time']:.2f}s")

        if self.stats['errors']:
            print(f"  Erreurs: {len(self.stats['errors'])}")
        else:
            print("  Aucune erreur")

        print("\nVotre système RAG est prêt !")
        print(f"Index FAISS: {settings.faiss_index_path}")
        print("=" * 60)

    def _create_success_result(self, skip_build: bool = False) -> Dict[str, Any]:
        """Crée un résultat de succès"""
        return {
            'success': True,
            'skip_build': skip_build,
            'faiss_index_path': str(settings.faiss_index_path),
            'idmap_path': str(settings.idmap_path),
            'metadata_path': str(settings.metadata_path),
            'chunks_path': str(settings.chunks_path),
            'embeddings_path': str(settings.embeddings_path),
            'stats': self.stats
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Crée un résultat d'erreur"""
        return {
            'success': False,
            'error': error_message,
            'stats': self.stats
        }

    def test_rag_system(self, test_question: str = "Qu'est-ce que webMethods?") -> Dict[str, Any]:
        """
        Test rapide du système RAG après construction
        
        Args:
            test_question: Question de test
            
        Returns:
            Résultats du test
        """
        print(f"\nTest du système RAG avec la question: '{test_question}'")

        try:
            # Crée un engine RAG
            rag_engine = RAGEngine(faiss_indexer=self.faiss_indexer)

            if not rag_engine.initialize():
                return {"success": False, "error": "Échec d'initialisation de l'engine RAG"}

            # Test de récupération seulement
            retrieval_result = rag_engine.retrieve_only(test_question, top_k=3)

            print(f"Résultats de récupération:")
            print(f"  Chunks trouvés: {retrieval_result['context_count']}")
            print(
                f"  Temps de récupération: {retrieval_result['retrieval_time']}s")

            if retrieval_result['context_count'] > 0:
                print("  Premier résultat:")
                first_hit = retrieval_result['context_hits'][0]
                print(f"    Score: {first_hit['score']:.3f}")
                print(f"    Contenu: {first_hit['content'][:100]}...")

            return retrieval_result

        except Exception as e:
            return {"success": False, "error": str(e)}

# Fonction utilitaire pour l'exécution rapide


def run_rag_pipeline_with_faiss(documents_path: Optional[Path] = None,
                                output_path: Optional[Path] = None,
                                rebuild_index: bool = False) -> Dict[str, Any]:
    """
    Fonction utilitaire pour exécuter le pipeline RAG complet avec FAISS
    
    Args:
        documents_path: Dossier des documents
        output_path: Dossier de sortie
        rebuild_index: Forcer la reconstruction de l'index
        
    Returns:
        Résultats du pipeline
    """
    pipeline = RAGPipelineWithFAISS()
    return pipeline.run_full_pipeline(documents_path, output_path, rebuild_index)
