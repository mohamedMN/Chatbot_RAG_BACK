"""
Serveur RAG pour traiter les requêtes en temps réel
Compatible avec votre code existant
"""
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path

from config.settings import settings
from rag.rag_engine import RAGEngine
from indexing.faiss_indexer import FAISSIndexer


class RAGServer:
    """Serveur RAG pour traitement de requêtes"""

    def __init__(self, llm=None):
        self.llm = llm
        self.rag_engine = None
        self.initialized = False

    def initialize(self, faiss_indexer: Optional[FAISSIndexer] = None) -> bool:
        """
        Initialise le serveur RAG
        
        Args:
            faiss_indexer: Indexer FAISS pré-configuré (optionnel)
            
        Returns:
            True si initialisé avec succès
        """
        print("Initialisation du serveur RAG...")

        try:
            # Crée l'engine RAG
            self.rag_engine = RAGEngine(self.llm, faiss_indexer)

            # Initialise l'engine
            if not self.rag_engine.initialize():
                print("Erreur: Impossible d'initialiser l'engine RAG")
                return False

            self.initialized = True
            print("Serveur RAG initialisé avec succès")

            # Affiche les statistiques
            stats = self.rag_engine.get_stats()
            print(
                f"Index chargé: {stats['retriever']['total_vectors']} vecteurs")
            print(f"LLM disponible: {stats['llm_available']}")

            return True

        except Exception as e:
            print(f"Erreur lors de l'initialisation: {e}")
            return False

    def process_query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Traite une requête utilisateur
        
        Args:
            question: Question utilisateur
            **kwargs: Paramètres additionnels
            
        Returns:
            Réponse structurée
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Serveur non initialisé"
            }

        return self.rag_engine.query(question, **kwargs)

    def process_retrieval_only(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Traite une requête de récupération seulement
        
        Args:
            question: Question utilisateur
            **kwargs: Paramètres de récupération
            
        Returns:
            Résultats de récupération
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Serveur non initialisé"
            }

        return self.rag_engine.retrieve_only(question, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé
        
        Returns:
            État du serveur
        """
        if not self.initialized or not self.rag_engine:
            return {
                "status": "error",
                "message": "Serveur non initialisé"
            }

        return self.rag_engine.health_check()

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistiques du serveur
        
        Returns:
            Statistiques détaillées
        """
        if not self.initialized:
            return {"initialized": False}

        return self.rag_engine.get_stats()

    def interactive_mode(self) -> None:
        """Mode interactif pour tests"""
        if not self.initialized:
            print("Serveur non initialisé")
            return

        print("\n" + "="*50)
        print("MODE INTERACTIF RAG")
        print("="*50)
        print("Tapez vos questions (ou 'quit' pour quitter)")
        print("Commandes spéciales:")
        print("  /stats - Affiche les statistiques")
        print("  /health - Vérification de santé")
        print("  /retrieval <question> - Récupération seulement")
        print()

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Au revoir!")
                    break

                elif question == '/stats':
                    stats = self.get_stats()
                    print("\nStatistiques:")
                    print(json.dumps(stats, indent=2, ensure_ascii=False))

                elif question == '/health':
                    health = self.health_check()
                    print(f"\nÉtat de santé: {health['overall']}")
                    if health['issues']:
                        for issue in health['issues']:
                            print(f"  - {issue}")

                elif question.startswith('/retrieval '):
                    query = question[11:].strip()
                    if query:
                        result = self.process_retrieval_only(query, top_k=3)
                        print(f"\nRécupération pour: {query}")
                        print(
                            f"Chunks trouvés: {result.get('context_count', 0)}")
                        for i, hit in enumerate(result.get('context_hits', []), 1):
                            print(f"\n  [{i}] Score: {hit['score']:.3f}")
                            print(f"      Source: {Path(hit['source']).name}")
                            print(f"      Contenu: {hit['content'][:150]}...")

                elif question:
                    print("\nTraitement en cours...")
                    result = self.process_query(question, return_context=True)

                    if result['success']:
                        print(
                            f"\nRéponse ({result['timing']['total_time']:.2f}s):")
                        print("-" * 40)
                        print(result['answer'])

                        if result.get('context_count', 0) > 0:
                            print(
                                f"\n(Basé sur {result['context_count']} chunk(s) de contexte)")
                    else:
                        print(f"\nErreur: {result.get('error')}")

            except KeyboardInterrupt:
                print("\n\nInterruption - Au revoir!")
                break
            except Exception as e:
                print(f"Erreur: {e}")


def setup_llm():
    """
    Configure le LLM (à adapter selon vos besoins)
    Vous pouvez ici intégrer votre LLM existant
    """
    # Exemple avec un mock LLM pour tests
    # À remplacer par votre configuration LLM réelle
    print("Configuration du LLM...")

    try:
        # Exemple d'intégration possible (à adapter)
        # from langchain_openai import ChatOpenAI
        # llm = ChatOpenAI(model="gpt-3.5-turbo")
        # return llm

        # Pour les tests sans LLM
        print("Attention: Aucun LLM configuré - mode récupération seulement")
        return None

    except Exception as e:
        print(f"Erreur de configuration LLM: {e}")
        return None


def main():
    """Fonction principale du serveur RAG"""
    parser = argparse.ArgumentParser(
        description="Serveur RAG pour traitement de requêtes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python rag_server.py                    # Mode interactif
  python rag_server.py --test "question"  # Test d'une question
  python rag_server.py --stats            # Affiche les statistiques
        """
    )

    parser.add_argument(
        '--test', '-t',
        type=str,
        help='Teste une question spécifique'
    )

    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Affiche les statistiques et quitte'
    )

    parser.add_argument(
        '--health',
        action='store_true',
        help='Vérification de santé et quitte'
    )

    parser.add_argument(
        '--retrieval-only',
        action='store_true',
        help='Mode récupération seulement (sans génération)'
    )

    args = parser.parse_args()

    # Configuration du LLM
    llm = setup_llm()

    # Initialisation du serveur
    server = RAGServer(llm)

    if not server.initialize():
        print("Échec de l'initialisation du serveur")
        return 1

    # Mode statistiques
    if args.stats:
        stats = server.get_stats()
        print("\nStatistiques du serveur RAG:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    # Mode vérification de santé
    if args.health:
        health = server.health_check()
        print(f"\nÉtat de santé: {health['overall']}")
        for component, status in health['components'].items():
            print(f"  {component}: {status}")
        if health['issues']:
            print("\nProblèmes détectés:")
            for issue in health['issues']:
                print(f"  - {issue}")
        return 0

    # Mode test d'une question
    if args.test:
        question = args.test
        print(f"Test de la question: {question}")

        if args.retrieval_only:
            result = server.process_retrieval_only(question, top_k=5)
            print(f"\nRécupération:")
            print(f"  Chunks trouvés: {result.get('context_count', 0)}")
            print(f"  Temps: {result.get('retrieval_time', 0):.3f}s")

            for i, hit in enumerate(result.get('context_hits', []), 1):
                print(f"\n  [{i}] Score: {hit['score']:.3f}")
                print(f"      Source: {Path(hit['source']).name}")
                print(f"      Section: {hit.get('subject', 'N/A')}")
                print(f"      Contenu: {hit['content'][:200]}...")
        else:
            result = server.process_query(question, return_context=True)

            if result['success']:
                print(
                    f"\nRéponse générée en {result['timing']['total_time']:.2f}s:")
                print("-" * 50)
                print(result['answer'])
                print("-" * 50)
                print(f"Contexte utilisé: {result['context_count']} chunk(s)")
            else:
                print(f"Erreur: {result.get('error')}")

        return 0

    # Mode interactif par défaut
    server.interactive_mode()
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
