"""
Engine RAG complet orchestrant la récupération et la génération
"""
from typing import Dict, List, Any, Optional, Tuple
import time
from pathlib import Path

from config.settings import settings
from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator
from rag.helpers import format_context_for_llm
from indexing.faiss_indexer import FAISSIndexer


class RAGEngine:
    """
    Engine RAG complet combinant récupération vectorielle et génération de réponses
    """

    def __init__(self, llm=None, faiss_indexer: Optional[FAISSIndexer] = None):
        self.llm = llm
        self.retriever = RAGRetriever(faiss_indexer)
        self.generator = RAGGenerator(llm)

        # Statistiques de session
        self.session_stats = {
            "queries_processed": 0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_total_time": 0.0
        }

    def initialize(self) -> bool:
        """
        Initialise l'engine RAG (charge l'index, vérifie les composants)
        
        Returns:
            True si initialisé avec succès
        """
        print("Initialisation de l'engine RAG...")

        # Charge les données runtime
        if not self.retriever.load_runtime():
            print("Erreur: Impossible de charger l'index FAISS")
            return False

        # Vérifie le LLM
        if not self.llm:
            print("Attention: Aucun LLM configuré - mode récupération seulement")

        print("Engine RAG initialisé avec succès")
        return True

    def query(self, question: str, top_k: Optional[int] = None,
              min_score: Optional[float] = None,
              return_context: bool = False) -> Dict[str, Any]:
        """
        Traite une requête complète (récupération + génération)
        
        Args:
            question: Question de l'utilisateur
            top_k: Nombre de chunks à récupérer (optionnel)
            min_score: Score minimum pour la récupération (optionnel)
            return_context: Inclure les chunks récupérés dans la réponse
            
        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        start_time = time.time()

        try:
            # Phase 1: Récupération
            retrieval_start = time.time()
            context_hits = self.retriever.retrieve(question, top_k, min_score)
            retrieval_time = time.time() - retrieval_start

            # Phase 2: Génération
            generation_start = time.time()
            answer = self.generator.generate_answer(question, context_hits)
            generation_time = time.time() - generation_start

            total_time = time.time() - start_time

            # Mise à jour des statistiques
            self._update_stats(retrieval_time, generation_time, total_time)

            # Préparation de la réponse
            result = {
                "question": question,
                "answer": answer,
                "context_found": len(context_hits) > 0,
                "context_count": len(context_hits),
                "timing": {
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3)
                },
                "success": True
            }

            # Ajoute le contexte si demandé
            if return_context:
                result["context_hits"] = context_hits
                result["formatted_context"] = format_context_for_llm(
                    context_hits)

            return result

        except Exception as e:
            return {
                "question": question,
                "answer": f"Erreur lors du traitement de la requête: {str(e)}",
                "success": False,
                "error": str(e)
            }

    def retrieve_only(self, question: str, top_k: Optional[int] = None,
                      min_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Récupération seulement (sans génération)
        
        Args:
            question: Question de l'utilisateur
            top_k: Nombre de résultats
            min_score: Score minimum
            
        Returns:
            Résultats de récupération
        """
        try:
            start_time = time.time()
            context_hits = self.retriever.retrieve(question, top_k, min_score)
            retrieval_time = time.time() - start_time

            return {
                "question": question,
                "context_hits": context_hits,
                "context_count": len(context_hits),
                "formatted_context": format_context_for_llm(context_hits),
                "retrieval_time": round(retrieval_time, 3),
                "success": True
            }

        except Exception as e:
            return {
                "question": question,
                "success": False,
                "error": str(e)
            }

    def generate_only(self, question: str, context_hits: List[Dict]) -> Dict[str, Any]:
        """
        Génération seulement (avec contexte fourni)
        
        Args:
            question: Question utilisateur
            context_hits: Contexte à utiliser
            
        Returns:
            Réponse générée
        """
        try:
            start_time = time.time()
            answer = self.generator.generate_answer(question, context_hits)
            generation_time = time.time() - start_time

            return {
                "question": question,
                "answer": answer,
                "context_count": len(context_hits),
                "generation_time": round(generation_time, 3),
                "success": True
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"Erreur de génération: {str(e)}",
                "success": False,
                "error": str(e)
            }

    def batch_query(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Traite plusieurs questions en lot
        
        Args:
            questions: Liste des questions
            **kwargs: Paramètres pour chaque query
            
        Returns:
            Liste des réponses
        """
        results = []
        print(f"Traitement de {len(questions)} questions en lot...")

        for i, question in enumerate(questions, 1):
            print(f"  Question {i}/{len(questions)}")
            result = self.query(question, **kwargs)
            results.append(result)

        return results

    def stream_query(self, question: str, top_k: Optional[int] = None,
                     min_score: Optional[float] = None):
        """
        Traite une requête avec réponse en streaming
        
        Args:
            question: Question utilisateur
            top_k: Nombre de chunks
            min_score: Score minimum
            
        Yields:
            Chunks de la réponse
        """
        try:
            # Récupération
            context_hits = self.retriever.retrieve(question, top_k, min_score)

            # Génération en streaming
            yield from self.generator.generate_streaming_answer(question, context_hits)

        except Exception as e:
            yield f"Erreur de streaming: {str(e)}"

    def evaluate_query(self, question: str, expected_answer: str = None,
                       **query_kwargs) -> Dict[str, Any]:
        """
        Évalue la qualité d'une requête (pour tests et optimisation)
        
        Args:
            question: Question à évaluer
            expected_answer: Réponse attendue (optionnel)
            **query_kwargs: Paramètres de requête
            
        Returns:
            Métriques d'évaluation
        """
        result = self.query(question, return_context=True, **query_kwargs)

        if not result["success"]:
            return {"success": False, "error": result.get("error")}

        # Métriques de base
        metrics = {
            "question": question,
            "context_found": result["context_found"],
            "context_count": result["context_count"],
            "retrieval_time": result["timing"]["retrieval_time"],
            "generation_time": result["timing"]["generation_time"],
            "total_time": result["timing"]["total_time"]
        }

        # Évaluation de la qualité de réponse
        if "context_hits" in result:
            response_quality = self.generator.evaluate_response_quality(
                question, result["context_hits"], result["answer"]
            )
            metrics.update(response_quality)

        # Comparaison avec réponse attendue si fournie
        if expected_answer:
            metrics["expected_match"] = self._compare_answers(
                result["answer"], expected_answer
            )

        return metrics

    def _compare_answers(self, generated: str, expected: str) -> Dict[str, float]:
        """
        Compare la réponse générée avec la réponse attendue
        
        Args:
            generated: Réponse générée
            expected: Réponse attendue
            
        Returns:
            Métriques de comparaison
        """
        from rag.helpers import extract_keywords

        gen_keywords = set(extract_keywords(generated.lower()))
        exp_keywords = set(extract_keywords(expected.lower()))

        if not exp_keywords:
            return {"keyword_overlap": 0.0}

        overlap = len(gen_keywords & exp_keywords)
        return {
            "keyword_overlap": overlap / len(exp_keywords),
            "generated_keywords": len(gen_keywords),
            "expected_keywords": len(exp_keywords)
        }

    def _update_stats(self, retrieval_time: float, generation_time: float,
                      total_time: float) -> None:
        """Met à jour les statistiques de session"""
        self.session_stats["queries_processed"] += 1
        n = self.session_stats["queries_processed"]

        # Moyenne mobile
        self.session_stats["avg_retrieval_time"] = (
            (self.session_stats["avg_retrieval_time"]
             * (n - 1) + retrieval_time) / n
        )
        self.session_stats["avg_generation_time"] = (
            (self.session_stats["avg_generation_time"]
             * (n - 1) + generation_time) / n
        )
        self.session_stats["avg_total_time"] = (
            (self.session_stats["avg_total_time"] * (n - 1) + total_time) / n
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistiques complètes de l'engine
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            "session": self.session_stats.copy(),
            "retriever": self.retriever.get_stats(),
            "initialized": self.retriever.runtime_data is not None,
            "llm_available": self.llm is not None
        }

        return stats

    def update_config(self, retriever_config: Optional[Dict] = None,
                      generator_config: Optional[Dict] = None) -> None:
        """
        Met à jour la configuration des composants
        
        Args:
            retriever_config: Configuration du retriever
            generator_config: Configuration du generator
        """
        if retriever_config:
            self.retriever.update_config(**retriever_config)

        if generator_config:
            if "max_chars" in generator_config:
                self.generator.max_chars = generator_config["max_chars"]

    def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé de l'engine
        
        Returns:
            État des composants
        """
        health = {
            "overall": "healthy",
            "components": {},
            "issues": []
        }

        # Vérifie le retriever
        if not self.retriever.runtime_data:
            health["components"]["retriever"] = "error"
            health["issues"].append("Index FAISS non chargé")
            health["overall"] = "error"
        else:
            health["components"]["retriever"] = "healthy"

        # Vérifie le generator
        if not self.llm:
            health["components"]["generator"] = "warning"
            health["issues"].append("LLM non configuré")
            if health["overall"] == "healthy":
                health["overall"] = "warning"
        else:
            health["components"]["generator"] = "healthy"

        return health
