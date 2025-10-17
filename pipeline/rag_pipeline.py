# rag/pipeline_faiss.py
"""
Pipeline RAG complet avec indexation FAISS.

Étapes :
1) Chargement des documents
2) Découpage en chunks
3) Embeddings
4) Construction & sauvegarde de l’index FAISS
5) Sauvegarde des artefacts (chunks/embeddings/stats)

La classe conserve l’API d’origine :
- RAGPipelineWithFAISS.run_full_pipeline(...)
- run_rag_pipeline_with_faiss(...)

Dépendances (inchangées) :
- config.settings.settings  : chemins & conf
- core.document_loader      : DocumentLoader
- core.chunker              : DocumentChunker
- core.embedder             : ChunkEmbedder
- indexing.faiss_indexer    : FAISSIndexer
- rag.rag_engine            : RAGEngine (pour test rapide)
- utils.file_utils          : save_json
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from core.embedder import ChunkEmbedder
from indexing.faiss_indexer import FAISSIndexer
from rag.rag_engine import RAGEngine
from utils.file_utils import save_json

log = logging.getLogger(__name__)


class RAGPipelineWithFAISS:
    """Pipeline RAG complet incluant la création d'un index FAISS."""

    def __init__(self) -> None:
        self.document_loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.embedder = ChunkEmbedder()
        self.faiss_indexer = FAISSIndexer()

        # Statistiques du pipeline (agrégées pour un résumé final)
        self.stats: Dict[str, Any] = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "faiss_index_created": False,
            "processing_time": 0.0,
            "errors": [],
        }

    # --------------------------------------------------------------------- #
    # PUBLIC
    # --------------------------------------------------------------------- #

    def run_full_pipeline(
        self,
        documents_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        rebuild_index: bool = False,
    ) -> Dict[str, Any]:
        """
        Exécute le pipeline RAG complet et construit l’index FAISS.

        Args:
            documents_path: Dossier des documents à ingérer (défaut: settings.documents_path)
            output_path: Dossier où écrire les artefacts (défaut: settings.output_path)
            rebuild_index: Forcer la reconstruction de l’index même s’il existe

        Returns:
            Dictionnaire récapitulatif avec chemins et statistiques.
        """
        t0 = time.time()

        docs_path = documents_path or settings.documents_path
        out_path = output_path or settings.output_path

        log.info("=" * 60)
        log.info("DÉMARRAGE DU PIPELINE RAG COMPLET AVEC FAISS")
        log.info("=" * 60)
        log.info("Documents: %s", docs_path)
        log.info("Sortie    : %s", out_path)
        log.info("Index     : %s", settings.index_path)

        try:
            # 0) Index déjà présent ?
            if not rebuild_index and self._index_exists():
                log.info(
                    "Index FAISS déjà présent. Utilisez rebuild_index=True pour le reconstruire.")
                self.stats["processing_time"] = time.time() - t0
                self._print_summary()
                return self._create_success_result(skip_build=True)

            # 1) Charger documents
            documents = self._load_documents(docs_path)
            if not documents:
                return self._create_error_result("Aucun document trouvé")

            # 2) Chunking
            chunks = self._process_chunks(documents)
            if not chunks:
                return self._create_error_result("Aucun chunk généré")

            # 3) Embeddings
            embeddings = self._generate_embeddings(chunks)

            # 4) Index FAISS
            self._create_faiss_index(embeddings)

            # 5) Sauvegardes
            self._save_results(chunks, embeddings, out_path)

            # Temps total + résumé
            self.stats["processing_time"] = time.time() - t0
            self._print_summary()
            return self._create_success_result()

        except Exception as e:  # pragma: no cover (protection globale)
            msg = f"Erreur dans le pipeline: {e}"
            log.exception(msg)
            self.stats["errors"].append(msg)
            return self._create_error_result(msg)

    def test_rag_system(self, test_question: str = "Qu'est-ce que webMethods ?") -> Dict[str, Any]:
        """
        Test de bout en bout après construction de l’index.
        Exécute uniquement la récupération (sans génération) pour une question simple.
        """
        log.info("Test du système RAG avec la question: %r", test_question)
        try:
            rag_engine = RAGEngine(faiss_indexer=self.faiss_indexer)
            if not rag_engine.initialize():
                return {"success": False, "error": "Échec d'initialisation de l'engine RAG"}

            res = rag_engine.retrieve_only(test_question, top_k=3)
            log.info("Chunks trouvés: %s | Temps de récupération: %ss",
                     res.get("context_count"), res.get("retrieval_time"))
            if res.get("context_count", 0) > 0:
                first = res["context_hits"][0]
                log.info("Premier résultat — score: %.3f | snippet: %s…",
                         first.get("score", 0.0), (first.get("content", "")[:100]))
            return res
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------------------- #
    # PRIVATE
    # --------------------------------------------------------------------- #

    def _index_exists(self) -> bool:
        """Vérifie si l’index FAISS (et ses métadonnées) existe déjà."""
        return (
            Path(settings.faiss_index_path).exists()
            and Path(settings.idmap_path).exists()
            and Path(settings.metadata_path).exists()
        )

    def _load_documents(self, documents_path: Path) -> List[Dict[str, Any]]:
        """Charge les documents depuis `documents_path` via DocumentLoader."""
        docs = self.document_loader.load_documents(documents_path)
        self.stats["documents_processed"] = len(docs)

        log.info("  %d document(s) chargé(s)", len(docs))
        for d in docs:
            name = Path(d.get("source", "inconnu")).name
            length = len(d.get("content", ""))
            sections = len(d.get("predefined_sections", []))
            log.debug("    %s : %d caractères, %d sections",
                      name, length, sections)

        return docs

    def _process_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Découpe les documents en chunks via DocumentChunker, avec stats."""
        all_chunks: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents, 1):
            name = Path(doc.get("source", "inconnu")).name
            log.info("    Document %d/%d : %s", i, len(documents), name)
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            log.info("      %d chunks générés", len(chunks))

        stats = self.chunker.get_chunking_stats(all_chunks)
        self.stats["chunks_created"] = stats.get("total_chunks", 0)

        log.info("  Statistiques de chunking :")
        log.info("    Total chunks : %s", stats.get("total_chunks"))
        log.info("    Tokens moyens : %.1f", stats.get("avg_tokens", 0.0))
        log.info("    Plage : %s-%s tokens",
                 stats.get("min_tokens"), stats.get("max_tokens"))

        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Génère et valide les embeddings via ChunkEmbedder."""
        embeddings = self.embedder.create_embeddings(chunks)
        self.stats["embeddings_generated"] = len(embeddings)

        validation = self.embedder.validate_embeddings(embeddings)
        log.info("  Statistiques des embeddings :")
        log.info("    Total : %d", len(embeddings))
        log.info("    Valides : %s", validation.get("valid_count"))
        log.info("    Taux de succès : %.1f%%", 100.0 *
                 validation.get("success_rate", 0.0))
        if validation.get("issues"):
            log.warning("    Problèmes détectés : %d",
                        len(validation["issues"]))

        return embeddings

    def _create_faiss_index(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> None:
        """Construit et enregistre l’index FAISS à partir des embeddings."""
        faiss_cfg = settings.get_faiss_config()
        index_type = faiss_cfg["index_type"]

        log.info("  Création de l’index %s…", index_type)
        self.faiss_indexer.create_index(embeddings_dict, index_type)
        self.faiss_indexer.save_index()

        self.stats["faiss_index_created"] = True

        idx_stats = self.faiss_indexer.get_stats()
        log.info("  Index créé : %s vecteurs | type=%s | métrique=%s",
                 idx_stats.get("total_vectors"),
                 idx_stats.get("index_type"),
                 idx_stats.get("metric"))

    def _save_results(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Dict[str, Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Sauvegarde chunks, embeddings et statistiques au format JSON."""
        output_path.mkdir(parents=True, exist_ok=True)

        chunks_path = output_path / "chunks.json"
        save_json(chunks, chunks_path)
        log.info("    Chunks sauvegardés : %s", chunks_path)

        embeddings_path = output_path / "embeddings.json"
        save_json(embeddings, embeddings_path)
        log.info("    Embeddings sauvegardés : %s", embeddings_path)

        stats_path = output_path / "pipeline_stats.json"
        save_json(self.stats, stats_path)
        log.info("    Statistiques sauvegardées : %s", stats_path)

    def _print_summary(self) -> None:
        """Affiche un récapitulatif du pipeline."""
        log.info("\n" + "=" * 60)
        log.info("PIPELINE RAG TERMINÉ")
        log.info("=" * 60)
        log.info("RÉSUMÉ :")
        log.info("  Documents traités : %s", self.stats["documents_processed"])
        log.info("  Chunks créés      : %s", self.stats["chunks_created"])
        log.info("  Embeddings générés: %s",
                 self.stats["embeddings_generated"])
        log.info("  Index FAISS       : %s",
                 "Créé" if self.stats["faiss_index_created"] else "Échec")
        log.info("  Temps d’exécution : %.2fs", self.stats["processing_time"])
        if self.stats["errors"]:
            log.warning("  Erreurs : %d", len(self.stats["errors"]))
        else:
            log.info("  Aucune erreur")
        log.info("Index FAISS : %s", settings.faiss_index_path)
        log.info("=" * 60)

    def _create_success_result(self, skip_build: bool = False) -> Dict[str, Any]:
        """Fabrique la réponse succès standardisée."""
        return {
            "success": True,
            "skip_build": skip_build,
            "faiss_index_path": str(settings.faiss_index_path),
            "idmap_path": str(settings.idmap_path),
            "metadata_path": str(settings.metadata_path),
            "chunks_path": str(settings.chunks_path),
            "embeddings_path": str(settings.embeddings_path),
            "stats": self.stats,
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Fabrique la réponse d’erreur standardisée."""
        return {
            "success": False,
            "error": error_message,
            "stats": self.stats,
        }


# ------------------------------------------------------------------------- #
# Helper d’exécution rapide (API stable)
# ------------------------------------------------------------------------- #

def run_rag_pipeline_with_faiss(
    documents_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    rebuild_index: bool = False,
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet en une seule ligne.

    Returns:
        Dictionnaire de résultat identique à `run_full_pipeline`.
    """
    pipeline = RAGPipelineWithFAISS()
    return pipeline.run_full_pipeline(documents_path, output_path, rebuild_index)
