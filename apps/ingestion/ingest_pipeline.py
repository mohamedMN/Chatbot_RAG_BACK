from typing import Dict, Any, List
from .schema import IngestionConfig, EmbeddedChunk
from .chunk_text import DocumentChunker
from .embeddings import DocumentEmbedder
from .fusion import EmbeddingsFusion


class IngestionPipeline:
    """Main ingestion pipeline orchestrator"""

    def __init__(self, data_dir: str = None):
        self.config = IngestionConfig(
            data_dir) if data_dir else IngestionConfig()
        self.chunker = DocumentChunker(self.config)
        self.embedder = DocumentEmbedder(self.config)
        self.fusion = EmbeddingsFusion(self.config)

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete ingestion pipeline"""
        print("=" * 50)
        print("🚀 STARTING RAG INGESTION PIPELINE")
        print("=" * 50)

        results = {
            'success': False,
            'chunks_count': 0,
            'embeddings_count': 0,
            'files_created': [],
            'error': None
        }

        try:
            # Step 1: Document Chunking
            print("\n📄 STEP 1: Document Chunking")
            chunks = self.chunker.chunk_documents()

            if not chunks:
                print("⚠️  No chunks created. Pipeline stopped.")
                return results

            results['chunks_count'] = len(chunks)

            # Step 2: Generate Embeddings
            print("\n🧠 STEP 2: Embedding Generation")
            embedded_chunks = self.embedder.embed_chunks(chunks)

            if not embedded_chunks:
                print("⚠️  No embeddings created. Pipeline stopped.")
                return results

            results['embeddings_count'] = len(embedded_chunks)

            # Step 3: Fusion Process
            print("\n🔗 STEP 3: Fusion Process")
            fusion_results = self.fusion.process_fusion(embedded_chunks)

            # Update results
            results['success'] = True
            results['files_created'] = [
                self.config.chunks_embeddings_file,
                self.config.faiss_index_file,
                self.config.faiss_metadata_file
            ]
            results['total_embeddings'] = fusion_results['total_chunks']

            print("\n" + "=" * 50)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 50)
            print(f"📊 Total chunks processed: {results['chunks_count']}")
            print(f"🧠 Total embeddings: {results['total_embeddings']}")
            print(f"📁 Files created:")
            for file_path in results['files_created']:
                print(f"   • {file_path}")

            return results

        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            results['error'] = str(e)
            return results

    def run_chunking_only(self) -> List:
        """Run only the chunking process"""
        print("📄 Running chunking only...")
        return self.chunker.chunk_documents()

    def run_embedding_only(self, chunks) -> List[EmbeddedChunk]:
        """Run only the embedding process"""
        print("🧠 Running embedding only...")
        return self.embedder.embed_chunks(chunks)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of pipeline files"""
        import os

        status = {
            'data_directory': self.config.data_dir,
            'files_exist': {},
            'file_sizes': {}
        }

        files_to_check = [
            ('chunks_embeddings', self.config.chunks_embeddings_file),
            ('faiss_index', self.config.faiss_index_file),
            ('faiss_metadata', self.config.faiss_metadata_file)
        ]

        for name, file_path in files_to_check:
            exists = os.path.exists(file_path)
            status['files_exist'][name] = exists

            if exists:
                try:
                    size = os.path.getsize(file_path)
                    status['file_sizes'][name] = f"{size / 1024 / 1024:.2f} MB"
                except:
                    status['file_sizes'][name] = "Unknown"
            else:
                status['file_sizes'][name] = "N/A"

        return status

# Convenience function for quick usage


def run_ingestion_pipeline(data_dir: str = None) -> Dict[str, Any]:
    """Quick function to run the complete pipeline"""
    pipeline = IngestionPipeline(data_dir)
    return pipeline.run_complete_pipeline()


# Main execution
if __name__ == "__main__":
    # Run the pipeline
    results = run_ingestion_pipeline()

    if results['success']:
        print(f"\n🎉 Success! Created {results['total_embeddings']} embeddings")
    else:
        print(f"\n💥 Failed: {results.get('error', 'Unknown error')}")
