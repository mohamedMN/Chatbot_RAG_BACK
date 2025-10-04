import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from .schema import EmbeddedChunk, IngestionConfig


class EmbeddingsFusion:
    """Handles fusion of embeddings and FAISS index creation"""

    def __init__(self, config: IngestionConfig):
        self.config = config

    def merge_embeddings(self, new_embeddings: List[EmbeddedChunk],
                         existing_embeddings: List[EmbeddedChunk]) -> List[EmbeddedChunk]:
        """Merge new embeddings with existing ones, avoiding duplicates"""
        existing_ids = {emb.id for emb in existing_embeddings}
        merged_embeddings = existing_embeddings.copy()

        new_count = 0
        for new_emb in new_embeddings:
            if new_emb.id not in existing_ids:
                merged_embeddings.append(new_emb)
                new_count += 1

        print(
            f"✓ Merged embeddings: {len(existing_embeddings)} existing + {new_count} new = {len(merged_embeddings)} total")
        return merged_embeddings

    def build_faiss_index(self, embedded_chunks: List[EmbeddedChunk]) -> Optional[faiss.IndexFlatL2]:
        """Build FAISS index from embeddings"""
        if not embedded_chunks:
            print("No embeddings to index")
            return None

        print("=== Building FAISS Index ===")

        # Extract embeddings matrix
        embeddings_matrix = np.array(
            [chunk.embedding for chunk in embedded_chunks])

        # Create FAISS index
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_matrix.astype('float32'))

        print(
            f"✓ FAISS index built: {index.ntotal} vectors, dimension {dimension}")
        return index

    def save_faiss_index(self, index: faiss.IndexFlatL2, embedded_chunks: List[EmbeddedChunk]):
        """Save FAISS index and metadata to files"""
        if index is None:
            print("No index to save")
            return

        try:
            # Save FAISS index
            faiss.write_index(index, self.config.faiss_index_file)
            print(f"✓ FAISS index saved to: {self.config.faiss_index_file}")

            # Save metadata for index mapping
            metadata = {
                'total_vectors': index.ntotal,
                'dimension': index.d,
                'chunks_mapping': [
                    {
                        'id': chunk.id,
                        'source': chunk.source,
                        'section': chunk.section,
                        'content_preview': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content
                    }
                    for chunk in embedded_chunks
                ]
            }

            with open(self.config.faiss_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(
                f"✓ FAISS metadata saved to: {self.config.faiss_metadata_file}")

        except Exception as e:
            print(f"✗ Error saving FAISS index: {e}")
            raise

    def load_faiss_index(self) -> Optional[faiss.IndexFlatL2]:
        """Load existing FAISS index"""
        try:
            index = faiss.read_index(self.config.faiss_index_file)
            print(f"✓ FAISS index loaded: {index.ntotal} vectors")
            return index
        except Exception as e:
            print(f"No existing FAISS index found: {e}")
            return None

    def load_faiss_metadata(self) -> Dict[str, Any]:
        """Load FAISS metadata"""
        try:
            with open(self.config.faiss_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✓ FAISS metadata loaded")
            return metadata
        except Exception as e:
            print(f"No existing FAISS metadata found: {e}")
            return {}

    def process_fusion(self, new_embeddings: List[EmbeddedChunk]) -> Dict[str, Any]:
        """Main fusion process"""
        print("=== Starting Fusion Process ===")

        # Load existing embeddings
        from .embeddings import DocumentEmbedder
        embedder = DocumentEmbedder(self.config)
        existing_embeddings = embedder.load_existing_embeddings()

        # Merge embeddings
        merged_embeddings = self.merge_embeddings(
            new_embeddings, existing_embeddings)

        # Save merged embeddings
        embedder.save_embeddings(merged_embeddings)

        # Build and save FAISS index
        faiss_index = self.build_faiss_index(merged_embeddings)
        if faiss_index:
            self.save_faiss_index(faiss_index, merged_embeddings)

        print("✓ Fusion process complete")

        return {
            'total_chunks': len(merged_embeddings),
            'faiss_index': faiss_index,
            'embeddings_file': self.config.chunks_embeddings_file,
            'faiss_index_file': self.config.faiss_index_file
        }
