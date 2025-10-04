"""
Identify New Chunks module - Implements the process to identify chunks without embeddings
"""
from typing import List, Dict, Any, Tuple


def identify_new_chunks(chunks: List[Dict[str, Any]],
                        existing_embeddings: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identify chunks that don't have embeddings yet
    This corresponds to the 'Identification des nouveaux' step in the diagram
    
    Args:
        chunks: List of chunks to check
        existing_embeddings: Dictionary of existing embeddings with chunk IDs as keys
        
    Returns:
        Tuple of (chunks_without_embeddings, chunks_with_embeddings)
    """
    chunks_without_embeddings = []
    chunks_with_embeddings = []

    for chunk in chunks:
        chunk_id = chunk.get('id')

        if chunk_id not in existing_embeddings:
            chunks_without_embeddings.append(chunk)
        else:
            # For chunks that already have embeddings, add the embedding to the chunk
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = existing_embeddings[chunk_id]
            chunks_with_embeddings.append(chunk_with_embedding)

    return chunks_without_embeddings, chunks_with_embeddings
