"""
Split into Chunks module - Implements the text splitting with overlap as shown in the diagram
"""
from typing import List, Dict, Any
import re


def split_into_chunks(chunk_text: Dict[str, Any], taille_max: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split the text into smaller chunks with overlap according to the diagram
    Using the sliding window technique with an overlap
    
    Args:
        chunk_text: Dictionary with the chunked text from the previous step
        taille_max: Maximum size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        List of chunks, each with ID, content, and source information
    """
    text = chunk_text.get('content', '')
    source = chunk_text.get('source', '')
    doc_id = chunk_text.get('id', '')
    metadata = chunk_text.get('metadata', {})

    # Use a sliding window to split text with fenêtre glissante as shown in diagram
    words = text.split()
    chunks = []

    # If text is smaller than taille_max, return it as a single chunk
    if len(words) <= taille_max:
        return [{
            'id': f"{doc_id}-chunk-1",
            'content': text,
            'source': source,
            'metadata': metadata
        }]

    # Otherwise, create overlapping chunks
    i = 0
    chunk_index = 1
    while i < len(words):
        # Calculate end of this chunk (either taille_max words or the end of the text)
        end = min(i + taille_max, len(words))

        # Get the text for this chunk
        chunk_text = ' '.join(words[i:end])

        chunks.append({
            'id': f"{doc_id}-chunk-{chunk_index}",
            'content': chunk_text,
            'source': source,
            'metadata': metadata
        })

        # Move to the next chunk, accounting for overlap
        i += taille_max - overlap
        chunk_index += 1

        # If we've reached or passed the end of the text, break
        if i >= len(words):
            break

    return chunks
