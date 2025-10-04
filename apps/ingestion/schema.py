from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os

# Default data directory
DEFAULT_DATA_DIR = os.path.join(
    "C:\\", "Users", "lenovo", "Desktop", "LA_FAC", "PFE-orange",
    "ChatBot", "backend", "apps", "data"
)


class DocumentChunk(BaseModel):
    """Schema for document chunks"""
    id: str
    content: str
    source: str
    section: str


class EmbeddedChunk(BaseModel):
    """Schema for chunks with embeddings"""
    id: str
    content: str
    source: str
    section: str
    embedding: List[float]


class SearchResult(BaseModel):
    """Schema for search results"""
    id: str
    content: str
    source: str
    section: str
    score: float


class IngestionConfig:
    """Configuration for ingestion pipeline"""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.max_chunk_length = 500
        self.chunk_overlap = 50
        self.embedding_model_name = 'all-MiniLM-L6-v2'

        # Output file paths
        self.chunks_embeddings_file = os.path.join(
            data_dir, "chunks_with_embeddings.json")
        self.faiss_index_file = os.path.join(data_dir, "faiss_index.bin")
        self.faiss_metadata_file = os.path.join(
            data_dir, "faiss_metadata.json")

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
