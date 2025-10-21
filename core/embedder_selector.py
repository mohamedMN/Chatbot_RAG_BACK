# core/embedder_selector.py
from __future__ import annotations
import os

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "").lower(
) or os.getenv("EMBED_SOURCE", "").lower()


def get_embedder():
    """
    Returns an object with .embed(List[str]) -> List[List[float]].
    Priority:
      - EMBED_PROVIDER=lmstudio -> LM Studio embeddings API
      - else -> sentence-transformers local (all-MiniLM-L6-v2, etc.)
    """
    if EMBED_PROVIDER == "lmstudio":
        from core.lmstudio_embeddings import LMStudioEmbedder
        return LMStudioEmbedder()

    # default: sentence-transformers local
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is not installed and EMBED_PROVIDER is not 'lmstudio'. "
            "Install with: pip install sentence-transformers torch --upgrade"
        ) from e

    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    class STEmbedder:
        def __init__(self, name: str):
            self.m = SentenceTransformer(name)

        def embed(self, texts):
            # returns List[List[float]]
            v = self.m.encode(
                list(texts), convert_to_tensor=False, show_progress_bar=False)
            # .encode may return numpy array; ensure plain lists
            return v.tolist() if hasattr(v, "tolist") else v
    return STEmbedder(model_name)
