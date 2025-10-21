# backend/core/st_embedder.py
from __future__ import annotations
import math
import os
import random
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENT_AVAIL = True
except Exception:
    _SENT_AVAIL = False

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDING_BATCH_SIZE

_ST: Optional["SentenceTransformer"] = None


def _ensure_st(model_name: str) -> Optional["SentenceTransformer"]:
    global _ST
    if not _SENT_AVAIL:
        return None
    if _ST is None:
        try:
            _ST = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[embeddings] failed to load ST model '{model_name}': {e}")
            _ST = None
    return _ST


class EmbeddingModel:
    """SentenceTransformers-only (isolated, no selector imports)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.embedding_dim = EMBEDDING_DIMENSION
        self._model = _ensure_st(model_name) if _SENT_AVAIL else None
        if self._model is not None:
            try:
                self.embedding_dim = int(
                    self._model.get_sentence_embedding_dimension())
                print(
                    f"[embeddings] ST ready: {self.model_name} (dim={self.embedding_dim})")
            except Exception as e:
                print(f"[embeddings] ST dim read failed: {e}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._model is not None:
            try:
                vecs = self._model.encode(
                    texts,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )
                return vecs.tolist() if hasattr(vecs, "tolist") else vecs  # type: ignore
            except Exception as e:
                print(f"[embeddings] ST encode failed: {e}")
        # deterministic dummy fallback
        out: List[List[float]] = []
        dim = int(self.embedding_dim) if self.embedding_dim else 384
        for t in texts:
            random.seed(hash(t) % (2**32))
            v = [random.uniform(-1, 1) for _ in range(dim)]
            mag = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / mag for x in v])
        return out
