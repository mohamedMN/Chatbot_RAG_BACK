# backend/rag/embeddings.py
from __future__ import annotations
import os
import math
import time
import logging
import random
from typing import List, Dict, Any, Optional
from functools import lru_cache

log = logging.getLogger("rag.embeddings")

# Keep CPU thread counts small to avoid long stalls on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# -------------------- Environment / defaults --------------------
LMSTUDIO_BASE_URL = os.getenv(
    "LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_EMBED_MODEL = os.getenv(
    "LMSTUDIO_EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "bge-m3"))

DEFAULT_DIM = int(os.getenv("LMSTUDIO_EMBED_DIM",
                  os.getenv("EMBEDDING_DIMENSION", "1024")))
DEFAULT_BATCH = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBED_NORMALIZE = os.getenv("FAISS_NORMALIZE_VECTORS", "true").lower() in {
    "1", "true", "yes"}

REQUEST_TIMEOUT = float(os.getenv("EMBED_REQUEST_TIMEOUT", "20"))
MAX_RETRIES = int(os.getenv("EMBED_REQUEST_RETRIES", "2"))

# -------------------- OpenAI-compatible client (LM Studio) --------------------


@lru_cache(maxsize=1)
def _client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "The 'openai' package is required for LM Studio embeddings. "
            "Install with: pip install openai"
        ) from e
    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


def _normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _lmstudio_embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Single request to LM Studio for embeddings (OpenAI-compatible).
    Retries a couple of times on connection/timeouts.
    """
    if not texts:
        return []

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = _client().embeddings.create(
                model=LMSTUDIO_EMBED_MODEL,
                input=texts,
                timeout=REQUEST_TIMEOUT,
            )
            # LM Studio follows OpenAI semantics: data[i].embedding
            vecs = [list(map(float, item.embedding)) for item in resp.data]
            return vecs
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise

    # Unreachable (raised above), but satisfy type checker
    if last_exc:
        raise last_exc
    return []

# -------------------- Public class --------------------


class EmbeddingModel:
    """
    Minimal LM Studio embedder:
      - uses LM Studio /embeddings (OpenAI-compatible)
      - batching
      - optional L2 normalization for cosine/IP FAISS
      - deterministic dummy fallback helper (not used by default unless caller wants it)
    """

    def __init__(
        self,
        model_name: str = LMSTUDIO_EMBED_MODEL,
        batch_size: int = DEFAULT_BATCH,
        dimension: int = DEFAULT_DIM,
        normalize: bool = EMBED_NORMALIZE,
    ):
        self.model_name = model_name
        self.embedding_dim = int(dimension)
        self.batch_size = max(1, int(batch_size))
        self.normalize = bool(normalize)

        # Light ping to log connectivity (non-fatal if it fails here;
        # first embed() call will raise clearly).
        try:
            _ = _lmstudio_embed_batch(["ping"])
            log.info("[embeddings] LM Studio ready at %s (model=%s, dim≈%d)",
                     LMSTUDIO_BASE_URL, self.model_name, self.embedding_dim)
        except Exception as e:
            log.warning("[embeddings] LM Studio ping failed: %s", e)

    # ---- public API ----
    def embed(self, texts: List[str]) -> List[List[float]]:
        texts = [t if isinstance(t, str) else str(t) for t in (texts or [])]
        if not texts:
            return []

        out: List[List[float]] = []
        t0 = time.perf_counter()

        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                vecs = _lmstudio_embed_batch(batch)
                if self.normalize:
                    vecs = [_normalize(v) for v in vecs]
                out.extend(vecs)

            dt = (time.perf_counter() - t0) * 1000
            log.debug("[embeddings] encoded %d texts in %.1f ms (batch=%d)",
                      len(texts), dt, self.batch_size)
            return out

        except Exception as e:
            # Surface a clear error — upstream caller can choose to catch and swap to dummy.
            log.error("[embeddings] LM Studio encode failed: %s", e)
            raise

    # --- optional deterministic dummy (not used automatically) ---
    def embed_dummy(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        dim = int(self.embedding_dim or DEFAULT_DIM)
        for t in texts:
            random.seed(hash(t) & 0xFFFFFFFF)
            v = [random.uniform(-1, 1) for _ in range(dim)]
            out.append(_normalize(v))
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "provider": "lmstudio",
            "base_url": LMSTUDIO_BASE_URL,
            "api_key_set": bool(LMSTUDIO_API_KEY),
            "model_name": self.model_name,
            "dimension": self.embedding_dim,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
        }
