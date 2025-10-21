# core/ollama_embeddings.py
from __future__ import annotations
import os
import json
import threading
from typing import List, Dict, Any, Optional

import numpy as np

# Optional fast path via langchain_ollama
_USE_LANGCHAIN = False
try:
    # pip install langchain-ollama
    from langchain_ollama import OllamaEmbeddings as _LC_OllamaEmbeddings  # type: ignore
    _USE_LANGCHAIN = True
except Exception:
    try:
        # older community path
        from langchain_community.embeddings import OllamaEmbeddings as _LC_OllamaEmbeddings  # type: ignore
        _USE_LANGCHAIN = True
    except Exception:
        _USE_LANGCHAIN = False

try:
    import requests
except Exception:
    requests = None  # type: ignore


def _norm_rows(M: np.ndarray) -> np.ndarray:
    if M.ndim == 1:
        M = M.reshape(1, -1)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


class OllamaEmbeddingModel:
    """
    Local embedding model backed by Ollama's /api/embeddings.
    - Uses langchain_ollama if available, else plain HTTP.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        normalize: bool = True,
        timeout: float = 30.0,
    ):
        self.model_name = model_name or os.getenv(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        )
        self.normalize = bool(normalize)
        self.timeout = float(timeout)

        self._dim: Optional[int] = None
        self._lock = threading.Lock()
        self._lc = None

        if _USE_LANGCHAIN:
            try:
                self._lc = _LC_OllamaEmbeddings(
                    model=self.model_name, base_url=self.base_url)  # type: ignore
            except Exception:
                self._lc = None

        if not self._lc and requests is None:
            raise RuntimeError(
                "Neither langchain_ollama nor 'requests' is available. "
                "Install one: pip install langchain-ollama OR pip install requests"
            )

        # Soft check; actual errors will surface on first embed()
        try:
            self._check_server_and_model()
        except Exception:
            pass

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            vecs = self.embed(["probe"])
            if vecs:
                self._dim = len(vecs[0])
            else:
                self._dim = 0
        return self._dim or 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        texts = [t if isinstance(t, str) else str(t) for t in (texts or [])]
        if not texts:
            return []

        try:
            if self._lc:
                vecs = self._embed_via_langchain(texts)
            else:
                vecs = self._embed_via_http(texts)
        except Exception as e:
            raise RuntimeError(
                f"Ollama embedding failed for model '{self.model_name}' at {self.base_url}: {e}"
            )

        M = np.asarray(vecs, dtype="float32")
        if self.normalize:
            M = _norm_rows(M)

        with self._lock:
            self._dim = int(M.shape[1]) if M.ndim == 2 else int(M.size)

        return M.tolist()

    def get_embedding_info(self) -> Dict[str, Any]:
        return {
            "provider": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "dimension": self.embedding_dim,
            "normalized": self.normalize,
            "via_langchain": bool(self._lc is not None),
        }

    def _check_server_and_model(self) -> None:
        if requests is None:
            return
        url = f"{self.base_url.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        names = set()
        for m in (data.get("models") or []):
            names.add(m.get("name") or m.get("model") or "")
        if self.model_name not in names:
            raise RuntimeError(
                f"Ollama model '{self.model_name}' not found locally.\n"
                f"Run: OLLAMA_HOST={self.base_url} ollama pull {self.model_name}"
            )

    def _embed_via_langchain(self, texts: List[str]) -> List[List[float]]:
        if not self._lc:
            raise RuntimeError("langchain embeddings client not initialized")
        vecs = self._lc.embed_documents(texts)  # type: ignore
        return [list(map(float, v)) for v in vecs]

    def _embed_via_http(self, texts: List[str]) -> List[List[float]]:
        if requests is None:
            raise RuntimeError("requests not available")
        out: List[List[float]] = []
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        for t in texts:
            payload = {"model": self.model_name, "prompt": t}
            r = requests.post(url, json=payload, timeout=self.timeout)
            if r.status_code >= 400:
                # some servers prefer raw body
                r = requests.post(url, data=json.dumps(
                    payload), timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(f"Bad embeddings response: {data}")
            out.append([float(x) for x in vec])
        return out


def get_ollama_embedder(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    normalize: bool = True,
    timeout: float = 30.0,
) -> OllamaEmbeddingModel:
    return OllamaEmbeddingModel(model_name, base_url, normalize, timeout)
