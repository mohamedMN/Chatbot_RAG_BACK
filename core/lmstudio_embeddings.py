# core/lmstudio_embeddings.py
from __future__ import annotations
import os
import time
import httpx
from typing import Sequence, List

LMSTUDIO_BASE_URL = os.getenv(
    "LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_TIMEOUT_SEC = int(os.getenv("LMSTUDIO_TIMEOUT_SEC", "120"))
LMSTUDIO_EMBED_MODEL = os.getenv(
    "LMSTUDIO_EMBED_MODEL", "text-embedding-all-minilm-l6-v2-embedding")

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_SUBBATCH = int(os.getenv("EMBEDDING_SUBBATCH", "4"))
EMBEDDING_RETRIES = int(os.getenv("EMBEDDING_RETRIES", "5"))
EMBEDDING_RETRY_DELAY = float(os.getenv("EMBEDDING_RETRY_DELAY", "1.0"))


class LMStudioEmbedder:
    """
    Drop-in embedder with .embed(texts) -> List[List[float]]
    Calls LM Studio's OpenAI-compatible /v1/embeddings.
    """

    def __init__(self, model: str | None = None):
        self.model = model or LMSTUDIO_EMBED_MODEL
        self._client = httpx.Client(timeout=LMSTUDIO_TIMEOUT_SEC)
        self._url = f"{LMSTUDIO_BASE_URL}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        # batch + subbatch to manage memory
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            for j in range(0, len(batch), EMBEDDING_SUBBATCH):
                part = batch[j:j + EMBEDDING_SUBBATCH]
                last_err = None
                for attempt in range(1, EMBEDDING_RETRIES + 1):
                    try:
                        payload = {"model": self.model, "input": list(part)}
                        r = self._client.post(
                            self._url, json=payload, headers=self._headers)
                        r.raise_for_status()
                        data = r.json()
                        vecs = [d["embedding"] for d in data.get("data", [])]
                        out.extend(vecs)
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < EMBEDDING_RETRIES:
                            time.sleep(EMBEDDING_RETRY_DELAY)
                        else:
                            raise last_err
        return out
