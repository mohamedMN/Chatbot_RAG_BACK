# llm/ollama_client.py
from __future__ import annotations
import os
import httpx
from typing import List, Dict, Any, Iterable, Optional

LMSTUDIO_BASE_URL = os.getenv(
    "LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_TIMEOUT_SEC = int(os.getenv("LMSTUDIO_TIMEOUT_SEC", "120"))
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "llama3.2:1b-instruct")
LMSTUDIO_MAX_TOKENS = int(os.getenv("LMSTUDIO_MAX_TOKENS", "256"))

_client = httpx.Client(timeout=LMSTUDIO_TIMEOUT_SEC)
_hdrs = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}",
         "Content-Type": "application/json"}


def chat_once(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = None,
    num_ctx: int = None,      # kept for signature compatibility, unused by LM Studio
    timeout: float = None,
    system_prompt: str = "You are a concise, helpful assistant.",
) -> str:
    """Drop-in replacement: same signature, LM Studio under the hood."""
    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens or LMSTUDIO_MAX_TOKENS),
        # "stream": False,  # default
    }
    r = _client.post(url, json=payload, headers=_hdrs)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def stream_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = None,
) -> Iterable[str]:
    """
    Optional: if some code streams from 'ollama_client', keep a compatible stream.
    Returns text chunks. If you don't use streaming anywhere, you can omit this.
    """
    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens or LMSTUDIO_MAX_TOKENS),
        "stream": True,
    }
    with _client.stream("POST", url, json=payload, headers=_hdrs) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            # LM Studio streams OpenAI-style "data: {json}"
            if line.startswith(b"data: "):
                chunk = line[6:]
                if chunk == b"[DONE]":
                    break
                try:
                    obj = httpx.Response(
                        200, content=chunk).json()  # quick parse
                except Exception:
                    continue
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
