# llm/lmstudio_client.py
from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "Missing dependency: pip install openai"
    ) from e


def _pick(var: str, *fallbacks: str, default: Optional[str] = None) -> Optional[str]:
    for k in (var, *fallbacks):
        v = os.getenv(k)
        if v:
            return v
    return default


def _base_url() -> str:
    return _pick("LMSTUDIO_BASE_URL", "OLLAMA_BASE_URL", "OLLAMA_HOST", default="http://localhost:1234/v1")


def _model() -> str:
    return _pick("LMSTUDIO_MODEL", "OLLAMA_MODEL", default="llama3.2:1b-instruct")


def _num_ctx() -> int:
    return int(_pick("LMSTUDIO_NUM_CTX", "OLLAMA_NUM_CTX", default="2048"))


def _max_tokens() -> int:
    # LM Studio’s OpenAI endpoint uses max_tokens (not num_predict)
    return int(_pick("LMSTUDIO_MAX_TOKENS", "OLLAMA_NUM_PREDICT", default="256"))


def _api_key() -> str:
    # LM Studio ignores API key, but OpenAI client requires a non-empty string
    return os.getenv("LMSTUDIO_API_KEY", "lm-studio")


def chat_once(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: Optional[int] = None,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
) -> str:
    client = OpenAI(base_url=_base_url(), api_key=_api_key())
    mdl = model or _model()
    ctx = num_ctx if isinstance(num_ctx, int) else _num_ctx()
    out_tokens = max_tokens if isinstance(max_tokens, int) else _max_tokens()

    # LM Studio accepts OpenAI-style payload; num_ctx often supported via extra_body
    resp = client.chat.completions.create(
        model=mdl,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=out_tokens,
        timeout=timeout,
        extra_body={"num_ctx": ctx},
    )
    return (resp.choices[0].message.content or "").strip()
