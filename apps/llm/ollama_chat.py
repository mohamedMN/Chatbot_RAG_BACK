# backend/apps/llm/ollama_chat.py
from __future__ import annotations
import os
from functools import lru_cache
from ollama import Client  # official python client

from apps.index.cleanup import cleanup_copy_if_ollama_down


def _assert_server_up(base_url: str) -> None:
    c = Client(host=base_url)
    _ = c.list()


def _ensure_model_available(model: str, base_url: str) -> None:
    _assert_server_up(base_url)
    c = Client(host=base_url)
    listed = c.list().get("models", []) or []
    names = {m.get("model") or m.get("name") for m in listed}
    if model not in names:
        raise RuntimeError(
            f"Ollama model '{model}' not found locally.\n"
            f"Run:\n"
            f"    OLLAMA_HOST={base_url} ollama pull {model}\n"
        )


@lru_cache(maxsize=1)
def get_chat_llm(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.2,
):
    model = model or os.getenv("OLLAMA_MODEL", "deepseek-coder:instruct")
    base_url = base_url or os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

    try:
        _ensure_model_available(model, base_url)
    except Exception:
        # AUTO-CLEANUP: Ollama is offline — delete the working copy to keep only original
        cleanup_copy_if_ollama_down(base_url)
        # re-raise so the caller knows Ollama is down
        raise

    from langchain_ollama import ChatOllama
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
