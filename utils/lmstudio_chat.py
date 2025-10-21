# utils/lmstudio_chat.py
from __future__ import annotations
import os
import httpx
from typing import List, Dict, Any, Union, Optional

BASE = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
TIMEOUT = int(os.getenv("LMSTUDIO_TIMEOUT_SEC", "120"))
# should match an id from GET /v1/models
MODEL = os.getenv("LMSTUDIO_MODEL", "")
MAXTOK = int(os.getenv("LMSTUDIO_MAX_TOKENS", "256"))

_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


class _AIMessage:
    def __init__(self, content: str, raw: Optional[Dict[str, Any]] = None):
        self.content = content
        self.raw = raw or {}


class LMStudioChat:
    """
    OpenAI-compatible wrapper with .invoke(...).
    It auto-detects which endpoint LM Studio exposes:
      1) /v1/chat/completions  (OpenAI Chat API)
      2) /v1/responses         (new unified endpoint)
      3) /v1/completions       (legacy text completion)
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or MODEL
        self._http = httpx.Client(timeout=TIMEOUT)
        self._endpoint = None  # "chat", "responses", or "completions"
        self._detect_endpoint()

    def _detect_endpoint(self):
        # Prefer chat/completions → responses → completions
        for kind, path, payload in [
            ("chat",       f"{BASE}/chat/completions", {"model": self.model,
             "messages": [{"role": "user", "content": "ping"}]}),
            ("responses",  f"{BASE}/responses",        {"model": self.model,
             "input": [{"role": "user", "content": "ping"}]}),
            ("completions", f"{BASE}/completions",
             {"model": self.model, "prompt": "ping"}),
        ]:
            try:
                r = self._http.post(path, json=payload, headers=_HEADERS)
                if r.status_code < 400:
                    self._endpoint = kind
                    return
            except Exception:
                pass
        raise RuntimeError(
            f"LM Studio API not available at {BASE}. "
            f"Enable the local server in LM Studio and ensure the model id is valid. "
            f"Tip: GET {BASE}/models to list ids."
        )

    def _build_messages(self, input_: Union[str, List[Dict[str, str]]], system: Optional[str]) -> List[Dict[str, str]]:
        if isinstance(input_, str):
            msgs = [{"role": "user", "content": input_}]
        else:
            msgs = list(input_)
        if system:
            if not msgs or msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": system}] + msgs
        return msgs

    def invoke(
        self,
        input_: Union[str, List[Dict[str, str]]],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        system: str = "You are a concise, helpful assistant.",
        **kwargs: Any,
    ) -> _AIMessage:
        mtok = int(max_tokens or MAXTOK)

        if self._endpoint == "chat":
            url = f"{BASE}/chat/completions"
            payload = {
                "model": self.model,
                "messages": self._build_messages(input_, system),
                "temperature": float(temperature),
                "max_tokens": mtok,
            }
            r = self._http.post(url, json=payload, headers=_HEADERS)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]

        elif self._endpoint == "responses":
            url = f"{BASE}/responses"
            payload = {
                "model": self.model,
                "input": self._build_messages(input_, system),
                "temperature": float(temperature),
                "max_output_tokens": mtok,  # some servers accept max_tokens too
            }
            r = self._http.post(url, json=payload, headers=_HEADERS)
            r.raise_for_status()
            data = r.json()
            # responses schema: choices[].message.content or output_text
            content = (
                data.get("output_text")
                or (data.get("choices", [{}])[0].get("message") or {}).get("content")
                or ""
            )

        else:  # "completions"
            url = f"{BASE}/completions"
            # flatten messages into a single prompt
            if isinstance(input_, str):
                prompt = input_
            else:
                # simple concat; adjust if you keep a system prompt
                parts = []
                if system:
                    parts.append(f"[system] {system}")
                for m in input_:
                    parts.append(f"[{m.get('role')}] {m.get('content')}")
                prompt = "\n".join(parts)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": float(temperature),
                "max_tokens": mtok,
            }
            r = self._http.post(url, json=payload, headers=_HEADERS)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["text"]

        return _AIMessage(content=content, raw=data)
