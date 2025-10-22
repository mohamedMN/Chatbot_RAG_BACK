# llm/base.py
from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Interface standardisée pour tous les LLMs."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Génère une réponse à partir d'un prompt."""
        pass

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Alternative avec format messages."""
        # Par défaut, convertir en prompt simple
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.generate(prompt, max_tokens, temperature)
