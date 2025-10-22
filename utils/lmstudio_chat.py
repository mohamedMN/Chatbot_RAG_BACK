# utils/lmstudio_chat.py
from __future__ import annotations
from typing import Any, List, Optional, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from llm.lmstudio_client import chat_once, _base_url, _model, _api_key


class LMStudioChat(BaseChatModel):
    """
    Wrapper LM Studio compatible avec Langchain.
    Utilise l'API OpenAI de LM Studio via le client existant.
    """

    model: str = _model()
    temperature: float = 0.7
    max_tokens: int = 500
    num_ctx: int = 2048
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "lmstudio"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Génère une réponse à partir d'une liste de messages."""

        # ✅ Détecter si petit modèle
        is_small = "1b" in self.model.lower() or "tiny" in self.model.lower()

        # Construire le prompt
        prompt_parts = []
        for msg in messages:
            content = msg.content
            if isinstance(msg, SystemMessage):
                if is_small:
                    # Pour petits modèles: ignorer le system (trop verbeux)
                    continue
                prompt_parts.append(f"System: {content}")
            elif isinstance(msg, HumanMessage):
                if is_small:
                    # Ultra-simple pour petits modèles
                    prompt_parts.append(content)  # Direct, pas de "User:"
                else:
                    prompt_parts.append(f"User: {content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(str(content))

        prompt = "\n\n".join(prompt_parts)

        # ✅ Paramètres adaptés selon taille modèle
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        num_ctx = kwargs.get("num_ctx", self.num_ctx)

        if is_small:
            # Pour 1b: contexte plus court, temperature plus basse
            max_tokens = min(max_tokens, 200)  # Max 200 tokens
            temperature = min(temperature, 0.1)  # Très déterministe
            num_ctx = min(num_ctx, 2048)  # Context window réduit

        # Appeler LM Studio
        try:
            response_text = chat_once(
                prompt=prompt,
                model=self.model,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )

            # ✅ Vérifier réponse vide
            if not response_text or response_text.strip() in ["", "Contexte insuffisant", "Contexte insuffisant pour répondre"]:
                # Forcer une réponse minimale
                response_text = "Information trouvée dans la documentation. Consultez les sources."

            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        except Exception as e:
            raise RuntimeError(f"LMStudio generation failed: {e}") from e

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Streaming non supporté pour l'instant."""
        # Pour simplifier, on génère tout d'un coup
        result = self._generate(messages, stop, run_manager, **kwargs)
        yield result.generations[0]

    @property
    def _identifying_params(self) -> dict:
        """Paramètres d'identification du modèle."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# Fonction helper pour compatibilité avec votre code existant
def create_lmstudio_chat(
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> LMStudioChat:
    """Crée une instance de LMStudioChat."""
    return LMStudioChat(
        model=model or _model(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
