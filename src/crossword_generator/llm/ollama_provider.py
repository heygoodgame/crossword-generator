"""Ollama LLM provider implementation."""

from __future__ import annotations

import logging

import ollama

from crossword_generator.config import OllamaConfig
from crossword_generator.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.base_url, timeout=config.timeout)

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, prompt: str, **kwargs: object) -> str:
        model = kwargs.get("model", self._config.model)
        temperature = kwargs.get("temperature", 0.7)
        response = self._client.generate(
            model=str(model),
            prompt=prompt,
            options={"temperature": float(temperature)},  # type: ignore[arg-type]
        )
        return response["response"]

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            logger.debug("Ollama server not reachable at %s", self._config.base_url)
            return False
