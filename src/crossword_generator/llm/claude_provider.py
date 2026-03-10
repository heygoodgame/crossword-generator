"""Claude (Anthropic API) LLM provider implementation."""

from __future__ import annotations

import logging
import os

from crossword_generator.config import ClaudeConfig
from crossword_generator.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """LLM provider backed by the Anthropic Claude API.

    Requires the ``anthropic`` package (optional dependency) and
    the ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(self, config: ClaudeConfig) -> None:
        self._config = config
        try:
            import anthropic

            self._client = anthropic.Anthropic(timeout=config.timeout)
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Claude provider. "
                "Install it with: uv pip install crossword-generator[claude]"
            ) from exc

    @property
    def name(self) -> str:
        return "claude"

    def generate(self, prompt: str, **kwargs: object) -> str:
        model = str(kwargs.get("model", self._config.model))
        temperature = float(kwargs.get("temperature", 0.7))
        message = self._client.messages.create(
            model=model,
            max_tokens=self._config.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text  # type: ignore[union-attr]

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.debug("anthropic package not installed")
            return False
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.debug("ANTHROPIC_API_KEY not set")
            return False
        return True
