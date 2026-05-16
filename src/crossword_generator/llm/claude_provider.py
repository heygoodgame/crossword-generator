"""Claude (Anthropic API) LLM provider implementation."""

from __future__ import annotations

import logging
import os
import time

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
        max_overload_retries = int(
            os.environ.get("CROSSWORD_CLAUDE_OVERLOAD_RETRIES", "6")
        )
        for attempt in range(max_overload_retries + 1):
            try:
                message = self._client.messages.create(
                    model=model,
                    max_tokens=self._config.max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as exc:
                if (
                    not _is_overload_error(exc)
                    or attempt >= max_overload_retries
                ):
                    raise
                sleep_seconds = min(90, 10 * (2**attempt))
                logger.warning(
                    "Claude overloaded; retrying in %ss (attempt %s/%s)",
                    sleep_seconds,
                    attempt + 1,
                    max_overload_retries,
                )
                time.sleep(sleep_seconds)
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


def _is_overload_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 529:
        return True
    class_name = exc.__class__.__name__.lower()
    return "overload" in class_name
