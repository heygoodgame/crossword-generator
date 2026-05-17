"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this LLM provider."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user-facing prompt (per-call dynamic content).
            system: Optional static system prompt (role, rubric, output
                format). Providers that support prompt caching SHOULD
                cache this block; it is the same static text across many
                calls so cache hits make sense.
            **kwargs: Provider-specific options (model, temperature, etc.).

        Returns:
            The generated text response.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM provider is reachable and ready."""
