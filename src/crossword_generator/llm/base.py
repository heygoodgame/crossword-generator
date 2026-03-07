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
    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt.
            **kwargs: Provider-specific options (model, temperature, etc.).

        Returns:
            The generated text response.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM provider is reachable and ready."""
