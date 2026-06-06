"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMCallResponse:
    """Detailed response metadata for one provider call."""

    text: str
    provider: str
    model: str | None = None
    request: dict[str, object] = field(default_factory=dict)
    response: dict[str, object] = field(default_factory=dict)
    usage: dict[str, int | float] = field(default_factory=dict)
    cost: dict[str, object] = field(default_factory=dict)


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

    def generate_with_details(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> LLMCallResponse:
        """Generate a response and return best-effort metadata.

        Providers that can expose token usage or response details should
        override this method. The default keeps existing test/mock providers
        compatible while still allowing the logging wrapper to capture prompts
        and raw responses.
        """
        text = self.generate(prompt, system=system, **kwargs)
        return LLMCallResponse(
            text=text,
            provider=self.name,
            request={
                "prompt": prompt,
                "system": system,
                "kwargs": dict(kwargs),
            },
            response={"text": text},
        )

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM provider is reachable and ready."""
