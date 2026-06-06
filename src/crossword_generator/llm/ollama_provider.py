"""Ollama LLM provider implementation."""

from __future__ import annotations

import logging

import ollama

from crossword_generator.config import OllamaConfig
from crossword_generator.llm.base import LLMCallResponse, LLMProvider
from crossword_generator.llm.costs import estimate_llm_cost

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.base_url, timeout=config.timeout)

    @property
    def name(self) -> str:
        return "ollama"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> str:
        return self.generate_with_details(prompt, system=system, **kwargs).text

    def generate_with_details(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> LLMCallResponse:
        model = kwargs.get("model", self._config.model)
        temperature = kwargs.get("temperature", 0.7)
        response = self._client.generate(
            model=str(model),
            prompt=prompt,
            system=system,
            options={"temperature": float(temperature)},  # type: ignore[arg-type]
        )
        text = str(response["response"])
        usage = _extract_usage(response)
        return LLMCallResponse(
            text=text,
            provider=self.name,
            model=str(model),
            request={
                "model": str(model),
                "prompt": prompt,
                "system": system,
                "options": {"temperature": float(temperature)},
                "prompt_chars": len(prompt),
                "system_chars": len(system) if system else 0,
            },
            response={
                "text": text,
                "text_chars": len(text),
                "done": response.get("done"),
                "done_reason": response.get("done_reason"),
                "total_duration": response.get("total_duration"),
                "load_duration": response.get("load_duration"),
                "prompt_eval_duration": response.get("prompt_eval_duration"),
                "eval_duration": response.get("eval_duration"),
            },
            usage=usage,
            cost=estimate_llm_cost(self.name, str(model), usage),
        )

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            logger.debug("Ollama server not reachable at %s", self._config.base_url)
            return False


def _extract_usage(response: dict[str, object]) -> dict[str, int | float]:
    usage: dict[str, int | float] = {}
    key_map = {
        "prompt_eval_count": "input_tokens",
        "eval_count": "output_tokens",
    }
    for source_key, target_key in key_map.items():
        value = response.get(source_key)
        if isinstance(value, int | float):
            usage[target_key] = value
    return usage
