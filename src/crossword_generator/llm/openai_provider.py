"""OpenAI (Chat Completions API) LLM provider implementation."""

from __future__ import annotations

import logging
import os
import time

from crossword_generator.config import OpenAIConfig
from crossword_generator.llm.base import LLMCallResponse, LLMProvider
from crossword_generator.llm.costs import estimate_llm_cost

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API.

    Requires the ``openai`` package (optional dependency) and an API key.
    Prefers ``OPENAI_API_KEY_CROSSWORD_GENERATOR`` when set (so individuals
    can scope a dedicated key to this project), and falls back to
    ``OPENAI_API_KEY``.

    Used as a cross-family fact-checker: routing the clue fact-check step to
    a different model family breaks the "same model grades its own confident
    error" loop that lets plain-wrong trivia (e.g. an Ice-Age-sloth clue for
    MANNY) slip through a Claude-grades-Claude pass.
    """

    def __init__(self, config: OpenAIConfig) -> None:
        self._config = config
        try:
            import openai

            api_key = _resolve_api_key()
            self._client = openai.OpenAI(api_key=api_key, timeout=config.timeout)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: uv pip install crossword-generator[openai]"
            ) from exc

    @property
    def name(self) -> str:
        return "openai"

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
        model = str(kwargs.get("model", self._config.model))
        max_rate_limit_retries = int(
            os.environ.get("CROSSWORD_OPENAI_RATELIMIT_RETRIES", "6")
        )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
        }
        # Reasoning models (o-series, gpt-5*) reject sampling params and use
        # ``max_completion_tokens`` instead of ``max_tokens``.
        if _is_reasoning_model(model):
            request_kwargs["max_completion_tokens"] = self._config.max_tokens
            if self._config.reasoning_effort:
                request_kwargs["reasoning_effort"] = self._config.reasoning_effort
        else:
            request_kwargs["max_tokens"] = self._config.max_tokens
            request_kwargs["temperature"] = float(
                kwargs.get("temperature", self._config.temperature)
            )

        for attempt in range(max_rate_limit_retries + 1):
            try:
                completion = self._client.chat.completions.create(**request_kwargs)
                break
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= max_rate_limit_retries:
                    raise
                sleep_seconds = min(90, 10 * (2**attempt))
                logger.warning(
                    "OpenAI rate-limited; retrying in %ss (attempt %s/%s)",
                    sleep_seconds,
                    attempt + 1,
                    max_rate_limit_retries,
                )
                time.sleep(sleep_seconds)

        text = _extract_text_content(completion)
        usage = _extract_usage(completion)
        return LLMCallResponse(
            text=text,
            provider=self.name,
            model=model,
            request={
                "model": model,
                "messages": messages,
                "prompt": prompt,
                "system_text": system,
                "prompt_chars": len(prompt),
                "system_chars": len(system) if system else 0,
            },
            response={
                "id": _get_attr(completion, "id"),
                "model": _get_attr(completion, "model"),
                "finish_reason": _finish_reason(completion),
                "text": text,
                "text_chars": len(text),
            },
            usage=usage,
            cost=estimate_llm_cost(self.name, model, usage),
        )

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            logger.debug("openai package not installed")
            return False
        if _resolve_api_key() is None:
            logger.debug(
                "No API key set "
                "(checked OPENAI_API_KEY_CROSSWORD_GENERATOR, OPENAI_API_KEY)"
            )
            return False
        return True


def _is_reasoning_model(model: str) -> bool:
    """True for models that reject sampling params and use reasoning_effort.

    Covers the o-series (o1/o3/o4) and gpt-5 family.
    """
    m = model.lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5"))


def _resolve_api_key() -> str | None:
    override = os.environ.get("OPENAI_API_KEY_CROSSWORD_GENERATOR")
    if override:
        logger.debug("Using OPENAI_API_KEY_CROSSWORD_GENERATOR for OpenAI API")
        return override
    return os.environ.get("OPENAI_API_KEY") or None


def _extract_text_content(completion: object) -> str:
    choices = _get_attr(completion, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response did not contain any choices")
    message = _get_attr(choices[0], "message")
    content = _get_attr(message, "content")
    if isinstance(content, str) and content:
        return content
    # Some models return content as a list of parts.
    if isinstance(content, list):
        parts = [
            _get_attr(part, "text")
            for part in content
            if isinstance(_get_attr(part, "text"), str)
        ]
        if parts:
            return "".join(parts)
    raise ValueError("OpenAI response did not contain a text content block")


def _finish_reason(completion: object) -> object:
    choices = _get_attr(completion, "choices")
    if isinstance(choices, list) and choices:
        return _get_attr(choices[0], "finish_reason")
    return None


def _extract_usage(completion: object) -> dict[str, int | float]:
    usage = _get_attr(completion, "usage")
    if usage is None:
        return {}
    result: dict[str, int | float] = {}
    # Map OpenAI usage keys to the same names the cost estimator expects.
    mapping = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
    }
    for src, dest in mapping.items():
        value = _get_attr(usage, src)
        if isinstance(value, int | float):
            result[dest] = value
    return result


def _get_attr(obj: object, name: str) -> object:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    class_name = exc.__class__.__name__.lower()
    return "ratelimit" in class_name
