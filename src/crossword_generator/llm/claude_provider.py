"""Claude (Anthropic API) LLM provider implementation."""

from __future__ import annotations

import logging
import os
import time

from crossword_generator.config import ClaudeConfig
from crossword_generator.llm.base import LLMCallResponse, LLMProvider
from crossword_generator.llm.costs import estimate_llm_cost

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """LLM provider backed by the Anthropic Claude API.

    Requires the ``anthropic`` package (optional dependency) and an
    API key. Prefers ``ANTHROPIC_API_KEY_CROSSWORD_GENERATOR`` when set
    (so individuals can scope a dedicated key to this project), and
    falls back to ``ANTHROPIC_API_KEY``.
    """

    def __init__(self, config: ClaudeConfig) -> None:
        self._config = config
        try:
            import anthropic

            api_key = _resolve_api_key()
            self._client = anthropic.Anthropic(
                api_key=api_key, timeout=config.timeout
            )
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Claude provider. "
                "Install it with: uv pip install crossword-generator[claude]"
            ) from exc

    @property
    def name(self) -> str:
        return "claude"

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
        requested_temperature = float(kwargs.get("temperature", 0.7))
        temperature = 1.0 if self._config.thinking_enabled else requested_temperature
        max_overload_retries = int(
            os.environ.get("CROSSWORD_CLAUDE_OVERLOAD_RETRIES", "6")
        )

        # Build request kwargs. When a static system prompt is provided,
        # pass it as a cache-eligible block so repeated calls with the
        # same rubric hit the prompt cache. Anthropic silently skips
        # caching for blocks below the per-model minimum, so this is
        # always safe to send.
        request_kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": (
                self._config.thinking_max_tokens
                if self._config.thinking_enabled
                else self._config.max_tokens
            ),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._config.thinking_enabled:
            thinking: dict[str, str] = {"type": self._config.thinking_type}
            if self._config.thinking_display:
                thinking["display"] = self._config.thinking_display
            request_kwargs["thinking"] = thinking
        if self._config.effort:
            request_kwargs["output_config"] = {"effort": self._config.effort}
        if system:
            request_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        for attempt in range(max_overload_retries + 1):
            try:
                message = self._client.messages.create(**request_kwargs)
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

        text = _extract_text_content(message.content)
        usage = _extract_usage(message)
        return LLMCallResponse(
            text=text,
            provider=self.name,
            model=model,
            request={
                **_json_safe_request_kwargs(request_kwargs),
                "prompt": prompt,
                "system_text": system,
                "prompt_chars": len(prompt),
                "system_chars": len(system) if system else 0,
            },
            response={
                "id": _get_attr(message, "id"),
                "model": _get_attr(message, "model"),
                "role": _get_attr(message, "role"),
                "stop_reason": _get_attr(message, "stop_reason"),
                "stop_sequence": _get_attr(message, "stop_sequence"),
                "text": text,
                "text_chars": len(text),
            },
            usage=usage,
            cost=estimate_llm_cost(self.name, model, usage),
        )

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.debug("anthropic package not installed")
            return False
        if _resolve_api_key() is None:
            logger.debug(
                "No API key set "
                "(checked ANTHROPIC_API_KEY_CROSSWORD_GENERATOR, ANTHROPIC_API_KEY)"
            )
            return False
        return True


def _resolve_api_key() -> str | None:
    override = os.environ.get("ANTHROPIC_API_KEY_CROSSWORD_GENERATOR")
    if override:
        logger.debug("Using ANTHROPIC_API_KEY_CROSSWORD_GENERATOR for Claude API")
        return override
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _extract_text_content(content: object) -> str:
    text_parts: list[str] = []
    if not isinstance(content, list):
        raise ValueError("Claude response content is not a list")

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            text = block.get("text")
        else:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
        if isinstance(text, str) and (
            block_type == "text"
            or block_type is None
            or not isinstance(block_type, str)
        ):
            text_parts.append(text)

    if not text_parts:
        raise ValueError("Claude response did not contain a text block")
    return "".join(text_parts)


def _extract_usage(message: object) -> dict[str, int | float]:
    usage = getattr(message, "usage", None)
    if usage is None and isinstance(message, dict):
        usage = message.get("usage")
    if usage is None:
        return {}

    result: dict[str, int | float] = {}
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    for key in keys:
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        if isinstance(value, int | float):
            result[key] = value
    return result


def _json_safe_request_kwargs(request_kwargs: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in request_kwargs.items()
        if key not in {"messages", "system"}
    } | {
        "messages": request_kwargs.get("messages"),
        "system": request_kwargs.get("system"),
    }


def _get_attr(obj: object, name: str) -> object:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _is_overload_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 529:
        return True
    class_name = exc.__class__.__name__.lower()
    return "overload" in class_name
