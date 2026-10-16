"""Claude (Anthropic API) LLM provider implementation."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from crossword_generator.config import ClaudeConfig
from crossword_generator.llm.base import LLMCallResponse, LLMProvider
from crossword_generator.llm.costs import estimate_llm_cost

logger = logging.getLogger(__name__)

# Server-side refusal fallback (beta): when a safety classifier declines a
# request, the API re-runs it on Anthropic's recommended substitute for that
# refusal category inside the same call, instead of returning the refusal.
_SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeProvider(LLMProvider):
    """LLM provider backed by the Anthropic Claude API.

    Requires the ``anthropic`` package (optional dependency) and an
    API key. Prefers ``ANTHROPIC_API_KEY_CROSSWORD_GENERATOR`` when set
    (so individuals can scope a dedicated key to this project), and
    falls back to ``ANTHROPIC_API_KEY``.
    """

    def __init__(self, config: ClaudeConfig) -> None:
        self._config = config
        self._client = self._make_client()

    def _make_client(self) -> Any:
        try:
            import anthropic

            api_key = _resolve_api_key()
            return anthropic.Anthropic(api_key=api_key, timeout=self._config.timeout)
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Claude provider. "
                "Install it with: uv pip install crossword-generator[claude]"
            ) from exc

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()

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
        if self._client is None:
            self._client = self._make_client()
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
            "messages": [{"role": "user", "content": prompt}],
        }
        # Opus 4.7/4.8 removed the sampling params (temperature/top_p/top_k):
        # sending them returns a 400. Only include temperature for models that
        # still accept it (Sonnet/Haiku and older Opus).
        if not _rejects_sampling_params(model):
            request_kwargs["temperature"] = temperature
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
        use_server_fallback = _supports_server_fallback(model)
        if use_server_fallback:
            request_kwargs["betas"] = [_SERVER_FALLBACK_BETA]
            request_kwargs["fallbacks"] = "default"
        create = (
            self._client.beta.messages.create
            if use_server_fallback
            else self._client.messages.create
        )

        for attempt in range(max_overload_retries + 1):
            try:
                message = create(**request_kwargs)
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

        _raise_on_refusal(message, model)
        text = _extract_text_content(message.content)
        usage = _extract_usage(message)
        # A server-side fallback reports the model that actually served the
        # request; price the call at that model's rates.
        served_model = _get_attr(message, "model")
        if not isinstance(served_model, str) or not served_model:
            served_model = model
        if served_model != model:
            logger.warning(
                "Claude %s declined the request; served by fallback %s",
                model,
                served_model,
            )
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
            cost=estimate_llm_cost(self.name, served_model, usage),
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


def _rejects_sampling_params(model: str) -> bool:
    """True for models that 400 on temperature/top_p/top_k.

    Opus 4.7/4.8 and the entire Claude 5 family (Fable 5, Sonnet 5, and any
    Opus/Haiku 5) removed the sampling parameters. Sonnet/Haiku 4.x and Opus
    4.6 and earlier still accept temperature.
    """
    m = model.lower()
    if "opus-4-7" in m or "opus-4-8" in m:
        return True
    # Claude 5 family: e.g. claude-sonnet-5, claude-opus-5, claude-fable-5,
    # claude-haiku-5. Match the "-5" generation suffix (optionally followed by
    # a dotted minor or date, e.g. -5-1 / -5-20260101).
    return bool(re.search(r"-(?:sonnet|opus|haiku|fable)-5(?:\b|-)", m))


def _supports_server_fallback(model: str) -> bool:
    """True for models that accept ``fallbacks: "default"``.

    Opus 5, Opus 5.5, Sonnet 5.5 and Fable 5.1 run safety classifiers that can
    decline a request and support the "default" fallback form. Haiku 5.5 has
    no server-side fallback (a declined request stays declined), so it and
    every other model stay on the plain endpoint.
    """
    m = model.lower()
    return bool(
        re.search(r"-(?:opus-5(?:-5)?|sonnet-5-5|fable-5-1)(?:\b|-)", m)
    )


def _raise_on_refusal(message: object, model: str) -> None:
    """Raise ValueError when Claude declined the request.

    A refusal is an HTTP 200 with ``stop_reason="refusal"``; its content can be
    empty or a partial answer that must not be parsed. ValueError lets caller
    retry loops treat it like any other unusable response.
    """
    if _get_attr(message, "stop_reason") != "refusal":
        return
    details = _get_attr(message, "stop_details")
    category = _get_attr(details, "category") if details is not None else None
    explanation = (
        _get_attr(details, "explanation") if details is not None else None
    )
    raise ValueError(
        f"Claude {model} declined the request "
        f"(stop_reason=refusal, category={category!r}): {explanation or ''}"
    )


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
        block_types = [
            block.get("type")
            if isinstance(block, dict)
            else getattr(block, "type", None)
            for block in content
        ]
        if block_types and all(bt == "thinking" for bt in block_types):
            # Claude 5 thinks by default; if max_tokens is too small for the
            # prompt, the whole budget is spent thinking and the response
            # truncates (stop_reason=max_tokens) with no text block. Raise a
            # ValueError so caller retry loops treat it as a parse failure.
            raise ValueError(
                "Claude response contained only a thinking block "
                "(likely truncated at max_tokens); raise max_tokens"
            )
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
