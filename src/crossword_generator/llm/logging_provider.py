"""LLM provider wrapper that writes structured JSONL call logs."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crossword_generator.config import LLMLoggingConfig, find_project_root
from crossword_generator.llm.base import LLMCallResponse, LLMProvider


class LoggingLLMProvider(LLMProvider):
    """Decorates an LLM provider and records each call as one JSONL object."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        log_path: Path,
        context: dict[str, object] | None = None,
        enabled: bool = True,
    ) -> None:
        self._inner = inner
        self._log_path = log_path
        self._context = dict(context or {})
        self._enabled = enabled

    @property
    def name(self) -> str:
        return self._inner.name

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
        request_id = str(uuid.uuid4())
        started_at = _utc_now()
        started = time.monotonic()
        try:
            result = self._inner.generate_with_details(
                prompt,
                system=system,
                **kwargs,
            )
        except Exception as exc:
            self._write_record(
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "context": dict(self._context),
                    "provider": self._inner.name,
                    "model": _model_from_kwargs(kwargs),
                    "request": _fallback_request(prompt, system, kwargs),
                    "response": None,
                    "usage": {},
                    "cost": {},
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
            )
            raise

        self._write_record(
            {
                "schema_version": 1,
                "request_id": request_id,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "context": dict(self._context),
                "provider": result.provider,
                "model": result.model,
                "request": result.request
                or _fallback_request(prompt, system, kwargs),
                "response": result.response or {"text": result.text},
                "usage": result.usage,
                "cost": result.cost,
                "error": None,
            }
        )
        return result

    def is_available(self) -> bool:
        return self._inner.is_available()

    def _write_record(self, record: dict[str, object]) -> None:
        if not self._enabled:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def resolve_llm_logging(
    config: LLMLoggingConfig,
    *,
    project_root: Path | None = None,
) -> tuple[bool, Path]:
    """Resolve logging settings from config plus environment overrides."""
    enabled = config.enabled
    env_enabled = os.environ.get("CROSSWORD_LLM_LOG_ENABLED")
    if env_enabled is not None:
        enabled = env_enabled.strip().lower() not in {"0", "false", "no", "off"}

    raw_path = os.environ.get("CROSSWORD_LLM_LOG_PATH") or config.path
    path = Path(raw_path)
    if not path.is_absolute():
        path = (project_root or find_project_root()) / path
    return enabled, path


def _fallback_request(
    prompt: str,
    system: str | None,
    kwargs: dict[str, object],
) -> dict[str, object]:
    return {
        "prompt": prompt,
        "system": system,
        "kwargs": dict(kwargs),
        "prompt_chars": len(prompt),
        "system_chars": len(system) if system else 0,
    }


def _model_from_kwargs(kwargs: dict[str, object]) -> str | None:
    model = kwargs.get("model")
    return str(model) if model is not None else None


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [_jsonable(item) for item in value]
        return str(value)
    return value
