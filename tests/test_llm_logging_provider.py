"""Tests for structured LLM call logging."""

from __future__ import annotations

import json
from pathlib import Path

from crossword_generator.llm.base import LLMCallResponse, LLMProvider
from crossword_generator.llm.logging_provider import LoggingLLMProvider


class DetailedMockLLM(LLMProvider):
    @property
    def name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> str:
        return "fallback"

    def generate_with_details(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> LLMCallResponse:
        return LLMCallResponse(
            text="logged response",
            provider="mock",
            model="mock-model",
            request={
                "prompt": prompt,
                "system": system,
                "kwargs": kwargs,
            },
            response={"text": "logged response", "text_chars": 15},
            usage={"input_tokens": 3, "output_tokens": 2},
            cost={"estimated_cost_usd": 0.01},
        )

    def is_available(self) -> bool:
        return True


def test_logging_provider_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "llm.jsonl"
    provider = LoggingLLMProvider(
        DetailedMockLLM(),
        log_path=log_path,
        context={"step": "clue_generation", "seed": 7},
    )

    result = provider.generate("user prompt", system="system prompt", temperature=0.2)

    assert result == "logged response"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == 1
    assert record["context"] == {"step": "clue_generation", "seed": 7}
    assert record["provider"] == "mock"
    assert record["model"] == "mock-model"
    assert record["request"]["prompt"] == "user prompt"
    assert record["request"]["system"] == "system prompt"
    assert record["response"]["text"] == "logged response"
    assert record["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert record["cost"] == {"estimated_cost_usd": 0.01}
    assert record["error"] is None
