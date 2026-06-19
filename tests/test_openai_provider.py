"""Tests for the OpenAI LLM provider."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from crossword_generator.config import OpenAIConfig


class FakeRateLimitError(RuntimeError):
    pass


@pytest.fixture
def config() -> OpenAIConfig:
    return OpenAIConfig(model="gpt-4.1-mini", max_tokens=4096, timeout=120)


@pytest.fixture
def reasoning_config() -> OpenAIConfig:
    return OpenAIConfig(model="gpt-5", reasoning_effort="medium", max_tokens=4096)


@pytest.fixture(autouse=True)
def fake_openai_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=MagicMock()),
    )


def _completion(text: str, *, prompt_tokens: int = 100, completion_tokens: int = 20):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        id="cmpl-1",
        model="gpt-4.1-mini",
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


class TestOpenAIProvider:
    def test_name(self, config: OpenAIConfig) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                from crossword_generator.llm.openai_provider import OpenAIProvider

                provider = OpenAIProvider(config)
        assert provider.name == "openai"

    def test_non_reasoning_model_sends_temperature_and_max_tokens(
        self, config: OpenAIConfig
    ) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = _completion(
                    "checked"
                )

                from crossword_generator.llm.openai_provider import OpenAIProvider

                provider = OpenAIProvider(config)
                detailed = provider.generate_with_details(
                    "Fact-check this", system="rubric"
                )

        assert detailed.text == "checked"
        assert detailed.provider == "openai"
        assert detailed.usage == {"input_tokens": 100, "output_tokens": 20}
        # gpt-4.1-mini: 100/M * 0.40 + 20/M * 1.60
        assert detailed.cost["estimated_cost_usd"] == pytest.approx(0.000072)
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "rubric"},
                {"role": "user", "content": "Fact-check this"},
            ],
            max_tokens=4096,
            temperature=0.2,
        )

    def test_reasoning_model_uses_reasoning_effort_and_completion_tokens(
        self, reasoning_config: OpenAIConfig
    ) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = _completion("ok")

                from crossword_generator.llm.openai_provider import OpenAIProvider

                provider = OpenAIProvider(reasoning_config)
                provider.generate("payload")

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-5",
            messages=[{"role": "user", "content": "payload"}],
            max_completion_tokens=4096,
            reasoning_effort="medium",
        )

    def test_is_available_false_without_key(self, config: OpenAIConfig) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("openai.OpenAI"):
                from crossword_generator.llm.openai_provider import OpenAIProvider

                provider = OpenAIProvider(config)
                assert provider.is_available() is False

    def test_prefers_scoped_api_key(self, config: OpenAIConfig) -> None:
        env = {
            "OPENAI_API_KEY": "fallback",
            "OPENAI_API_KEY_CROSSWORD_GENERATOR": "scoped",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("openai.OpenAI") as mock_cls:
                from crossword_generator.llm.openai_provider import OpenAIProvider

                OpenAIProvider(config)
        _, kwargs = mock_cls.call_args
        assert kwargs["api_key"] == "scoped"
