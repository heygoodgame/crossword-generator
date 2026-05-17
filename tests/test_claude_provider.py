"""Tests for the Claude LLM provider."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from crossword_generator.config import ClaudeConfig


class FakeOverloadedError(RuntimeError):
    pass


@pytest.fixture
def config() -> ClaudeConfig:
    return ClaudeConfig(
        model="claude-haiku-4-5-20251001", max_tokens=4096, timeout=120
    )


class TestClaudeProvider:
    def test_name(self, config: ClaudeConfig) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)
        assert provider.name == "claude"

    def test_generate_calls_messages_create(self, config: ClaudeConfig) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_text_block = MagicMock()
                mock_text_block.text = "Generated clue"
                mock_message = MagicMock()
                mock_message.content = [mock_text_block]
                mock_client.messages.create.return_value = mock_message

                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)
                result = provider.generate("Write a clue for OCEAN")

        assert result == "Generated clue"
        mock_client.messages.create.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": "Write a clue for OCEAN"}],
        )

    def test_generate_with_system_caches_block(
        self, config: ClaudeConfig
    ) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_text_block = MagicMock()
                mock_text_block.text = "ok"
                mock_message = MagicMock()
                mock_message.content = [mock_text_block]
                mock_client.messages.create.return_value = mock_message

                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)
                provider.generate("user payload", system="static rubric")

        mock_client.messages.create.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": "user payload"}],
            system=[
                {
                    "type": "text",
                    "text": "static rubric",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )

    def test_generate_with_custom_kwargs(self, config: ClaudeConfig) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_text_block = MagicMock()
                mock_text_block.text = "Custom response"
                mock_message = MagicMock()
                mock_message.content = [mock_text_block]
                mock_client.messages.create.return_value = mock_message

                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)
                result = provider.generate(
                    "prompt", model="claude-sonnet-4-20250514", temperature=0.3
                )

        assert result == "Custom response"
        mock_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.3,
            messages=[{"role": "user", "content": "prompt"}],
        )

    def test_generate_retries_overload_errors(self, config: ClaudeConfig) -> None:
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-key",
                "CROSSWORD_CLAUDE_OVERLOAD_RETRIES": "2",
            },
        ):
            with (
                patch("anthropic.Anthropic") as mock_cls,
                patch("crossword_generator.llm.claude_provider.time.sleep") as sleep,
            ):
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_text_block = MagicMock()
                mock_text_block.text = "Recovered response"
                mock_message = MagicMock()
                mock_message.content = [mock_text_block]

                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)
                mock_client.messages.create.side_effect = [
                    FakeOverloadedError("overloaded"),
                    mock_message,
                ]

                result = provider.generate("prompt")

        assert result == "Recovered response"
        assert mock_client.messages.create.call_count == 2
        sleep.assert_called_once_with(10)

    def test_is_available_false_when_no_api_key(self, config: ClaudeConfig) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)

        with patch.dict(os.environ, {}, clear=True):
            assert provider.is_available() is False

    def test_is_available_true_when_configured(self, config: ClaudeConfig) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)
            assert provider.is_available() is True

    def test_override_env_var_takes_precedence(self, config: ClaudeConfig) -> None:
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "default-key",
                "ANTHROPIC_API_KEY_CROSSWORD_GENERATOR": "override-key",
            },
        ):
            with patch("anthropic.Anthropic") as mock_cls:
                from crossword_generator.llm.claude_provider import ClaudeProvider

                ClaudeProvider(config)

        mock_cls.assert_called_once_with(api_key="override-key", timeout=120)

    def test_falls_back_to_default_env_var(self, config: ClaudeConfig) -> None:
        env = {k: v for k, v in os.environ.items()}
        env.pop("ANTHROPIC_API_KEY_CROSSWORD_GENERATOR", None)
        env["ANTHROPIC_API_KEY"] = "default-key"
        with patch.dict(os.environ, env, clear=True):
            with patch("anthropic.Anthropic") as mock_cls:
                from crossword_generator.llm.claude_provider import ClaudeProvider

                ClaudeProvider(config)

        mock_cls.assert_called_once_with(api_key="default-key", timeout=120)

    def test_is_available_true_with_override_only(
        self, config: ClaudeConfig
    ) -> None:
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY_CROSSWORD_GENERATOR": "override-key"},
            clear=True,
        ):
            with patch("anthropic.Anthropic"):
                from crossword_generator.llm.claude_provider import ClaudeProvider

                provider = ClaudeProvider(config)
                assert provider.is_available() is True

    def test_import_error_when_anthropic_missing(self, config: ClaudeConfig) -> None:
        import importlib
        import sys

        # Temporarily remove anthropic from sys.modules if present
        saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            # Need to reload the module so the import inside __init__ re-triggers
            import crossword_generator.llm.claude_provider as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="anthropic"):
                mod.ClaudeProvider(config)
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved
            else:
                sys.modules.pop("anthropic", None)
            # Reload again to restore clean state
            import crossword_generator.llm.claude_provider as mod2

            importlib.reload(mod2)
