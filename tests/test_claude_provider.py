"""Tests for the Claude LLM provider."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from crossword_generator.config import ClaudeConfig


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
