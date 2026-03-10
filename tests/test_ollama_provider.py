"""Tests for the Ollama LLM provider."""

from unittest.mock import MagicMock, patch

import pytest

from crossword_generator.config import OllamaConfig
from crossword_generator.llm.ollama_provider import OllamaProvider


@pytest.fixture
def config() -> OllamaConfig:
    return OllamaConfig(base_url="http://localhost:11434", model="llama3", timeout=60)


class TestOllamaProvider:
    def test_name(self, config: OllamaConfig) -> None:
        with patch("crossword_generator.llm.ollama_provider.ollama.Client"):
            provider = OllamaProvider(config)
        assert provider.name == "ollama"

    def test_generate_calls_client(self, config: OllamaConfig) -> None:
        with patch("crossword_generator.llm.ollama_provider.ollama.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.generate.return_value = {"response": "Test clue"}

            provider = OllamaProvider(config)
            result = provider.generate("Write a clue for OCEAN")

        assert result == "Test clue"
        mock_client.generate.assert_called_once_with(
            model="llama3",
            prompt="Write a clue for OCEAN",
            options={"temperature": 0.7},
        )

    def test_generate_with_custom_kwargs(self, config: OllamaConfig) -> None:
        with patch("crossword_generator.llm.ollama_provider.ollama.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.generate.return_value = {"response": "Custom response"}

            provider = OllamaProvider(config)
            result = provider.generate(
                "prompt", model="mistral", temperature=0.3
            )

        assert result == "Custom response"
        mock_client.generate.assert_called_once_with(
            model="mistral",
            prompt="prompt",
            options={"temperature": 0.3},
        )

    def test_is_available_true(self, config: OllamaConfig) -> None:
        with patch("crossword_generator.llm.ollama_provider.ollama.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.list.return_value = {"models": []}

            provider = OllamaProvider(config)
            assert provider.is_available() is True

    def test_is_available_false_on_connection_error(
        self, config: OllamaConfig
    ) -> None:
        with patch("crossword_generator.llm.ollama_provider.ollama.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.list.side_effect = ConnectionError("refused")

            provider = OllamaProvider(config)
            assert provider.is_available() is False
