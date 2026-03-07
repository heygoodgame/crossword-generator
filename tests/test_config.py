"""Tests for the configuration module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from crossword_generator.config import Config, FillConfig, load_config


class TestConfigDefaults:
    """Test that Config() produces valid defaults."""

    def test_default_puzzle(self) -> None:
        cfg = Config()
        assert cfg.puzzle.type == "mini"
        assert cfg.puzzle.grid_size == 5

    def test_default_dictionary(self) -> None:
        cfg = Config()
        assert cfg.dictionary.path == "dictionaries/XwiJeffChenList.txt"
        assert cfg.dictionary.min_word_score == 50
        assert cfg.dictionary.min_2letter_score == 30

    def test_default_fill(self) -> None:
        cfg = Config()
        assert cfg.fill.provider == "go-crossword"
        assert cfg.fill.max_retries == 5

    def test_default_grading(self) -> None:
        cfg = Config()
        assert cfg.grading.fill.min_score == 70
        assert cfg.grading.fill.retry_on_fail is True
        assert cfg.grading.clue.min_score == 70
        assert cfg.grading.clue.regenerate_on_fail is True

    def test_default_llm(self) -> None:
        cfg = Config()
        assert cfg.llm.provider == "ollama"
        assert cfg.llm.ollama.base_url == "http://localhost:11434"
        assert cfg.llm.ollama.model == "llama3"
        assert cfg.llm.ollama.timeout == 120

    def test_default_output(self) -> None:
        cfg = Config()
        assert cfg.output.directory == "output"
        assert cfg.output.formats == ["puz", "ipuz"]

    def test_nested_defaults(self) -> None:
        fill = FillConfig()
        assert (
            fill.go_crossword.docker_image
            == "crossword-generator/go-crossword-cli:latest"
        )
        assert fill.go_crossword.output_format == "json"


class TestLoadConfig:
    """Test load_config with various inputs."""

    def test_load_example_config(self, project_root: Path) -> None:
        cfg = load_config(project_root / "config.example.yaml")
        assert cfg.puzzle.type == "mini"
        assert cfg.puzzle.grid_size == 5
        assert cfg.fill.provider == "go-crossword"
        assert cfg.llm.ollama.model == "llama3"

    def test_load_explicit_path_custom_values(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "puzzle:\n  type: midi\n  grid_size: 9\ndictionary:\n  min_word_score: 40\n"
        )
        cfg = load_config(yaml_file)
        assert cfg.puzzle.type == "midi"
        assert cfg.puzzle.grid_size == 9
        assert cfg.dictionary.min_word_score == 40
        # Unspecified sections get defaults
        assert cfg.fill.provider == "go-crossword"
        assert cfg.llm.provider == "ollama"

    def test_partial_yaml_fills_defaults(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text("grading:\n  fill:\n    min_score: 80\n")
        cfg = load_config(yaml_file)
        assert cfg.grading.fill.min_score == 80
        # Other grading defaults preserved
        assert cfg.grading.clue.min_score == 70
        # Other sections all default
        assert cfg.puzzle.type == "mini"

    def test_missing_explicit_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = load_config(yaml_file)
        assert cfg == Config()

    def test_invalid_types_raise_validation_error(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("puzzle:\n  grid_size: not_a_number\n")
        with pytest.raises(ValidationError):
            load_config(yaml_file)

    def test_fallback_to_defaults_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no config files exist, load_config returns defaults."""
        import crossword_generator.config as config_module

        monkeypatch.setattr(config_module, "find_project_root", lambda: tmp_path)
        cfg = load_config()
        assert cfg == Config()
