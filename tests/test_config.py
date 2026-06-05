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
        assert cfg.puzzle.difficulty == "easy"
        assert cfg.puzzle.grid_size == 5

    def test_default_dictionary(self) -> None:
        cfg = Config()
        assert cfg.dictionary.path == "dictionaries/HggCuratedCrosswordList.txt"
        assert cfg.dictionary.min_word_score == 50
        assert cfg.dictionary.min_2letter_score == 30
        assert cfg.dictionary.themed_path == "dictionaries/HggScoredCrosswordList.txt"
        assert cfg.dictionary.themed_min_word_score == 45

    def test_default_fill(self) -> None:
        cfg = Config()
        assert cfg.fill.provider == "csp"
        assert cfg.fill.max_retries == 5
        assert cfg.fill.max_long_entries_8_9 is None

    def test_default_grading(self) -> None:
        cfg = Config()
        assert cfg.grading.fill.min_score == 51
        assert cfg.grading.fill.retry_on_fail is True
        assert cfg.grading.clue.min_score == 70
        assert cfg.grading.clue.regenerate_on_fail is True
        assert cfg.grading.clue.fact_check_enabled is True
        assert cfg.grading.clue.fact_check_scope == "risky"

    def test_default_llm(self) -> None:
        cfg = Config()
        assert cfg.llm.provider == "ollama"
        assert cfg.llm.ollama.base_url == "http://localhost:11434"
        assert cfg.llm.ollama.model == "llama3"
        assert cfg.llm.ollama.timeout == 120
        assert cfg.llm.claude.theme_model == "claude-sonnet-4-6"
        assert cfg.llm.claude.clue_generation_model == "claude-sonnet-4-6"
        assert (
            cfg.llm.claude.clue_grading_model == "claude-haiku-4-5-20251001"
        )
        assert cfg.llm.claude.clue_fact_check_model == "claude-sonnet-4-6"
        assert cfg.llm.claude.thinking_enabled is False
        assert cfg.llm.claude.thinking_type == "adaptive"
        assert cfg.llm.claude.thinking_display == "omitted"
        assert cfg.llm.claude.thinking_max_tokens == 8192
        assert cfg.llm.claude.effort == ""
        assert cfg.llm.claude.clue_generation_thinking_enabled is True
        assert cfg.llm.claude.clue_generation_effort == "medium"

    def test_default_output(self) -> None:
        cfg = Config()
        assert cfg.output.directory == "output"
        assert cfg.output.formats == ["puz", "ipuz"]

    def test_nested_defaults(self) -> None:
        fill = FillConfig()
        assert fill.csp.timeout == 30
        assert fill.csp.min_word_score == 50


class TestLoadConfig:
    """Test load_config with various inputs."""

    def test_load_example_config(self, project_root: Path) -> None:
        cfg = load_config(project_root / "config.example.yaml")
        assert cfg.puzzle.type == "mini"
        assert cfg.puzzle.difficulty == "easy"
        assert cfg.puzzle.grid_size == 5
        assert cfg.fill.provider == "csp"
        assert cfg.llm.ollama.model == "llama3"

    @pytest.mark.parametrize(
        ("filename", "dictionary_path"),
        [
            (
                "config.easy.yaml",
                "dictionaries/hgg-easy.txt",
            ),
            ("config.easy9.yaml", "dictionaries/hgg-easy.txt"),
            ("config.hard.yaml", "dictionaries/hgg-hard-flat-55.txt"),
            ("config.hard5.yaml", "dictionaries/hgg-hard.txt"),
            ("config.hard7.yaml", "dictionaries/hgg-hard.txt"),
            ("config.hard9.yaml", "dictionaries/hgg-hard.txt"),
        ],
    )
    def test_load_phase_1_configs(
        self,
        project_root: Path,
        filename: str,
        dictionary_path: str,
    ) -> None:
        cfg = load_config(project_root / filename)
        expected_difficulty = "hard" if filename.startswith("config.hard") else "easy"
        assert cfg.puzzle.difficulty == expected_difficulty
        assert cfg.dictionary.path == dictionary_path
        expected_score = 55 if filename == "config.hard.yaml" else 50
        assert cfg.dictionary.min_word_score == expected_score
        assert cfg.fill.csp.dictionary_path == dictionary_path
        assert cfg.fill.csp.min_word_score == expected_score
        assert cfg.fill.csp.quality_tiers == [expected_score]
        if filename in {"config.easy.yaml", "config.easy9.yaml", "config.hard9.yaml"}:
            assert cfg.fill.max_long_entries_8_9 == 3
        else:
            assert cfg.fill.max_long_entries_8_9 is None
        if filename == "config.hard9.yaml":
            assert cfg.fill.csp.min_score_by_length == {8: 60, 9: 60}
        else:
            assert cfg.fill.csp.min_score_by_length == {}
        if filename in {"config.hard7.yaml", "config.hard9.yaml"}:
            assert cfg.dictionary.additional_paths == ["dictionaries/hgg-60.txt"]
            assert cfg.fill.csp.additional_dictionary_paths == [
                "dictionaries/hgg-60.txt"
            ]
        else:
            assert cfg.dictionary.additional_paths == []
            assert cfg.fill.csp.additional_dictionary_paths == []
        if filename == "config.hard9.yaml":
            assert cfg.dictionary.additional_min_length == 8
            assert cfg.dictionary.additional_max_length == 9
            assert cfg.fill.csp.additional_dictionary_min_length == 8
            assert cfg.fill.csp.additional_dictionary_max_length == 9
        else:
            assert cfg.dictionary.additional_min_length is None
            assert cfg.dictionary.additional_max_length is None
            assert cfg.fill.csp.additional_dictionary_min_length is None
            assert cfg.fill.csp.additional_dictionary_max_length is None
        assert cfg.theme.enabled is False
        assert cfg.llm.claude.theme_model == "claude-sonnet-4-6"
        assert cfg.llm.claude.clue_generation_model == "claude-sonnet-4-6"
        assert (
            cfg.llm.claude.clue_grading_model == "claude-haiku-4-5-20251001"
        )
        assert cfg.llm.claude.clue_fact_check_model == "claude-sonnet-4-6"
        assert cfg.llm.claude.thinking_enabled is False
        assert cfg.llm.claude.thinking_type == "adaptive"
        assert cfg.llm.claude.thinking_display == "omitted"
        assert cfg.llm.claude.thinking_max_tokens == 8192
        assert cfg.llm.claude.effort == ""
        assert cfg.llm.claude.clue_generation_thinking_enabled is True
        assert cfg.llm.claude.clue_generation_effort == "medium"
        assert cfg.grading.clue.fact_check_enabled is True
        assert cfg.grading.clue.fact_check_scope == "risky"
        rejected_model = "claude-sonnet-4-5" + "-20241022"
        assert rejected_model not in (project_root / filename).read_text()

    def test_load_explicit_path_custom_values(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "puzzle:\n  type: midi\n  difficulty: hard\n  grid_size: 9\n"
            "dictionary:\n  min_word_score: 40\n"
        )
        cfg = load_config(yaml_file)
        assert cfg.puzzle.type == "midi"
        assert cfg.puzzle.difficulty == "hard"
        assert cfg.puzzle.grid_size == 9
        assert cfg.dictionary.min_word_score == 40
        # Unspecified sections get defaults
        assert cfg.fill.provider == "csp"
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
