"""Configuration loading from YAML files."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Walk up from this module to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


class PuzzleConfig(BaseModel):
    """Puzzle type and grid size."""

    type: str = "mini"
    grid_size: int = 5


class DictionaryConfig(BaseModel):
    """Dictionary file path and score thresholds.

    ``path`` / ``min_word_score`` are used for non-themed (mini) puzzles.
    ``themed_path`` / ``themed_min_word_score`` are used when a theme is
    present, falling back to the non-themed values when empty / zero.
    """

    path: str = "dictionaries/HggCuratedCrosswordList.txt"
    min_word_score: int = 50
    min_2letter_score: int = 30
    themed_path: str = "dictionaries/HggScoredCrosswordList.txt"
    themed_min_word_score: int = 45
    themed_min_2letter_score: int = 30


class GoCrosswordConfig(BaseModel):
    """Settings for the go-crossword Docker filler."""

    docker_image: str = "crossword-generator/go-crossword-cli:latest"
    timeout: int = 60
    threads: int = 100
    output_format: str = "json"
    dictionary_path: str | None = "dictionaries/HggScoredCrosswordList.txt"
    min_dictionary_score: int = 50


class CSPFillerConfig(BaseModel):
    """Settings for the native Python CSP filler."""

    dictionary_path: str = "dictionaries/HggCuratedCrosswordList.txt"
    min_word_score: int = 50
    min_2letter_score: int = 30
    timeout: int = 30
    timeout_by_size: dict[int, int] | None = None
    quality_tiers: list[int] = [58, 52, 45]


class FillConfig(BaseModel):
    """Grid filler configuration."""

    provider: str = "go-crossword"
    max_retries: int = 5
    max_grid_variants: int = 100
    go_crossword: GoCrosswordConfig = GoCrosswordConfig()
    csp: CSPFillerConfig = CSPFillerConfig()


class FillGradingConfig(BaseModel):
    """Fill quality grading thresholds."""

    min_score: int = 51
    retry_on_fail: bool = True
    collect_boards: int = 1  # 1 = stop at first passing board
    llm_select: bool = False  # requires collect_boards > 1


class ClueGradingConfig(BaseModel):
    """Clue quality grading thresholds."""

    min_score: int = 70
    regenerate_on_fail: bool = True
    accuracy_repair_threshold: int = 12  # repair clues below this accuracy sub-score


class GradingConfig(BaseModel):
    """Combined grading configuration."""

    fill: FillGradingConfig = FillGradingConfig()
    clue: ClueGradingConfig = ClueGradingConfig()


class OllamaConfig(BaseModel):
    """Ollama LLM provider settings."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout: int = 120


class ClaudeConfig(BaseModel):
    """Claude (Anthropic API) LLM provider settings.

    ``model`` is the default for any step without its own override.
    Per-step fields (``theme_model``, ``fill_selection_model``, etc.)
    fall back to ``model`` when set to the empty string.
    """

    model: str = "claude-haiku-4-5-20251001"
    theme_model: str = ""
    fill_selection_model: str = ""
    clue_generation_model: str = "claude-sonnet-4-5-20241022"
    clue_grading_model: str = "claude-sonnet-4-5-20241022"
    max_tokens: int = 4096
    timeout: int = 120

    def model_for(self, step: str) -> str:
        """Return the resolved model ID for a pipeline step.

        Args:
            step: One of "theme", "fill_selection", "clue_generation",
                  "clue_grading".

        Returns:
            The per-step model if set, otherwise the default ``model``.
        """
        field = f"{step}_model"
        value = getattr(self, field, "")
        return value or self.model


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"  # "ollama" or "claude"
    ollama: OllamaConfig = OllamaConfig()
    claude: ClaudeConfig = ClaudeConfig()


class ThemeConfig(BaseModel):
    """Theme generation settings for midi puzzles."""

    enabled: bool = True
    max_retries: int = 5
    num_seed_entries: int = 3
    num_candidates: int = 12
    similarity_threshold: float = 0.6
    max_avoid_in_prompt: int = 30


class OutputConfig(BaseModel):
    """Output directory and format settings."""

    directory: str = "output"
    formats: list[str] = ["puz", "ipuz"]


class Config(BaseModel):
    """Top-level configuration aggregating all sections."""

    puzzle: PuzzleConfig = PuzzleConfig()
    dictionary: DictionaryConfig = DictionaryConfig()
    fill: FillConfig = FillConfig()
    grading: GradingConfig = GradingConfig()
    llm: LLMConfig = LLMConfig()
    theme: ThemeConfig = ThemeConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Explicit path to a YAML file. If None, tries config.yaml
              then config.example.yaml in the project root, falling back
              to all defaults.

    Returns:
        A fully populated Config instance.

    Raises:
        FileNotFoundError: If an explicit path is given but does not exist.
    """
    project_root = find_project_root()

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        config_path = path
    else:
        candidates = [
            project_root / "config.yaml",
            project_root / "config.example.yaml",
        ]
        config_path = next((p for p in candidates if p.exists()), None)

    if config_path is None:
        logger.info("No config file found, using defaults")
        return Config()

    logger.info("Loading config from %s", config_path)
    raw = yaml.safe_load(config_path.read_text())

    if raw is None:
        return Config()

    return Config.model_validate(raw)
