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
    """Dictionary file path and score thresholds."""

    path: str = "dictionaries/XwiJeffChenList.txt"
    min_word_score: int = 50
    min_2letter_score: int = 30


class GoCrosswordConfig(BaseModel):
    """Settings for the go-crossword Docker filler."""

    docker_image: str = "crossword-generator/go-crossword-cli:latest"
    timeout: int = 60
    threads: int = 100
    output_format: str = "json"


class GenxwordConfig(BaseModel):
    """Settings for the genxword filler (placeholder)."""


class FillConfig(BaseModel):
    """Grid filler configuration."""

    provider: str = "go-crossword"
    max_retries: int = 5
    go_crossword: GoCrosswordConfig = GoCrosswordConfig()
    genxword: GenxwordConfig = GenxwordConfig()


class FillGradingConfig(BaseModel):
    """Fill quality grading thresholds."""

    min_score: int = 70
    retry_on_fail: bool = True


class ClueGradingConfig(BaseModel):
    """Clue quality grading thresholds."""

    min_score: int = 70
    regenerate_on_fail: bool = True


class GradingConfig(BaseModel):
    """Combined grading configuration."""

    fill: FillGradingConfig = FillGradingConfig()
    clue: ClueGradingConfig = ClueGradingConfig()


class OllamaConfig(BaseModel):
    """Ollama LLM provider settings."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout: int = 120


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"
    ollama: OllamaConfig = OllamaConfig()


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
