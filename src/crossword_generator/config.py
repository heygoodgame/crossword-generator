"""Configuration loading from YAML files."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

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
    difficulty: str = "easy"
    grid_size: int = 5


class DictionaryConfig(BaseModel):
    """Dictionary file path and score thresholds.

    ``path`` / ``min_word_score`` are used for non-themed (mini) puzzles.
    ``themed_path`` / ``themed_min_word_score`` are used when a theme is
    present, falling back to the non-themed values when empty / zero.
    """

    path: str = "dictionaries/HggCuratedCrosswordList.txt"
    additional_paths: list[str] = Field(default_factory=list)
    additional_min_length: int | None = None
    additional_max_length: int | None = None
    min_word_score: int = 50
    min_2letter_score: int = 30
    themed_path: str = "dictionaries/HggScoredCrosswordList.txt"
    themed_additional_paths: list[str] = Field(default_factory=list)
    themed_additional_min_length: int | None = None
    themed_additional_max_length: int | None = None
    themed_min_word_score: int = 45
    themed_min_2letter_score: int = 30


class CSPFillerConfig(BaseModel):
    """Settings for the native Python CSP filler."""

    dictionary_path: str = "dictionaries/HggCuratedCrosswordList.txt"
    additional_dictionary_paths: list[str] = Field(default_factory=list)
    additional_dictionary_min_length: int | None = None
    additional_dictionary_max_length: int | None = None
    min_word_score: int = 50
    min_2letter_score: int = 30
    min_score_by_length: dict[int, int] = Field(default_factory=dict)
    timeout: int = 30
    timeout_by_size: dict[int, int] | None = None
    quality_tiers: list[int] = [58, 52, 45]


class FillConfig(BaseModel):
    """Grid filler configuration."""

    provider: str = "csp"
    max_retries: int = 5
    max_grid_variants: int = 100
    max_long_entries_8_9: int | None = None
    csp: CSPFillerConfig = CSPFillerConfig()


class FillGradingConfig(BaseModel):
    """Fill quality grading thresholds."""

    min_score: int = 51
    retry_on_fail: bool = True
    collect_boards: int = 1  # 1 = stop at first passing board
    llm_select: bool = False  # requires collect_boards > 1
    exact_score_count_length: int | None = None
    exact_score_count_min_score: int | None = None
    exact_score_count: int | None = None
    # Path to a plain word list (one word per line) of Jeff's Hard-list
    # entries. When set, any grid where two of these entries cross each
    # other is a hard fail. Set on Hard configs only.
    hard_cross_words_path: str | None = None


class ClueGradingConfig(BaseModel):
    """Clue quality grading thresholds."""

    min_score: int = 70
    regenerate_on_fail: bool = True
    accuracy_repair_threshold: int = 12  # repair clues below this accuracy sub-score
    fairness_repair_threshold: int = 15  # repair clues below this fairness sub-score
    craft_repair_threshold: int = 8  # repair clues below this craft sub-score
    # Repair clues below this freshness sub-score. 0 disables; Hard configs
    # raise it so evaluator-flagged too-easy clues (freshness 0-9) get repaired.
    freshness_repair_threshold: int = 0
    individual_repair_score_threshold: int = 65
    # Skip whole-puzzle regeneration and go straight to surgical repair when at
    # least this fraction of clues already pass. None disables (always regen).
    surgical_repair_pass_ratio: float = 0.8
    # Extra repair rounds after the first surgical repair, for clues that a
    # single repair pass fails to fix.
    repair_verify_attempts: int = 2
    # Rounds of repair for clues that exactly duplicate an existing clue (with
    # --avoid-existing-clues). A stuck duplicate becomes a DUPLICATE: soft error
    # and is skipped at upload, never crashing the puzzle.
    duplicate_repair_attempts: int = 4
    # Split clue generation into chunks of at most this many entries per LLM
    # call (0 = one call for the whole puzzle). Smaller chunks improve rule
    # adherence on long puzzles; the cacheable system prompt is shared.
    generation_chunk_size: int = 0
    # When chunking is on, generate the chunks concurrently instead of serially.
    # The first chunk runs alone to populate the ephemeral system-prompt cache;
    # the remaining chunks then fan out and read that warm cache rather than
    # each racing to recreate it. Off by default — opt in per config.
    parallel_chunks: bool = False
    # Max chunks generated concurrently during the fan-out stage (after the
    # first warm-up chunk). Only used when parallel_chunks is true.
    parallel_chunk_max_workers: int = 4
    fact_check_enabled: bool = True
    fact_check_scope: str = "risky"  # "risky" or "all"


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
    theme_model: str = "claude-sonnet-4-6"
    fill_selection_model: str = ""
    # Opus 4.8 for the quality-critical generative step (Phase 3). Adaptive
    # thinking only; the provider omits temperature for Opus 4.7/4.8.
    clue_generation_model: str = "claude-opus-4-8"
    # Grading is the leak/accuracy gate — Sonnet 4.6 for a stronger judge (P5).
    clue_grading_model: str = "claude-sonnet-4-6"
    clue_fact_check_model: str = "claude-sonnet-4-6"
    # Naming is a trivial creative task — Haiku is sufficient (P5). Empty falls
    # back to ``model`` (also Haiku); set explicitly for clarity.
    puzzle_naming_model: str = "claude-haiku-4-5-20251001"
    thinking_enabled: bool = False
    thinking_type: str = "adaptive"
    thinking_display: str = "omitted"
    thinking_max_tokens: int = 8192
    effort: str = ""
    clue_generation_thinking_enabled: bool = True
    clue_generation_effort: str = "medium"
    max_tokens: int = 4096
    timeout: int = 120

    def model_for(self, step: str) -> str:
        """Return the resolved model ID for a pipeline step.

        Args:
            step: One of "theme", "fill_selection", "clue_generation",
                  "clue_grading", "clue_fact_check", "puzzle_naming".

        Returns:
            The per-step model if set, otherwise the default ``model``.
        """
        field = f"{step}_model"
        value = getattr(self, field, "")
        return value or self.model


class LLMLoggingConfig(BaseModel):
    """Structured JSONL logging for LLM requests and responses."""

    enabled: bool = True
    path: str = "output/llm-calls.jsonl"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"  # "ollama" or "claude"
    ollama: OllamaConfig = OllamaConfig()
    claude: ClaudeConfig = ClaudeConfig()
    logging: LLMLoggingConfig = LLMLoggingConfig()


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
