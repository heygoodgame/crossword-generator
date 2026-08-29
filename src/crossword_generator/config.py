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
    # Soft value-ordering penalty for answers recently used in the daily
    # schedule: inside each score tier candidates are drawn with weight
    # (1 + uses) ** -penalty, so a word used 7x is 8x less likely than a
    # fresh word to be tried first. Only applied when usage counts are
    # supplied (daily batches); never affects tier eligibility.
    answer_usage_penalty: float = 1.0


class FillConfig(BaseModel):
    """Grid filler configuration."""

    provider: str = "csp"
    max_retries: int = 5
    max_grid_variants: int = 100
    max_long_entries_8_9: int | None = None
    # Down-weights grid patterns in proportion to how many 3-letter slots
    # they contain, so a batch leans on grids with fewer 3-letter answers.
    # 0.0 keeps the raw catalog weights. Each pattern's weight is scaled by
    # exp(-bias * (threes - fewest_threes_in_catalog)).
    short_slot_bias: float = 0.0
    # Penalty on grids with more than 12 four-letter slots. Pair with
    # short_slot_bias: penalizing 3-letter slots alone steers toward
    # 4-letter-saturated grids, which fail Hard fill (hard_cross).
    four_glut_bias: float = 0.0
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
    # When true, the entries counted by exact_score_count are pre-placed in
    # the grid before filling instead of only being graded for afterwards.
    # Grading for the count still applies; seeding makes the entry a uniform
    # draw from the whole eligible pool rather than whichever entries the
    # filler converges on, which otherwise concentrates a handful of
    # easy-to-fill entries across a batch.
    seed_exact_score_entries: bool = False
    # Path to a plain word list (one word per line) of Jeff's Hard-list
    # entries. When set, any grid where two of these entries cross each
    # other is a hard fail. Set on Hard configs only.
    hard_cross_words_path: str | None = None
    # Path to the proper-noun classification file (WORD;P|C per line,
    # built by the classify-proper-nouns command). When set, grids with
    # more than max(min_proper_noun_allowance,
    # floor(max_proper_noun_ratio * answers)) proper-noun answers are a
    # hard fail — Jeff's "word puzzle, not trivia contest" rule.
    proper_nouns_path: str | None = None
    max_proper_noun_ratio: float = 0.15
    min_proper_noun_allowance: int = 2


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
    # How many times to re-fact-check after a fact-check repair rewrites a
    # clue. A repair can swap one factual error for a fresh one (e.g. an
    # ELENA clue rewritten from a bad "Helen of Troy" angle into the wrong
    # "Eva Longoria" angle); re-checking the rewrite catches that. Clues
    # still flagged incorrect after the last attempt are surfaced as soft
    # errors rather than shipped silently. Set to 0 to disable re-checking.
    fact_check_repair_attempts: int = 2


class GradingConfig(BaseModel):
    """Combined grading configuration."""

    fill: FillGradingConfig = FillGradingConfig()
    clue: ClueGradingConfig = ClueGradingConfig()


class OllamaConfig(BaseModel):
    """Ollama LLM provider settings."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout: int = 120


class OpenAIConfig(BaseModel):
    """OpenAI (Chat Completions API) LLM provider settings.

    Used as a cross-family fact-checker. ``model`` is a reasoning-capable
    default; ``reasoning_effort`` is sent only for o-series / gpt-5 models
    and ignored otherwise. ``temperature`` applies only to non-reasoning
    models (reasoning models reject it).
    """

    model: str = "gpt-5"
    reasoning_effort: str = "medium"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 120


class ClaudeConfig(BaseModel):
    """Claude (Anthropic API) LLM provider settings.

    ``model`` is the default for any step without its own override.
    Per-step fields (``theme_model``, ``fill_selection_model``, etc.)
    fall back to ``model`` when set to the empty string.
    """

    model: str = "claude-haiku-4-5-20251001"
    theme_model: str = "claude-sonnet-5"
    fill_selection_model: str = ""
    # Opus 4.8 for the quality-critical generative step (Phase 3). Adaptive
    # thinking only; the provider omits temperature for Opus 4.7/4.8.
    clue_generation_model: str = "claude-opus-4-8"
    # Repair rewrites the clues that already failed grading — the highest-
    # leverage place to spend on quality. Opus 4.8 (the first-pass generator)
    # also writes repairs; the grading-cost cuts (subset re-grade + terser
    # output) more than pay for the upgrade.
    clue_repair_model: str = "claude-opus-4-8"
    # Grading is the leak/accuracy gate — Sonnet 5 for a stronger judge (P5).
    clue_grading_model: str = "claude-sonnet-5"
    # Fact-check is the accuracy gate. Opus 4.8 is stricter about word-precision
    # (e.g. catching "Pope born in 2025" — Leo XIV was elected, not born, in
    # 2025 — which Sonnet rationalized as "safe"). At ~3% of pipeline spend the
    # upgrade adds only ~2% to total cost. See docs/clue-quality.md.
    clue_fact_check_model: str = "claude-opus-4-8"
    # Naming is a trivial creative task — Haiku is sufficient (P5). Empty falls
    # back to ``model`` (also Haiku); set explicitly for clarity.
    puzzle_naming_model: str = "claude-haiku-4-5-20251001"
    # Hints are easy beginner clues — a simpler task than clue generation, but
    # still answer-leak-sensitive. Sonnet 5 is a good quality/cost balance;
    # the leak detector screens its output. Empty falls back to ``model``.
    hint_generation_model: str = "claude-sonnet-5"
    # 8192 (not 4096): the Claude 5 family thinks by default even when we do
    # not request it, so a non-thinking call's budget must cover the model's
    # internal reasoning AND the answer, or a large grading response truncates
    # (stop_reason=max_tokens) with only a thinking block and no text — see
    # _extract_text_content / _rejects_sampling_params.
    max_tokens: int = 8192
    thinking_enabled: bool = False
    thinking_type: str = "adaptive"
    thinking_display: str = "omitted"
    thinking_max_tokens: int = 8192
    effort: str = ""
    clue_generation_thinking_enabled: bool = True
    clue_generation_effort: str = "medium"
    timeout: int = 120

    def model_for(self, step: str) -> str:
        """Return the resolved model ID for a pipeline step.

        Args:
            step: One of "theme", "fill_selection", "clue_generation",
                  "clue_repair", "clue_grading", "clue_fact_check",
                  "puzzle_naming", "hint_generation".

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
    # Per-step provider override for the clue fact-check pass. Empty string
    # inherits ``provider``. Set to "openai" to route fact-checking to a
    # different model family (cross-family checking catches confident errors
    # that a model grading its own output would pass).
    clue_fact_check_provider: str = ""
    ollama: OllamaConfig = OllamaConfig()
    claude: ClaudeConfig = ClaudeConfig()
    openai: OpenAIConfig = OpenAIConfig()
    logging: LLMLoggingConfig = LLMLoggingConfig()


class ThemeConfig(BaseModel):
    """Theme generation settings for midi puzzles."""

    enabled: bool = True
    max_retries: int = 5
    num_seed_entries: int = 3
    num_candidates: int = 12
    similarity_threshold: float = 0.6
    max_avoid_in_prompt: int = 30


class HintConfig(BaseModel):
    """Hint generation settings (an easy alternate clue per entry)."""

    enabled: bool = True
    max_retries: int = 3
    # Split hint generation into chunks of at most this many entries per LLM
    # call (0 = one call for the whole puzzle). The cacheable system prompt is
    # shared across chunks.
    generation_chunk_size: int = 0
    parallel_chunks: bool = False
    parallel_chunk_max_workers: int = 4
    # Fact-check generated hints (reusing the clue fact-checker on hint text)
    # and repair any that are flagged. Each chunk converges independently:
    # leak-screen + fact-check, repair flagged hints, re-screen, until a clean
    # sweep or the round budget is spent. A hint that cannot be made clean is
    # dropped (the entry simply ships without a hint).
    fact_check_enabled: bool = True
    # Max repair rounds per chunk before giving up on still-flagged hints.
    max_repair_rounds: int = 3


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
    hint: HintConfig = HintConfig()
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
