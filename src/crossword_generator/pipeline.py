"""Pipeline orchestration for crossword generation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from crossword_generator.config import Config, find_project_root
from crossword_generator.dictionary import Dictionary
from crossword_generator.exporters.base import Exporter
from crossword_generator.exporters.ipuz_exporter import IpuzExporter
from crossword_generator.exporters.puz_exporter import PuzExporter
from crossword_generator.fillers.csp import CSPFiller
from crossword_generator.fillers.go_crossword import GoCrosswordFiller
from crossword_generator.graders.clue_grader import ClueGrader
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.llm.claude_provider import ClaudeProvider
from crossword_generator.llm.ollama_provider import OllamaProvider
from crossword_generator.models import PuzzleEnvelope, PuzzleType
from crossword_generator.steps.base import PipelineStep
from crossword_generator.steps.clue_grading_step import ClueWithGradingStep
from crossword_generator.steps.fill_step import FillWithGradingStep
from crossword_generator.steps.theme_step import ThemeGenerationStep

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs a sequence of pipeline steps and exports the result."""

    def __init__(
        self,
        steps: list[PipelineStep],
        exporters: list[Exporter],
        output_dir: Path,
    ) -> None:
        self._steps = steps
        self._exporters = exporters
        self._output_dir = output_dir

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Execute all steps sequentially, saving intermediates and exporting."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        for step in self._steps:
            logger.info("Running step: %s", step.name)
            envelope = step.run(envelope)
            self._save_intermediate(envelope, step.name)

        # Export
        exported: list[Path] = []
        for exporter in self._exporters:
            try:
                path = exporter.export(envelope, self._output_dir)
                exported.append(path)
            except Exception:
                logger.exception("Export failed for format %s", exporter.file_extension)

        if exported:
            logger.info("Exported files: %s", [str(p) for p in exported])
        else:
            logger.warning("No files were exported")

        return envelope

    def _save_intermediate(self, envelope: PuzzleEnvelope, step_name: str) -> None:
        """Save an intermediate envelope JSON for debugging/resumption."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"intermediate_{step_name}_{timestamp}.json"
        filepath = self._output_dir / filename
        filepath.write_text(envelope.model_dump_json(indent=2))
        logger.debug("Saved intermediate: %s", filepath)


def create_pipeline(
    config: Config,
    *,
    seed: int | None = None,
    theme_file: Path | None = None,
) -> tuple[Pipeline, PuzzleEnvelope]:
    """Wire up a Pipeline and initial PuzzleEnvelope from config.

    Args:
        config: Loaded configuration.
        seed: Optional random seed for the filler.
        theme_file: Optional path to a pre-generated theme file.
            When provided, the theme is loaded and injected into the
            envelope, and the ThemeGenerationStep is skipped.

    Returns:
        Tuple of (Pipeline, initial PuzzleEnvelope).
    """
    # Build fill grader dictionary (shared with CSP filler if selected)
    project_root = find_project_root()
    dictionary = Dictionary.load(
        project_root / config.dictionary.path,
        min_word_score=config.dictionary.min_word_score,
        min_2letter_score=config.dictionary.min_2letter_score,
    )

    # Build filler
    if config.fill.provider == "go-crossword":
        filler = GoCrosswordFiller(config.fill.go_crossword)
    elif config.fill.provider == "csp":
        filler = CSPFiller(config.fill.csp, dictionary)
    else:
        raise ValueError(f"Unknown fill provider: {config.fill.provider}")
    grader = FillGrader(
        dictionary,
        min_passing_score=config.grading.fill.min_score,
    )
    fill_step = FillWithGradingStep(
        filler,
        grader,
        dictionary=dictionary,
        max_retries=config.fill.max_retries,
        max_grid_variants=config.fill.max_grid_variants,
        retry_on_fail=config.grading.fill.retry_on_fail,
    )

    # Build LLM provider
    if config.llm.provider == "ollama":
        llm_provider = OllamaProvider(config.llm.ollama)
    elif config.llm.provider == "claude":
        llm_provider = ClaudeProvider(config.llm.claude)
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm.provider}")

    clue_grader = ClueGrader(
        llm_provider, min_passing_score=config.grading.clue.min_score
    )
    clue_step = ClueWithGradingStep(
        llm_provider,
        clue_grader,
        max_retries=3,
        regenerate_on_fail=config.grading.clue.regenerate_on_fail,
    )

    # Build steps
    steps: list[PipelineStep] = []

    # Load pre-generated theme or add theme generation step
    pre_loaded_theme = None
    if theme_file is not None:
        from crossword_generator.theme_io import load_theme

        theme_data = load_theme(theme_file)
        pre_loaded_theme = theme_data.theme
        logger.info(
            "Using pre-generated theme from %s: %r",
            theme_file,
            pre_loaded_theme.topic,
        )
    elif PuzzleType(config.puzzle.type) == PuzzleType.MIDI and config.theme.enabled:
        theme_step = ThemeGenerationStep(
            llm_provider,
            dictionary,
            grid_size=config.puzzle.grid_size,
            max_retries=config.theme.max_retries,
            num_seed_entries=config.theme.num_seed_entries,
            num_candidates=config.theme.num_candidates,
        )
        steps.append(theme_step)

    steps.extend([fill_step, clue_step])

    # Build exporters
    exporters: list[Exporter] = []
    for fmt in config.output.formats:
        if fmt == "puz":
            exporters.append(PuzExporter())
        elif fmt == "ipuz":
            exporters.append(IpuzExporter())
        else:
            logger.warning("Unknown export format: %s", fmt)

    output_dir = Path(config.output.directory)

    pipeline = Pipeline(steps=steps, exporters=exporters, output_dir=output_dir)

    envelope = PuzzleEnvelope(
        puzzle_type=PuzzleType(config.puzzle.type),
        grid_size=config.puzzle.grid_size,
        theme=pre_loaded_theme,
        metadata={"seed": seed} if seed is not None else {},
    )

    return pipeline, envelope
