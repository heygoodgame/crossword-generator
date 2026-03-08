"""Grid fill pipeline steps."""

from __future__ import annotations

import logging

from crossword_generator.fillers.base import GridFiller
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.models import FillResult, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class FillStep(PipelineStep):
    """Pipeline step that fills an empty grid using a GridFiller backend."""

    def __init__(self, filler: GridFiller) -> None:
        self._filler = filler

    @property
    def name(self) -> str:
        return "grid-fill"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Fill the grid and return an updated envelope."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(f"FillStep validation failed: {'; '.join(errors)}")

        seed = envelope.metadata.get("seed")
        spec = get_grid_spec(envelope.puzzle_type, envelope.grid_size, seed=seed)

        logger.info(
            "Running grid fill with %s (%dx%d, %d black cells)",
            self._filler.name,
            spec.rows,
            spec.cols,
            len(spec.black_cells),
        )

        filled = self._filler.fill(spec)

        fill_result = FillResult(
            grid=filled.grid,
            filler_used=self._filler.name,
        )

        return envelope.model_copy(
            update={
                "fill": fill_result,
                "step_history": [*envelope.step_history, self.name],
            }
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        """Validate that the envelope is ready for filling."""
        errors: list[str] = []
        if envelope.fill is not None:
            errors.append("Envelope already has a fill result")
        if not self._filler.is_available():
            errors.append(f"Filler '{self._filler.name}' is not available")
        return errors


class FillWithGradingStep(PipelineStep):
    """Composite step: fill grid, grade quality, retry if below threshold.

    Wraps a GridFiller and FillGrader, looping up to max_retries attempts.
    The pipeline sees this as a single step.
    """

    def __init__(
        self,
        filler: GridFiller,
        grader: FillGrader,
        *,
        max_retries: int = 5,
        retry_on_fail: bool = True,
    ) -> None:
        self._filler = filler
        self._grader = grader
        self._max_retries = max_retries
        self._retry_on_fail = retry_on_fail

    @property
    def name(self) -> str:
        return "grid-fill-with-grading"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Fill and grade the grid, retrying on low scores."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"FillWithGradingStep validation failed: {'; '.join(errors)}"
            )

        seed = envelope.metadata.get("seed")
        spec = get_grid_spec(envelope.puzzle_type, envelope.grid_size, seed=seed)
        max_attempts = self._max_retries if self._retry_on_fail else 1

        best_result: FillResult | None = None

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Fill attempt %d/%d with %s (%dx%d)",
                attempt,
                max_attempts,
                self._filler.name,
                spec.rows,
                spec.cols,
            )

            filled = self._filler.fill(spec)
            report = self._grader.grade(filled.grid)

            logger.info(
                "Attempt %d score: %.1f/100 (%s)",
                attempt,
                report.overall_score,
                "PASS" if report.passing else "FAIL",
            )

            result = FillResult(
                grid=filled.grid,
                filler_used=self._filler.name,
                quality_score=report.overall_score,
                grade_report=report,
                attempt_number=attempt,
            )

            if best_result is None or report.overall_score > (
                best_result.quality_score or 0.0
            ):
                best_result = result

            if report.passing:
                break

        assert best_result is not None  # At least one attempt always runs

        new_errors = list(envelope.errors)
        if not best_result.grade_report or not best_result.grade_report.passing:
            new_errors.append(
                f"Fill quality below threshold after {max_attempts} attempt(s): "
                f"best score {best_result.quality_score:.1f}"
            )

        return envelope.model_copy(
            update={
                "fill": best_result,
                "step_history": [*envelope.step_history, self.name],
                "errors": new_errors,
            }
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        """Validate that the envelope is ready for filling."""
        errors: list[str] = []
        if envelope.fill is not None:
            errors.append("Envelope already has a fill result")
        if not self._filler.is_available():
            errors.append(f"Filler '{self._filler.name}' is not available")
        return errors
