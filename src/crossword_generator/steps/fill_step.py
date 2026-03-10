"""Grid fill pipeline steps."""

from __future__ import annotations

import logging

from crossword_generator.fillers.base import FillError, GridFiller, GridSpec
from crossword_generator.fillers.csp import extract_slots
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.models import FillResult, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep
from crossword_generator.steps.theme_slot_assigner import assign_seed_entries_to_slots

logger = logging.getLogger(__name__)


def _assign_theme_to_spec(
    envelope: PuzzleEnvelope, spec: GridSpec
) -> dict[str, str]:
    """Assign theme seed entries and revealer to grid slots.

    Args:
        envelope: The puzzle envelope with theme data.
        spec: The grid specification to assign slots from.

    Returns:
        A dict mapping "row,col,direction" to uppercase word.

    Raises:
        ValueError: If any theme word has no matching available slot.
    """
    black = set(spec.black_cells)
    slots = extract_slots(spec.rows, spec.cols, black)
    assignments = assign_seed_entries_to_slots(
        envelope.theme.seed_entries,
        envelope.theme.revealer,
        slots,
    )
    seed_entries = {
        f"{a.row},{a.col},{a.direction}": a.word.upper()
        for a in assignments
    }
    logger.info("Assigned %d theme entries to grid slots", len(assignments))
    return seed_entries


def _has_theme(envelope: PuzzleEnvelope) -> bool:
    """Check if the envelope has theme data that needs slot assignment."""
    return bool(
        envelope.theme
        and (envelope.theme.seed_entries or envelope.theme.revealer)
    )


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

        # Propagate theme seed entries as fill constraints
        if _has_theme(envelope):
            spec.seed_entries = _assign_theme_to_spec(envelope, spec)

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
    When theme seed entries are present and the filler raises FillError
    (e.g. AC-3 infeasibility), tries alternative grid patterns before
    giving up. The pipeline sees this as a single step.
    """

    _MAX_GRID_VARIANTS = 5

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
        """Fill and grade the grid, retrying on low scores.

        For themed puzzles, tries multiple grid patterns if the filler
        raises FillError (seed entry infeasibility). For each viable
        grid pattern, retries fills up to max_retries for quality.
        """
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"FillWithGradingStep validation failed: {'; '.join(errors)}"
            )

        base_seed = envelope.metadata.get("seed")
        has_theme = _has_theme(envelope)
        max_grid_variants = self._MAX_GRID_VARIANTS if has_theme else 1
        max_fill_attempts = self._max_retries if self._retry_on_fail else 1

        best_result: FillResult | None = None
        total_attempts = 0

        for grid_variant in range(max_grid_variants):
            grid_seed = (
                base_seed + grid_variant if base_seed is not None else None
            )
            spec = get_grid_spec(
                envelope.puzzle_type, envelope.grid_size, seed=grid_seed
            )

            # Assign seed entries to this grid's slots
            if has_theme:
                try:
                    spec.seed_entries = _assign_theme_to_spec(envelope, spec)
                except ValueError:
                    logger.info(
                        "Grid variant %d: no matching slots for theme entries, "
                        "skipping",
                        grid_variant,
                    )
                    continue  # no matching slots in this grid pattern

            if grid_variant > 0:
                logger.info(
                    "Trying grid variant %d (seed=%s)",
                    grid_variant,
                    grid_seed,
                )

            for attempt in range(1, max_fill_attempts + 1):
                total_attempts += 1
                logger.info(
                    "Fill attempt %d/%d (grid variant %d) with %s (%dx%d)",
                    attempt,
                    max_fill_attempts,
                    grid_variant,
                    self._filler.name,
                    spec.rows,
                    spec.cols,
                )

                try:
                    filled = self._filler.fill(spec)
                except FillError:
                    if has_theme:
                        logger.warning(
                            "Grid variant %d: fill infeasible with theme "
                            "entries, trying next pattern",
                            grid_variant,
                        )
                        break  # try next grid pattern
                    logger.warning(
                        "Fill attempt %d failed with FillError, retrying",
                        attempt,
                    )
                    continue  # no theme -> just bad luck, retry

                report = self._grader.grade(filled.grid)

                logger.info(
                    "Attempt %d score: %.1f/100 (%s)",
                    total_attempts,
                    report.overall_score,
                    "PASS" if report.passing else "FAIL",
                )

                result = FillResult(
                    grid=filled.grid,
                    filler_used=self._filler.name,
                    quality_score=report.overall_score,
                    grade_report=report,
                    attempt_number=total_attempts,
                )

                if best_result is None or report.overall_score > (
                    best_result.quality_score or 0.0
                ):
                    best_result = result

                if report.passing:
                    break
            else:
                # All fill attempts exhausted for this grid variant;
                # continue to next variant if available
                continue

            # If we broke out of the fill loop with a passing result, stop
            if (
                best_result
                and best_result.grade_report
                and best_result.grade_report.passing
            ):
                break

        if best_result is None:
            raise FillError(
                f"All grid variants exhausted: could not fill grid after "
                f"trying {max_grid_variants} pattern(s) with "
                f"{total_attempts} total attempt(s)"
            )

        new_errors = list(envelope.errors)
        if not best_result.grade_report or not best_result.grade_report.passing:
            new_errors.append(
                f"Fill quality below threshold after {total_attempts} "
                f"attempt(s): best score {best_result.quality_score:.1f}"
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
