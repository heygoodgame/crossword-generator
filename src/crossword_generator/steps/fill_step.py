"""Grid fill pipeline steps."""

from __future__ import annotations

import itertools
import logging

from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FillError, GridFiller, GridSpec
from crossword_generator.fillers.csp import extract_slots
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.models import FillResult, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep
from crossword_generator.steps.crossing_scorer import rank_candidates
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
    giving up.

    When candidate_entries are available (surplus theme generation), selects
    the most grid-friendly subsets and tries them with graceful degradation
    from 3 seeds → 2 → 1 → revealer-only.

    The pipeline sees this as a single step.
    """

    # Maximum subsets to try per target size
    MAX_SUBSETS_PER_SIZE = 10
    # Grid variant budget per subset (decreases as we try more subsets)
    GRID_VARIANTS_PER_SUBSET = 20

    def __init__(
        self,
        filler: GridFiller,
        grader: FillGrader,
        *,
        dictionary: Dictionary | None = None,
        max_retries: int = 5,
        max_grid_variants: int = 100,
        retry_on_fail: bool = True,
    ) -> None:
        self._filler = filler
        self._grader = grader
        self._dictionary = dictionary
        self._max_retries = max_retries
        self._max_grid_variants = max_grid_variants
        self._retry_on_fail = retry_on_fail

    @property
    def name(self) -> str:
        return "grid-fill-with-grading"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Fill and grade the grid, retrying on low scores.

        When candidate_entries are present, uses subset selection with
        crossing feasibility scoring. Otherwise falls back to the original
        behavior using seed_entries directly.
        """
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"FillWithGradingStep validation failed: {'; '.join(errors)}"
            )

        has_theme = _has_theme_or_candidates(envelope)

        # If we have candidates, use subset selection
        if (
            has_theme
            and envelope.theme
            and envelope.theme.candidate_entries
            and self._dictionary is not None
        ):
            return self._run_with_subset_selection(envelope)

        # Original path: use seed_entries directly
        return self._run_direct(envelope)

    def _run_with_subset_selection(
        self, envelope: PuzzleEnvelope
    ) -> PuzzleEnvelope:
        """Fill using subset selection from candidate_entries."""
        theme = envelope.theme
        assert theme is not None

        candidates = theme.candidate_entries
        revealer = theme.revealer
        grid_size = envelope.grid_size
        assert self._dictionary is not None

        # Score candidates for crossing friendliness
        ranked = rank_candidates(
            candidates, revealer, self._dictionary, grid_size
        )
        ranked_words = [word for word, _ in ranked]
        logger.info(
            "Ranked %d candidates by crossing score: %s",
            len(ranked),
            [(w, f"{s:.2f}") for w, s in ranked],
        )

        best_result: FillResult | None = None
        best_subset: list[str] = []
        total_attempts = 0

        # Try subsets of decreasing size: 3 → 2 → 1 → 0 (revealer-only)
        max_seed_size = min(len(ranked_words), 3)
        for target_size in range(max_seed_size, -1, -1):
            subsets = _generate_subsets(
                ranked_words, target_size, self.MAX_SUBSETS_PER_SIZE
            )

            if target_size == 0:
                subsets = [[]]  # revealer-only

            logger.info(
                "Trying %d subsets of size %d", len(subsets), target_size
            )

            for subset_idx, subset in enumerate(subsets):
                # Build a temporary envelope with this subset as seed_entries
                trial_theme = theme.model_copy(
                    update={"seed_entries": list(subset)}
                )
                trial_envelope = envelope.model_copy(
                    update={"theme": trial_theme}
                )

                result, attempts = self._try_fill_with_grid_variants(
                    trial_envelope,
                    max_grid_variants=self.GRID_VARIANTS_PER_SUBSET,
                )
                total_attempts += attempts

                if result is not None:
                    if best_result is None or (result.quality_score or 0) > (
                        best_result.quality_score or 0
                    ):
                        best_result = result
                        best_subset = list(subset)

                    if (
                        result.grade_report
                        and result.grade_report.passing
                    ):
                        logger.info(
                            "Passing fill found with subset %s "
                            "(size %d, subset %d)",
                            subset,
                            target_size,
                            subset_idx,
                        )
                        # Write back the placed subset to seed_entries
                        return self._finalize(
                            envelope,
                            best_result,
                            best_subset,
                            total_attempts,
                        )

            # If we found a passing result at this size, stop degrading
            if (
                best_result
                and best_result.grade_report
                and best_result.grade_report.passing
            ):
                break

        if best_result is None:
            raise FillError(
                f"All subsets exhausted: could not fill grid after "
                f"{total_attempts} total attempt(s)"
            )

        return self._finalize(
            envelope, best_result, best_subset, total_attempts
        )

    def _finalize(
        self,
        envelope: PuzzleEnvelope,
        best_result: FillResult,
        placed_subset: list[str],
        total_attempts: int,
    ) -> PuzzleEnvelope:
        """Build the final envelope with the best result and placed subset."""
        new_errors = list(envelope.errors)
        if not best_result.grade_report or not best_result.grade_report.passing:
            new_errors.append(
                f"Fill quality below threshold after {total_attempts} "
                f"attempt(s): best score {best_result.quality_score:.1f}"
            )

        # Write placed subset back to seed_entries
        updated_theme = None
        if envelope.theme:
            updated_theme = envelope.theme.model_copy(
                update={"seed_entries": placed_subset}
            )

        return envelope.model_copy(
            update={
                "fill": best_result,
                "theme": updated_theme,
                "step_history": [*envelope.step_history, self.name],
                "errors": new_errors,
            }
        )

    def _try_fill_with_grid_variants(
        self,
        envelope: PuzzleEnvelope,
        *,
        max_grid_variants: int,
    ) -> tuple[FillResult | None, int]:
        """Try filling with multiple grid variants for a given seed set.

        Returns:
            Tuple of (best_result, total_attempts).
        """
        base_seed = envelope.metadata.get("seed")
        has_theme = _has_theme(envelope)
        effective_variants = max_grid_variants if has_theme else 1
        max_fill_attempts = self._max_retries if self._retry_on_fail else 1

        best_result: FillResult | None = None
        total_attempts = 0

        for grid_variant in range(effective_variants):
            grid_seed = (
                base_seed + grid_variant if base_seed is not None else None
            )
            spec = get_grid_spec(
                envelope.puzzle_type, envelope.grid_size, seed=grid_seed
            )

            if has_theme:
                try:
                    spec.seed_entries = _assign_theme_to_spec(envelope, spec)
                except ValueError:
                    continue

            for attempt in range(1, max_fill_attempts + 1):
                total_attempts += 1

                try:
                    filled = self._filler.fill(spec)
                except FillError:
                    if has_theme:
                        break  # try next grid pattern
                    continue  # no theme -> retry

                report = self._grader.grade(filled.grid)

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
                    return best_result, total_attempts
            else:
                continue
            continue

        return best_result, total_attempts

    def _run_direct(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Original fill path: use seed_entries directly."""
        base_seed = envelope.metadata.get("seed")
        has_theme = _has_theme(envelope)
        max_grid_variants = self._max_grid_variants if has_theme else 1
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


def _has_theme_or_candidates(envelope: PuzzleEnvelope) -> bool:
    """Check if the envelope has any theme data (seeds or candidates)."""
    return bool(
        envelope.theme
        and (
            envelope.theme.seed_entries
            or envelope.theme.revealer
            or envelope.theme.candidate_entries
        )
    )


def _generate_subsets(
    ranked_words: list[str],
    target_size: int,
    max_subsets: int,
) -> list[list[str]]:
    """Generate subsets of ranked words, ordered by aggregate rank.

    Words are already sorted by crossing score (best first), so
    combinations that include earlier words are preferred.

    Args:
        ranked_words: Words sorted by crossing score descending.
        target_size: Number of words per subset.
        max_subsets: Maximum number of subsets to return.

    Returns:
        List of word subsets, best first.
    """
    if target_size == 0:
        return [[]]
    if target_size > len(ranked_words):
        return []

    # itertools.combinations preserves input order, so subsets with
    # earlier (higher-scored) words come first naturally.
    subsets: list[list[str]] = []
    for combo in itertools.combinations(range(len(ranked_words)), target_size):
        subsets.append([ranked_words[i] for i in combo])
        if len(subsets) >= max_subsets:
            break

    return subsets
