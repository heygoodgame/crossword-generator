"""Grid fill pipeline steps."""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field

from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FillError, GridFiller, GridSpec
from crossword_generator.fillers.csp import extract_slots
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.grid_builder import build_themed_grids
from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.models import FillResult, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep
from crossword_generator.steps.crossing_scorer import rank_candidates
from crossword_generator.steps.theme_slot_assigner import assign_seed_entries_to_slots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slot-length signature data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSignature:
    """Fingerprint of a grid pattern's slot-length distribution.

    Two grid variants with the same SlotSignature accept the same
    sets of theme words, so we only need to filter candidates once
    per signature.
    """

    length_counts: tuple[tuple[int, int], ...]

    @property
    def available_lengths(self) -> frozenset[int]:
        """Distinct slot lengths present in this signature."""
        return frozenset(length for length, _ in self.length_counts)

    def can_accommodate(
        self,
        word_lengths: list[int],
        revealer_length: int,
    ) -> bool:
        """Check if a set of words + revealer can fit this signature.

        Each word (and the revealer) needs a slot of matching length.
        We must not exceed the number of available slots per length.
        """
        needed = Counter(word_lengths)
        needed[revealer_length] += 1
        counts = dict(self.length_counts)
        for length, count in needed.items():
            if counts.get(length, 0) < count:
                return False
        return True


@dataclass
class SignatureGroup:
    """A group of grid seeds that share the same slot-length signature."""

    signature: SlotSignature
    grid_seeds: list[int | None] = field(default_factory=list)


def _prescan_grid_signatures(
    puzzle_type: str,
    grid_size: int,
    base_seed: int | None,
    num_variants: int,
) -> list[SignatureGroup]:
    """Scan grid variants and group them by slot-length signature.

    Returns groups sorted by size descending (most common patterns first).
    """
    groups: dict[SlotSignature, SignatureGroup] = {}

    for variant in range(num_variants):
        grid_seed = (
            base_seed + variant if base_seed is not None else None
        )
        spec = get_grid_spec(puzzle_type, grid_size, seed=grid_seed)
        black = set(spec.black_cells)
        slots = extract_slots(spec.rows, spec.cols, black)

        counts = Counter(s.length for s in slots)
        sig = SlotSignature(
            length_counts=tuple(sorted(counts.items()))
        )

        if sig not in groups:
            groups[sig] = SignatureGroup(signature=sig)
        groups[sig].grid_seeds.append(grid_seed)

    result = sorted(
        groups.values(),
        key=lambda g: len(g.grid_seeds),
        reverse=True,
    )

    logger.info(
        "Pre-scanned %d grid variants → %d signature groups: %s",
        num_variants,
        len(result),
        [
            (dict(g.signature.length_counts), len(g.grid_seeds))
            for g in result
        ],
    )

    return result


def _generate_subsets_for_signature(
    ranked_words: list[str],
    target_size: int,
    max_subsets: int,
    signature: SlotSignature,
    revealer: str,
) -> list[list[str]]:
    """Generate subsets filtered to those the signature can accommodate.

    Like _generate_subsets but skips subsets whose word lengths
    don't fit the signature's slot distribution.
    """
    if target_size == 0:
        return [[]]
    if target_size > len(ranked_words):
        return []

    # Pre-filter candidates to those whose length exists in the signature
    available = signature.available_lengths
    compatible = [w for w in ranked_words if len(w) in available]
    if len(compatible) < target_size:
        return []

    subsets: list[list[str]] = []
    rev_len = len(revealer)
    for combo in itertools.combinations(range(len(compatible)), target_size):
        subset = [compatible[i] for i in combo]
        word_lengths = [len(w) for w in subset]
        if signature.can_accommodate(word_lengths, rev_len):
            subsets.append(subset)
            if len(subsets) >= max_subsets:
                break

    return subsets


# ---------------------------------------------------------------------------
# Theme-to-spec assignment
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fill steps
# ---------------------------------------------------------------------------


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

    When candidate_entries are available (surplus theme generation), uses
    slot-aware subset selection: pre-scans grid variants to group them
    by slot-length signature, then tries subsets that are compatible with
    each signature. This avoids wasting attempts on incompatible pairings.

    The pipeline sees this as a single step.
    """

    # Maximum subsets to try per target size
    MAX_SUBSETS_PER_SIZE = 20
    # Maximum number of signature groups to consider (top N by frequency)
    MAX_ELIGIBLE_GROUPS = 5
    # Grid variant budget per subset (decreases as we try more subsets)
    GRID_VARIANTS_PER_SUBSET = 20
    # Number of themed grid variants to build per subset in theme-first mode
    THEME_FIRST_GRID_COUNT = 10

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

        # Auto-allow theme words in dictionary at score 60
        if envelope.theme and self._dictionary is not None:
            theme_words: dict[str, int] = {}
            for w in envelope.theme.seed_entries + envelope.theme.candidate_entries:
                theme_words[w] = 60
            if envelope.theme.revealer:
                theme_words[envelope.theme.revealer] = 60
            self._dictionary.add_words(theme_words)

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
        """Fill using theme-first construction, then random-grid fallback."""
        theme = envelope.theme
        assert theme is not None

        candidates = theme.candidate_entries
        revealer = theme.revealer
        grid_size = envelope.grid_size
        assert self._dictionary is not None

        # 1. Rank candidates once globally by crossing friendliness
        ranked = rank_candidates(
            candidates, revealer, self._dictionary, grid_size
        )
        ranked_words = [word for word, _ in ranked]
        logger.info(
            "Ranked %d candidates by crossing score: %s",
            len(ranked),
            [(w, f"{s:.2f}") for w, s in ranked],
        )

        # Phase 1: Theme-first construction (primary strategy)
        result, subset, attempts = self._try_theme_first_fill(
            envelope, ranked_words, revealer
        )
        if result and result.grade_report and result.grade_report.passing:
            logger.info(
                "Theme-first strategy succeeded with subset %s "
                "(%d attempts)",
                subset, attempts,
            )
            return self._finalize(envelope, result, subset, attempts)

        best_result = result
        best_subset = subset
        total_attempts = attempts

        if best_result:
            logger.info(
                "Theme-first best score: %.1f (not passing), "
                "falling back to random-grid search",
                best_result.quality_score or 0,
            )
        else:
            logger.info(
                "Theme-first produced no results, falling back to "
                "random-grid search"
            )

        # Phase 2: Random-grid search (fallback — existing signature-based)
        fallback_result, fallback_subset, fallback_attempts = (
            self._try_random_grid_fill(
                envelope, ranked_words, revealer
            )
        )
        total_attempts += fallback_attempts

        if fallback_result is not None:
            if best_result is None or (
                fallback_result.quality_score or 0
            ) > (best_result.quality_score or 0):
                best_result = fallback_result
                best_subset = fallback_subset

        if best_result is None:
            raise FillError(
                f"All strategies exhausted: could not fill grid after "
                f"{total_attempts} total attempt(s)"
            )

        return self._finalize(
            envelope, best_result, best_subset, total_attempts
        )

    def _try_theme_first_fill(
        self,
        envelope: PuzzleEnvelope,
        ranked_words: list[str],
        revealer: str,
    ) -> tuple[FillResult | None, list[str], int]:
        """Try building grids around theme entries (theme-first strategy).

        For each subset of candidates (size 3 → 2 → 1):
          - build_themed_grids() → multiple GridSpecs with entries pre-placed
          - CSP fill each
          - Grade and keep best

        Returns (best_result, placed_subset, total_attempts).
        """
        theme = envelope.theme
        assert theme is not None
        grid_size = envelope.grid_size
        base_seed = envelope.metadata.get("seed", 0) or 0

        best_result: FillResult | None = None
        best_subset: list[str] = []
        total_attempts = 0

        max_seed_size = min(len(ranked_words), 4)

        for target_size in range(max_seed_size, 0, -1):
            subsets = _generate_subsets(
                ranked_words, target_size, self.MAX_SUBSETS_PER_SIZE
            )

            for subset in subsets:
                # Build themed grids for this subset
                specs = build_themed_grids(
                    grid_size,
                    list(subset),
                    revealer,
                    seed=base_seed,
                    count=self.THEME_FIRST_GRID_COUNT,
                )

                if not specs:
                    logger.debug(
                        "Theme-first: no valid grids for subset %s",
                        subset,
                    )
                    continue

                for spec in specs:
                    total_attempts += 1

                    try:
                        filled = self._filler.fill(spec)
                    except FillError:
                        continue

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
                        best_subset = list(subset)

                    if report.passing:
                        logger.info(
                            "Theme-first passing fill: subset %s, "
                            "score %.1f",
                            subset, report.overall_score,
                        )
                        return best_result, best_subset, total_attempts

        return best_result, best_subset, total_attempts

    def _try_random_grid_fill(
        self,
        envelope: PuzzleEnvelope,
        ranked_words: list[str],
        revealer: str,
    ) -> tuple[FillResult | None, list[str], int]:
        """Try filling using random grid patterns (signature-based).

        This is the original slot-aware subset selection strategy.
        Returns (best_result, placed_subset, total_attempts).
        """
        theme = envelope.theme
        assert theme is not None
        grid_size = envelope.grid_size

        # Pre-scan grid variants → signature groups
        base_seed = envelope.metadata.get("seed")
        sig_groups = _prescan_grid_signatures(
            envelope.puzzle_type,
            grid_size,
            base_seed,
            self._max_grid_variants,
        )

        best_result: FillResult | None = None
        best_subset: list[str] = []
        total_attempts = 0

        max_seed_size = min(len(ranked_words), 4)
        for target_size in range(max_seed_size, -1, -1):
            subsets_tried = 0

            eligible_groups = [
                g for g in sig_groups
                if len(revealer) in g.signature.available_lengths
            ]
            if len(eligible_groups) > self.MAX_ELIGIBLE_GROUPS:
                logger.info(
                    "Capping eligible groups from %d to %d",
                    len(eligible_groups),
                    self.MAX_ELIGIBLE_GROUPS,
                )
                eligible_groups = eligible_groups[:self.MAX_ELIGIBLE_GROUPS]
            num_eligible = len(eligible_groups)
            if num_eligible == 0:
                continue
            per_group_budget = max(
                1, self.MAX_SUBSETS_PER_SIZE // num_eligible
            )

            for group in eligible_groups:
                sig = group.signature
                group_tried = 0

                if target_size == 0:
                    subsets = [[]]
                else:
                    subsets = _generate_subsets_for_signature(
                        ranked_words,
                        target_size,
                        per_group_budget,
                        sig,
                        revealer,
                    )

                logger.info(
                    "Random-grid: target size %d, signature %s: %d subsets "
                    "(budget %d), %d grid seeds",
                    target_size,
                    dict(sig.length_counts),
                    len(subsets),
                    per_group_budget,
                    len(group.grid_seeds),
                )

                for subset in subsets:
                    if (
                        group_tried >= per_group_budget
                        or subsets_tried >= self.MAX_SUBSETS_PER_SIZE
                    ):
                        break

                    trial_theme = theme.model_copy(
                        update={"seed_entries": list(subset)}
                    )
                    trial_envelope = envelope.model_copy(
                        update={"theme": trial_theme}
                    )

                    result, attempts = self._try_fill_with_grid_seeds(
                        trial_envelope,
                        group.grid_seeds,
                        max_seeds=self.GRID_VARIANTS_PER_SUBSET,
                    )
                    total_attempts += attempts
                    subsets_tried += 1
                    group_tried += 1

                    if result is not None:
                        if best_result is None or (
                            result.quality_score or 0
                        ) > (best_result.quality_score or 0):
                            best_result = result
                            best_subset = list(subset)

                        if (
                            result.grade_report
                            and result.grade_report.passing
                        ):
                            logger.info(
                                "Random-grid passing fill with subset %s "
                                "(size %d)",
                                subset,
                                target_size,
                            )
                            return best_result, best_subset, total_attempts

                if subsets_tried >= self.MAX_SUBSETS_PER_SIZE:
                    break

            if (
                best_result
                and best_result.grade_report
                and best_result.grade_report.passing
            ):
                break

        return best_result, best_subset, total_attempts

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

    def _try_fill_with_grid_seeds(
        self,
        envelope: PuzzleEnvelope,
        grid_seeds: list[int | None],
        *,
        max_seeds: int,
    ) -> tuple[FillResult | None, int]:
        """Try filling with an explicit list of grid seeds.

        Args:
            envelope: Envelope with theme seed_entries already set.
            grid_seeds: Pre-computed grid seeds to try.
            max_seeds: Maximum number of seeds to try from the list.

        Returns:
            Tuple of (best_result, total_attempts).
        """
        has_theme = _has_theme(envelope)
        max_fill_attempts = self._max_retries if self._retry_on_fail else 1

        best_result: FillResult | None = None
        total_attempts = 0

        for grid_seed in grid_seeds[:max_seeds]:
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
                        break  # try next grid seed
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
    """Generate subsets of ranked words, ordered by total entry length.

    Shorter total length means fewer constrained cells in the grid,
    giving the CSP filler more freedom to produce quality fills.
    Among equal-length subsets, prefer those with higher crossing
    scores (lower rank-index sum).

    Args:
        ranked_words: Words sorted by crossing score descending.
        target_size: Number of words per subset.
        max_subsets: Maximum number of subsets to return.

    Returns:
        List of word subsets, shortest total length first.
    """
    if target_size == 0:
        return [[]]
    if target_size > len(ranked_words):
        return []

    # Build all subsets, then sort by total entry length (ascending)
    # with sum of rank indices as tiebreaker (lower = higher-ranked).
    # For n≤12, k≤3 this produces at most 220 items — trivial cost.
    all_combos: list[tuple[int, int, list[str]]] = []
    for combo in itertools.combinations(range(len(ranked_words)), target_size):
        words = [ranked_words[i] for i in combo]
        total_len = sum(len(w) for w in words)
        rank_sum = sum(combo)
        all_combos.append((total_len, rank_sum, words))

    all_combos.sort()
    return [words for _, _, words in all_combos[:max_subsets]]
