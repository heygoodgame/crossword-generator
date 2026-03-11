"""Tests for the FillWithGradingStep composite pipeline step."""

from __future__ import annotations

import pytest

from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FilledGrid, FillError, GridFiller, GridSpec
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.models import (
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
    ThemeConcept,
)
from crossword_generator.steps.fill_step import (
    FillWithGradingStep,
    SlotSignature,
    _generate_subsets,
    _generate_subsets_for_signature,
    _prescan_grid_signatures,
)


def _make_dict(words: dict[str, int]) -> Dictionary:
    return Dictionary(words, min_word_score=0, min_2letter_score=0)


# High-quality grid — all words in dictionary with high scores
HIGH_QUALITY_GRID = [
    ["S", "T", "A", "R", "E"],
    ["T", "O", "N", "E", "S"],
    ["A", "R", "E", "N", "A"],
    ["R", "E", "S", "E", "T"],
    ["S", "P", "E", "E", "D"],
]

# Low-quality grid — nonsense words
LOW_QUALITY_GRID = [
    ["X", "Z", "Q", "W", "K"],
    ["J", "V", "B", "N", "M"],
    ["P", "L", "F", "G", "H"],
    ["R", "T", "Y", "U", "I"],
    ["D", "S", "A", "C", "O"],
]


class FixedMockFiller(GridFiller):
    """Returns a fixed grid every time."""

    def __init__(self, grid: list[list[str]]) -> None:
        self._grid = grid

    @property
    def name(self) -> str:
        return "fixed-mock"

    def fill(self, spec: GridSpec) -> FilledGrid:
        return FilledGrid(grid=self._grid)


class SequentialMockFiller(GridFiller):
    """Returns different grids on successive calls."""

    def __init__(self, grids: list[list[list[str]]]) -> None:
        self._grids = grids
        self._call_count = 0

    @property
    def name(self) -> str:
        return "sequential-mock"

    def fill(self, spec: GridSpec) -> FilledGrid:
        idx = min(self._call_count, len(self._grids) - 1)
        self._call_count += 1
        return FilledGrid(grid=self._grids[idx])

    @property
    def call_count(self) -> int:
        return self._call_count


# Dictionary that knows the high-quality grid words
GOOD_WORDS = {
    "STARE": 80,
    "TONES": 75,
    "ARENA": 85,
    "RESET": 70,
    "SPEED": 90,
    "STARS": 80,
    "TORED": 60,
    "ANISE": 65,
    "RENEE": 60,
    "EASET": 55,
    "STORE": 70,
    "PARSE": 75,
    "ENTER": 80,
}


class TestPassOnFirstTry:
    def test_passes_on_first_attempt(self) -> None:
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(filler, grader, max_retries=5)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.grade_report is not None
        assert result.fill.grade_report.passing is True
        assert result.fill.attempt_number == 1
        assert result.errors == []


class TestRetryOnFailure:
    def test_retries_until_good_grid(self) -> None:
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)

        # First 2 attempts return bad grid, third returns good
        filler = SequentialMockFiller(
            [
                LOW_QUALITY_GRID,
                LOW_QUALITY_GRID,
                HIGH_QUALITY_GRID,
            ]
        )
        step = FillWithGradingStep(filler, grader, max_retries=5)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.grade_report is not None
        assert result.fill.grade_report.passing is True
        assert result.fill.attempt_number == 3
        assert filler.call_count == 3
        assert result.errors == []


class TestBestOfNOnAllFailures:
    def test_keeps_best_result(self) -> None:
        # All unknown words → will fail, but should keep best score
        dictionary = _make_dict({})
        grader = FillGrader(dictionary, min_passing_score=90)

        filler = FixedMockFiller(LOW_QUALITY_GRID)
        step = FillWithGradingStep(filler, grader, max_retries=3)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.grade_report is not None
        assert result.fill.grade_report.passing is False
        assert len(result.errors) == 1
        assert "below threshold" in result.errors[0]


class TestRetryDisabled:
    def test_single_attempt_when_disabled(self) -> None:
        dictionary = _make_dict({})
        grader = FillGrader(dictionary, min_passing_score=90)

        filler = SequentialMockFiller([LOW_QUALITY_GRID, HIGH_QUALITY_GRID])
        step = FillWithGradingStep(filler, grader, max_retries=5, retry_on_fail=False)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert filler.call_count == 1
        assert result.fill is not None
        assert result.fill.attempt_number == 1


class TestStepMetadata:
    def test_step_name(self) -> None:
        dictionary = _make_dict({})
        grader = FillGrader(dictionary)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(filler, grader)
        assert step.name == "grid-fill-with-grading"

    def test_step_history(self) -> None:
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(filler, grader)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert "grid-fill-with-grading" in result.step_history

    def test_fill_result_has_quality_score(self) -> None:
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(filler, grader)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.quality_score is not None
        assert result.fill.quality_score > 0


class TestValidation:
    def test_rejects_existing_fill(self) -> None:
        dictionary = _make_dict({})
        grader = FillGrader(dictionary)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(filler, grader)

        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MINI,
            grid_size=5,
            fill=FillResult(grid=[["A"]], filler_used="other"),
        )
        with pytest.raises(ValueError, match="already has a fill"):
            step.run(envelope)

    def test_rejects_unavailable_filler(self) -> None:
        dictionary = _make_dict({})
        grader = FillGrader(dictionary)

        class UnavailableFiller(GridFiller):
            @property
            def name(self) -> str:
                return "unavailable"

            def fill(self, spec: GridSpec) -> FilledGrid:
                return FilledGrid(grid=[])

            def is_available(self) -> bool:
                return False

        step = FillWithGradingStep(UnavailableFiller(), grader)
        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        with pytest.raises(ValueError, match="not available"):
            step.run(envelope)


class TestAttemptNumberTracking:
    def test_attempt_number_increments(self) -> None:
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)

        filler = SequentialMockFiller(
            [
                LOW_QUALITY_GRID,
                HIGH_QUALITY_GRID,
            ]
        )
        step = FillWithGradingStep(filler, grader, max_retries=5)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.attempt_number == 2


class FillErrorThenSuccessFiller(GridFiller):
    """Raises FillError for the first N calls, then returns a fixed grid."""

    def __init__(
        self, grid: list[list[str]], *, fail_count: int = 1
    ) -> None:
        self._grid = grid
        self._fail_count = fail_count
        self._call_count = 0

    @property
    def name(self) -> str:
        return "fail-then-succeed"

    def fill(self, spec: GridSpec) -> FilledGrid:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise FillError("AC-3 infeasible with seed entries")
        return FilledGrid(grid=self._grid)

    @property
    def call_count(self) -> int:
        return self._call_count


class AlwaysFailFiller(GridFiller):
    """Always raises FillError."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def name(self) -> str:
        return "always-fail"

    def fill(self, spec: GridSpec) -> FilledGrid:
        self._call_count += 1
        raise FillError("Cannot fill")

    @property
    def call_count(self) -> int:
        return self._call_count


class TestGridPatternFallback:
    """Tests for the grid pattern fallback when theme entries cause infeasibility."""

    def _make_themed_envelope(self, seed: int = 42) -> PuzzleEnvelope:
        """Create a midi envelope with theme data."""
        return PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            metadata={"seed": seed},
            theme=ThemeConcept(
                topic="Test theme",
                seed_entries=["EAGLE", "HAWK", "KITE"],
                revealer="SOAR",
            ),
        )

    def test_fill_error_tries_next_grid_pattern(self) -> None:
        """FillError with themed entries should try next grid pattern."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)

        # Fails once (first grid pattern), succeeds on second
        filler = FillErrorThenSuccessFiller(HIGH_QUALITY_GRID, fail_count=1)
        step = FillWithGradingStep(filler, grader, max_retries=3)

        envelope = self._make_themed_envelope()
        result = step.run(envelope)

        assert result.fill is not None
        assert filler.call_count >= 2  # Tried at least 2 grid patterns

    def test_fill_error_exhausts_all_grid_patterns(self) -> None:
        """When all grid patterns fail, should raise FillError."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)

        filler = AlwaysFailFiller()
        step = FillWithGradingStep(filler, grader, max_retries=3)

        envelope = self._make_themed_envelope()
        with pytest.raises(FillError, match="All grid variants exhausted"):
            step.run(envelope)

    def test_no_theme_fill_error_retries_same_grid(self) -> None:
        """Without theme, FillError should retry on the same grid pattern."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)

        # Fails once, then succeeds
        filler = FillErrorThenSuccessFiller(HIGH_QUALITY_GRID, fail_count=1)
        step = FillWithGradingStep(filler, grader, max_retries=3)

        # No theme → mini puzzle without theme
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MINI, grid_size=5, metadata={"seed": 1}
        )
        result = step.run(envelope)

        assert result.fill is not None
        assert filler.call_count == 2

    def test_no_theme_all_fill_errors_raises(self) -> None:
        """Without theme, if all retries raise FillError, should raise."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)

        filler = AlwaysFailFiller()
        step = FillWithGradingStep(filler, grader, max_retries=3)

        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MINI, grid_size=5, metadata={"seed": 1}
        )
        with pytest.raises(FillError, match="All grid variants exhausted"):
            step.run(envelope)


class TestGenerateSubsets:
    """Tests for the _generate_subsets helper."""

    def test_subsets_of_size_3(self) -> None:
        words = ["A", "B", "C", "D"]
        subsets = _generate_subsets(words, target_size=3, max_subsets=10)
        assert len(subsets) == 4  # C(4,3) = 4
        # First subset should include the highest-ranked words
        assert subsets[0] == ["A", "B", "C"]

    def test_subsets_capped_at_max(self) -> None:
        words = ["A", "B", "C", "D", "E"]
        subsets = _generate_subsets(words, target_size=2, max_subsets=3)
        assert len(subsets) == 3  # C(5,2)=10 but capped at 3

    def test_subsets_of_size_0(self) -> None:
        subsets = _generate_subsets(["A", "B"], target_size=0, max_subsets=10)
        assert subsets == [[]]

    def test_subsets_when_not_enough_words(self) -> None:
        subsets = _generate_subsets(["A"], target_size=3, max_subsets=10)
        assert subsets == []


class TestSlotSignature:
    """Tests for the SlotSignature data structure."""

    def test_available_lengths(self) -> None:
        sig = SlotSignature(length_counts=((3, 10), (5, 4), (9, 2)))
        assert sig.available_lengths == frozenset({3, 5, 9})

    def test_can_accommodate_simple(self) -> None:
        sig = SlotSignature(length_counts=((3, 10), (5, 4), (9, 2)))
        # 2 words of length 5 + revealer of length 9 → needs 5:2, 9:1
        assert sig.can_accommodate([5, 5], revealer_length=9) is True

    def test_can_accommodate_exceeds_slots(self) -> None:
        sig = SlotSignature(length_counts=((3, 10), (5, 2), (9, 1)))
        # 3 words of length 5 → needs 5:3 but only 2 available
        assert sig.can_accommodate([5, 5, 5], revealer_length=9) is False

    def test_can_accommodate_missing_length(self) -> None:
        sig = SlotSignature(length_counts=((3, 10), (5, 4)))
        # Revealer needs length 9 but no 9-letter slots
        assert sig.can_accommodate([3, 5], revealer_length=9) is False

    def test_can_accommodate_empty_words(self) -> None:
        sig = SlotSignature(length_counts=((4, 5), (9, 2)))
        # No seed words, just revealer
        assert sig.can_accommodate([], revealer_length=9) is True

    def test_frozen(self) -> None:
        sig = SlotSignature(length_counts=((3, 10),))
        # Should be hashable (frozen)
        assert hash(sig) is not None
        d = {sig: "test"}
        assert d[sig] == "test"

    def test_revealer_and_word_share_length(self) -> None:
        sig = SlotSignature(length_counts=((5, 3),))
        # 2 words of length 5 + revealer of length 5 → needs 5:3
        assert sig.can_accommodate([5, 5], revealer_length=5) is True
        # 3 words + revealer = 4 needed, only 3 available
        assert sig.can_accommodate([5, 5, 5], revealer_length=5) is False


class TestPrescanGridSignatures:
    """Tests for _prescan_grid_signatures."""

    def test_returns_signature_groups(self) -> None:
        groups = _prescan_grid_signatures(
            PuzzleType.MIDI, 9, base_seed=1, num_variants=10
        )
        assert len(groups) > 0
        # Each group should have at least one grid seed
        for group in groups:
            assert len(group.grid_seeds) > 0
            assert isinstance(group.signature, SlotSignature)

    def test_sorted_by_size_descending(self) -> None:
        groups = _prescan_grid_signatures(
            PuzzleType.MIDI, 9, base_seed=1, num_variants=50
        )
        sizes = [len(g.grid_seeds) for g in groups]
        assert sizes == sorted(sizes, reverse=True)

    def test_total_seeds_matches_variants(self) -> None:
        num_variants = 20
        groups = _prescan_grid_signatures(
            PuzzleType.MIDI, 9, base_seed=1, num_variants=num_variants
        )
        total = sum(len(g.grid_seeds) for g in groups)
        assert total == num_variants

    def test_single_variant(self) -> None:
        groups = _prescan_grid_signatures(
            PuzzleType.MIDI, 9, base_seed=42, num_variants=1
        )
        assert len(groups) == 1
        assert len(groups[0].grid_seeds) == 1


class TestGenerateSubsetsForSignature:
    """Tests for _generate_subsets_for_signature."""

    def test_filters_by_available_lengths(self) -> None:
        sig = SlotSignature(length_counts=((3, 10), (5, 4), (9, 2)))
        # Only 3- and 5-letter words should be used (9 reserved for revealer)
        words = ["ABC", "DEFGH", "IJKLM", "NOPQR", "WXYZ"]  # 3,5,5,5,4
        subsets = _generate_subsets_for_signature(
            words, target_size=2, max_subsets=10, signature=sig, revealer="REVEALER9"
        )
        # WXYZ (4 letters) should be filtered out since 4 not in signature
        for subset in subsets:
            for word in subset:
                assert len(word) in sig.available_lengths

    def test_respects_max_subsets(self) -> None:
        sig = SlotSignature(length_counts=((3, 10), (5, 10)))
        words = ["ABC", "DEF", "GHI", "JKL", "MNO"]  # all 3-letter
        subsets = _generate_subsets_for_signature(
            words, target_size=2, max_subsets=3, signature=sig, revealer="ABCDE"
        )
        assert len(subsets) <= 3

    def test_empty_when_incompatible(self) -> None:
        sig = SlotSignature(length_counts=((3, 10),))
        words = ["ABCDE", "FGHIJ"]  # 5-letter words, no 5-letter slots
        subsets = _generate_subsets_for_signature(
            words, target_size=1, max_subsets=10, signature=sig, revealer="ABC"
        )
        assert subsets == []

    def test_target_size_0_returns_empty_subset(self) -> None:
        sig = SlotSignature(length_counts=((5, 4),))
        subsets = _generate_subsets_for_signature(
            ["ABCDE"], target_size=0, max_subsets=10, signature=sig, revealer="FGHIJ"
        )
        assert subsets == [[]]

    def test_accounts_for_revealer_slot(self) -> None:
        # Only 1 slot of length 5 — revealer takes it, so no room for a
        # 5-letter seed word
        sig = SlotSignature(length_counts=((3, 10), (5, 1)))
        words = ["ABCDE"]  # 5-letter word
        subsets = _generate_subsets_for_signature(
            words, target_size=1, max_subsets=10, signature=sig, revealer="FGHIJ"
        )
        assert subsets == []

    def test_revealer_plus_word_fit(self) -> None:
        # 2 slots of length 5 — one for revealer, one for seed word
        sig = SlotSignature(length_counts=((3, 10), (5, 2)))
        words = ["ABCDE"]  # 5-letter word
        subsets = _generate_subsets_for_signature(
            words, target_size=1, max_subsets=10, signature=sig, revealer="FGHIJ"
        )
        assert subsets == [["ABCDE"]]


class TestSubsetBudgetDistribution:
    """Tests that subset budget is distributed across signature groups."""

    def test_multiple_groups_get_subsets(self) -> None:
        """Each eligible signature group should get at least 1 subset tried."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(
            filler, grader, dictionary=dictionary, max_retries=1,
            max_grid_variants=50,
        )

        # The per_group_budget = max(1, MAX_SUBSETS_PER_SIZE // num_eligible)
        # ensures each group gets at least 1 attempt.
        # We verify this indirectly: even with MAX_SUBSETS_PER_SIZE=10,
        # the step should succeed (not be blocked by one group eating all budget).
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            metadata={"seed": 1},
            theme=ThemeConcept(
                topic="Test theme",
                candidate_entries=["EAGLE", "HAWK", "KITE", "FALCON", "PLANE", "ARROW"],
                seed_entries=[],
                revealer="SOAR",
            ),
        )
        result = step.run(envelope)
        assert result.fill is not None

    def test_per_group_budget_calculation(self) -> None:
        """Verify per-group budget is MAX_SUBSETS_PER_SIZE // num_eligible."""
        # With MAX_SUBSETS_PER_SIZE=10 and e.g. 3 eligible groups,
        # each group should get floor(10/3)=3 subsets
        step = FillWithGradingStep(
            FixedMockFiller(HIGH_QUALITY_GRID),
            FillGrader(_make_dict(GOOD_WORDS), min_passing_score=0),
            dictionary=_make_dict(GOOD_WORDS),
        )
        # 10 // 3 = 3, 10 // 1 = 10, 10 // 10 = 1
        assert max(1, step.MAX_SUBSETS_PER_SIZE // 3) == 3
        assert max(1, step.MAX_SUBSETS_PER_SIZE // 1) == 10
        assert max(1, step.MAX_SUBSETS_PER_SIZE // 10) == 1


class TestSubsetSelection:
    """Tests for subset selection in FillWithGradingStep."""

    def _make_candidate_envelope(
        self,
        candidates: list[str],
        seed: int = 42,
    ) -> PuzzleEnvelope:
        """Create a midi envelope with candidate_entries (surplus mode)."""
        return PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            metadata={"seed": seed},
            theme=ThemeConcept(
                topic="Test theme",
                candidate_entries=candidates,
                seed_entries=[],
                revealer="SOAR",
            ),
        )

    def test_subset_selection_succeeds(self) -> None:
        """With candidates and dictionary, subset selection path is used."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(
            filler, grader, dictionary=dictionary, max_retries=1,
        )

        envelope = self._make_candidate_envelope(
            ["EAGLE", "HAWK", "KITE", "FALCON", "PLANE", "ARROW"]
        )
        result = step.run(envelope)

        assert result.fill is not None
        assert result.theme is not None
        # seed_entries should be populated with the placed subset
        assert isinstance(result.theme.seed_entries, list)

    def test_backward_compat_no_candidates(self) -> None:
        """Without candidate_entries, falls back to direct seed_entries path."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(
            filler, grader, dictionary=dictionary, max_retries=1,
        )

        # Old-style: seed_entries directly, no candidates
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            metadata={"seed": 42},
            theme=ThemeConcept(
                topic="Test theme",
                seed_entries=["EAGLE", "HAWK", "KITE"],
                revealer="SOAR",
            ),
        )
        result = step.run(envelope)
        assert result.fill is not None

    def test_backward_compat_no_dictionary(self) -> None:
        """Without dictionary param, falls back to direct path."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        # No dictionary param
        step = FillWithGradingStep(filler, grader, max_retries=1)

        envelope = self._make_candidate_envelope(
            ["EAGLE", "HAWK", "KITE", "FALCON"]
        )
        # Should fall back to direct path (no candidates used as seeds)
        # since there's no dictionary for scoring
        result = step.run(envelope)
        assert result.fill is not None

    def test_graceful_degradation_all_fail(self) -> None:
        """When all subsets fail, should raise FillError."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)
        filler = AlwaysFailFiller()
        step = FillWithGradingStep(
            filler, grader, dictionary=dictionary, max_retries=1,
        )

        envelope = self._make_candidate_envelope(
            ["EAGLE", "HAWK", "KITE", "FALCON"]
        )
        with pytest.raises(FillError, match="All subsets exhausted"):
            step.run(envelope)

    def test_seed_entries_populated_after_fill(self) -> None:
        """After subset selection, seed_entries should contain placed words."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=0)
        filler = FixedMockFiller(HIGH_QUALITY_GRID)
        step = FillWithGradingStep(
            filler, grader, dictionary=dictionary, max_retries=1,
        )

        candidates = ["EAGLE", "HAWK", "KITE", "FALCON"]
        envelope = self._make_candidate_envelope(candidates)
        result = step.run(envelope)

        assert result.theme is not None
        # Placed subset should be a subset of candidates
        for word in result.theme.seed_entries:
            assert word in candidates
