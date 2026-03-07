"""Tests for the FillWithGradingStep composite pipeline step."""

from __future__ import annotations

import pytest

from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FilledGrid, GridFiller, GridSpec
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.models import FillResult, PuzzleEnvelope, PuzzleType
from crossword_generator.steps.fill_step import FillWithGradingStep


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
    "STARE": 80, "TONES": 75, "ARENA": 85, "RESET": 70, "SPEED": 90,
    "STARS": 80, "TORED": 60, "ANISE": 65, "RENEE": 60, "EASET": 55,
    "STORE": 70, "PARSE": 75, "ENTER": 80,
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
        filler = SequentialMockFiller([
            LOW_QUALITY_GRID,
            LOW_QUALITY_GRID,
            HIGH_QUALITY_GRID,
        ])
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

        filler = SequentialMockFiller([
            LOW_QUALITY_GRID,
            HIGH_QUALITY_GRID,
        ])
        step = FillWithGradingStep(filler, grader, max_retries=5)

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.attempt_number == 2
