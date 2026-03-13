"""Tests for the FillStep pipeline step."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from crossword_generator.fillers.base import FilledGrid, FillError, GridFiller, GridSpec
from crossword_generator.models import (
    FillGradeReport,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
    ThemeConcept,
)
from crossword_generator.steps.fill_step import FillStep, FillWithGradingStep


class MockFiller(GridFiller):
    """A mock filler that returns a fixed grid."""

    def __init__(self, *, available: bool = True, error: bool = False) -> None:
        self._available = available
        self._error = error
        self.fill_calls: list[GridSpec] = []

    @property
    def name(self) -> str:
        return "mock"

    def fill(self, spec: GridSpec) -> FilledGrid:
        self.fill_calls.append(spec)
        if self._error:
            raise FillError("Mock fill error")
        grid = [
            ["A", "B", "C", "D", "E"],
            ["F", "G", "H", "I", "J"],
            ["K", "L", "M", "N", "O"],
            ["P", "Q", "R", "S", "T"],
            ["U", "V", "W", "X", "Y"],
        ]
        return FilledGrid(
            grid=grid,
            words_across=["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXY"],
            words_down=["AFKPU", "BGLQV", "CHMRW", "DINSX", "EJOTY"],
        )

    def is_available(self) -> bool:
        return self._available


class TestFillStep:
    def test_run_fills_envelope(self) -> None:
        step = FillStep(MockFiller())
        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.filler_used == "mock"
        assert len(result.fill.grid) == 5
        assert len(result.fill.grid[0]) == 5

    def test_step_name(self) -> None:
        step = FillStep(MockFiller())
        assert step.name == "grid-fill"

    def test_step_history_updated(self) -> None:
        step = FillStep(MockFiller())
        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)
        assert "grid-fill" in result.step_history

    def test_original_envelope_unchanged(self) -> None:
        step = FillStep(MockFiller())
        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        step.run(envelope)
        assert envelope.fill is None  # Original should be unchanged

    def test_rejects_already_filled(self) -> None:
        step = FillStep(MockFiller())
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MINI,
            grid_size=5,
            fill=FillResult(grid=[["A"]], filler_used="other"),
        )
        with pytest.raises(ValueError, match="already has a fill"):
            step.run(envelope)

    def test_rejects_unavailable_filler(self) -> None:
        step = FillStep(MockFiller(available=False))
        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        with pytest.raises(ValueError, match="not available"):
            step.run(envelope)

    def test_filler_error_propagates(self) -> None:
        step = FillStep(MockFiller(error=True))
        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        with pytest.raises(FillError, match="Mock fill error"):
            step.run(envelope)

    def test_validate_input_returns_empty_on_valid(self) -> None:
        step = FillStep(MockFiller())
        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        errors = step.validate_input(envelope)
        assert errors == []


class TestFillWithGradingThemeFirst:
    """Tests for theme-first strategy in FillWithGradingStep."""

    def test_theme_first_strategy_attempted(self) -> None:
        """Verify theme-first is attempted when candidates are present."""
        filler = MockFiller()
        passing_report = FillGradeReport(
            overall_score=80.0, word_count=10, passing=True
        )

        mock_grader = _MockGrader(passing_report)
        mock_dict = _MockDictionary()

        step = FillWithGradingStep(
            filler, mock_grader, dictionary=mock_dict,
            max_retries=1, max_grid_variants=5,
        )

        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            theme=ThemeConcept(
                topic="test",
                seed_entries=[],
                candidate_entries=["GOLDMINES", "GOLDRATIO"],
                revealer="GOLDRATIO",
            ),
            metadata={"seed": 42},
        )

        # Patch build_themed_grids to return a dummy spec
        dummy_spec = GridSpec(rows=9, cols=9, black_cells=[], seed_entries={
            "0,0,across": "GOLDMINES",
            "4,0,across": "GOLDRATIO",
        })

        with patch(
            "crossword_generator.steps.fill_step.build_themed_grids",
            return_value=[dummy_spec],
        ) as mock_build:
            result = step.run(envelope)

        # Theme-first should have been called
        mock_build.assert_called()
        assert result.fill is not None
        assert result.fill.grade_report is not None
        assert result.fill.grade_report.passing

    def test_theme_first_fallback_to_random(self) -> None:
        """If theme-first fails, random-grid still runs."""
        filler = MockFiller()
        passing_report = FillGradeReport(
            overall_score=80.0, word_count=10, passing=True
        )
        mock_grader = _MockGrader(passing_report)
        mock_dict = _MockDictionary()

        step = FillWithGradingStep(
            filler, mock_grader, dictionary=mock_dict,
            max_retries=1, max_grid_variants=5,
        )

        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            theme=ThemeConcept(
                topic="test",
                seed_entries=[],
                candidate_entries=["GOLDMINES", "GOLDRATIO"],
                revealer="GOLDRATIO",
            ),
            metadata={"seed": 42},
        )

        # Theme-first returns nothing, forcing fallback
        with patch(
            "crossword_generator.steps.fill_step.build_themed_grids",
            return_value=[],
        ):
            result = step.run(envelope)

        # Should still produce a result via random-grid fallback
        assert result.fill is not None


class _MockGrader:
    """Minimal mock for FillGrader that returns a fixed report."""

    def __init__(self, report: FillGradeReport) -> None:
        self._report = report

    def grade(self, grid: list[list[str]]) -> FillGradeReport:
        return self._report


class _MockDictionary:
    """Minimal mock dictionary for testing."""

    def add_words(self, words: dict[str, int]) -> None:
        pass

    def words_by_length(self, length: int) -> list[str]:
        return []

    def lookup(self, word: str) -> int | None:
        return 50
