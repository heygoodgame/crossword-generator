"""Tests for the FillStep pipeline step."""

import pytest

from crossword_generator.fillers.base import FilledGrid, FillError, GridFiller, GridSpec
from crossword_generator.models import FillResult, PuzzleEnvelope, PuzzleType
from crossword_generator.steps.fill_step import FillStep


class MockFiller(GridFiller):
    """A mock filler that returns a fixed grid."""

    def __init__(self, *, available: bool = True, error: bool = False) -> None:
        self._available = available
        self._error = error

    @property
    def name(self) -> str:
        return "mock"

    def fill(self, spec: GridSpec) -> FilledGrid:
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
