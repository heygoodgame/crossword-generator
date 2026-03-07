"""Tests for grid spec catalog."""

import pytest

from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.models import PuzzleType


class TestGetGridSpec:
    def test_mini_5(self) -> None:
        spec = get_grid_spec(PuzzleType.MINI, 5)
        assert spec.rows == 5
        assert spec.cols == 5

    def test_mini_7(self) -> None:
        spec = get_grid_spec(PuzzleType.MINI, 7)
        assert spec.rows == 7
        assert spec.cols == 7

    def test_midi_9(self) -> None:
        spec = get_grid_spec(PuzzleType.MIDI, 9)
        assert spec.rows == 9
        assert spec.cols == 9

    def test_midi_10(self) -> None:
        spec = get_grid_spec(PuzzleType.MIDI, 10)
        assert spec.rows == 10
        assert spec.cols == 10

    def test_midi_11(self) -> None:
        spec = get_grid_spec(PuzzleType.MIDI, 11)
        assert spec.rows == 11
        assert spec.cols == 11

    def test_string_puzzle_type(self) -> None:
        spec = get_grid_spec("mini", 5)
        assert spec.rows == 5

    def test_black_cells_empty(self) -> None:
        spec = get_grid_spec(PuzzleType.MINI, 5)
        assert spec.black_cells == []

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            get_grid_spec(PuzzleType.MINI, 9)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            get_grid_spec("large", 5)

    def test_mini_3_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            get_grid_spec(PuzzleType.MINI, 3)

    def test_midi_5_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            get_grid_spec(PuzzleType.MIDI, 5)
