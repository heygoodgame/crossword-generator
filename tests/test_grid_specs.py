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

    def test_black_cells_present_mini_5(self) -> None:
        spec = get_grid_spec(PuzzleType.MINI, 5)
        assert len(spec.black_cells) > 0

    def test_black_cells_present_mini_7(self) -> None:
        spec = get_grid_spec(PuzzleType.MINI, 7)
        assert len(spec.black_cells) > 0

    def test_midi_has_no_black_cells(self) -> None:
        spec = get_grid_spec(PuzzleType.MIDI, 9)
        assert spec.black_cells == []

    def test_seed_selects_pattern(self) -> None:
        """Different seeds can select different patterns."""
        patterns_seen = set()
        for seed in range(100):
            spec = get_grid_spec(PuzzleType.MINI, 7, seed=seed)
            patterns_seen.add(tuple(sorted(spec.black_cells)))
        # With 3 patterns and 100 seeds, we should see more than 1
        assert len(patterns_seen) > 1

    def test_same_seed_same_pattern(self) -> None:
        spec1 = get_grid_spec(PuzzleType.MINI, 7, seed=42)
        spec2 = get_grid_spec(PuzzleType.MINI, 7, seed=42)
        assert spec1.black_cells == spec2.black_cells

    def test_no_seed_uses_default(self) -> None:
        """Without seed, always returns the first pattern."""
        spec1 = get_grid_spec(PuzzleType.MINI, 5)
        spec2 = get_grid_spec(PuzzleType.MINI, 5)
        assert spec1.black_cells == spec2.black_cells

    def test_pattern_valid_structure_mini_5(self) -> None:
        """All 5x5 patterns produce grids where every slot is >= 2 letters."""
        for seed in range(100):
            spec = get_grid_spec(PuzzleType.MINI, 5, seed=seed)
            _assert_valid_grid_structure(spec)

    def test_pattern_valid_structure_mini_7(self) -> None:
        """All 7x7 patterns produce grids where every slot is >= 2 letters."""
        for seed in range(100):
            spec = get_grid_spec(PuzzleType.MINI, 7, seed=seed)
            _assert_valid_grid_structure(spec)

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


def _assert_valid_grid_structure(spec: "GridSpec") -> None:  # noqa: F821
    """Assert that all word slots in the grid are at least 2 letters long."""
    black = set(spec.black_cells)
    rows, cols = spec.rows, spec.cols

    # Check across slots
    for r in range(rows):
        c = 0
        while c < cols:
            if (r, c) in black:
                c += 1
                continue
            length = 0
            while c < cols and (r, c) not in black:
                length += 1
                c += 1
            if length > 0:
                assert length >= 3, (
                    f"Across slot at row {r} has length {length} < 3 "
                    f"(black_cells={spec.black_cells})"
                )

    # Check down slots
    for c in range(cols):
        r = 0
        while r < rows:
            if (r, c) in black:
                r += 1
                continue
            length = 0
            while r < rows and (r, c) not in black:
                length += 1
                r += 1
            if length > 0:
                assert length >= 3, (
                    f"Down slot at col {c} has length {length} < 3 "
                    f"(black_cells={spec.black_cells})"
                )
