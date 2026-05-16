"""Tests for grid spec catalog."""

import pytest

from crossword_generator.grid_pattern_validation import (
    summarize_validations,
    validate_weighted_patterns,
)
from crossword_generator.grid_specs import get_grid_patterns, get_grid_spec
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

    def test_default_mini_5_is_open_grid(self) -> None:
        """Default 5x5 pattern (most common) has no black cells."""
        spec = get_grid_spec(PuzzleType.MINI, 5)
        assert spec.black_cells == []

    def test_black_cells_present_mini_7(self) -> None:
        spec = get_grid_spec(PuzzleType.MINI, 7)
        assert len(spec.black_cells) > 0

    def test_midi_has_black_cells(self) -> None:
        spec = get_grid_spec(PuzzleType.MIDI, 9)
        assert len(spec.black_cells) > 0

    def test_seed_selects_pattern(self) -> None:
        """Different seeds can select different patterns."""
        patterns_seen = set()
        for seed in range(100):
            spec = get_grid_spec(PuzzleType.MINI, 7, seed=seed)
            patterns_seen.add(tuple(sorted(spec.black_cells)))
        # With 4 patterns and 100 seeds, we should see more than 1
        assert len(patterns_seen) > 1

    def test_weighted_selection_mini_5(self) -> None:
        """5x5 patterns are selected with weighted probability."""
        patterns_seen = set()
        for seed in range(200):
            spec = get_grid_spec(PuzzleType.MINI, 5, seed=seed)
            patterns_seen.add(tuple(sorted(spec.black_cells)))
        # With 20 weighted patterns and 200 seeds, should see many
        assert len(patterns_seen) >= 10

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

    def test_black_cell_count_mini_7(self) -> None:
        """All 7x7 patterns have 4-13 black cells."""
        patterns_seen: set[tuple[tuple[int, int], ...]] = set()
        for seed in range(100):
            spec = get_grid_spec(PuzzleType.MINI, 7, seed=seed)
            patterns_seen.add(tuple(sorted(spec.black_cells)))
        for pattern in patterns_seen:
            assert 4 <= len(pattern) <= 13, (
                f"7x7 pattern has {len(pattern)} black cells, expected 4-13: {pattern}"
            )

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

    def test_midi_deterministic_default(self) -> None:
        """No seed uses seed=0, so result is deterministic."""
        spec1 = get_grid_spec(PuzzleType.MIDI, 9)
        spec2 = get_grid_spec(PuzzleType.MIDI, 9)
        assert spec1.black_cells == spec2.black_cells

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_midi_valid_structure(self, size: int) -> None:
        """Midi patterns have valid structure across 100 seeds."""
        for seed in range(100):
            spec = get_grid_spec(PuzzleType.MIDI, size, seed=seed)
            _assert_valid_grid_structure(spec)


class TestMiniGridPatternCatalog:
    @pytest.mark.parametrize(
        ("size", "expected_count", "expected_weight"),
        [(5, 34, 95), (7, 50, 86)],
    )
    def test_attachment_pattern_counts_and_weights(
        self,
        size: int,
        expected_count: int,
        expected_weight: int,
    ) -> None:
        patterns = get_grid_patterns(PuzzleType.MINI, size)
        assert len(patterns) == expected_count
        assert sum(pattern.weight for pattern in patterns) == expected_weight

    @pytest.mark.parametrize("size", [5, 7])
    def test_catalog_patterns_are_structurally_valid(self, size: int) -> None:
        patterns = [
            (list(pattern.black_cells), pattern.weight)
            for pattern in get_grid_patterns(PuzzleType.MINI, size)
        ]
        results = validate_weighted_patterns(size, patterns)
        assert all(result.valid for result in results)

    @pytest.mark.parametrize("size", [5, 7])
    def test_catalog_reports_symmetric_and_asymmetric_patterns(
        self, size: int
    ) -> None:
        patterns = get_grid_patterns(PuzzleType.MINI, size)
        summary = summarize_validations(
            validate_weighted_patterns(
                size,
                [(list(pattern.black_cells), pattern.weight) for pattern in patterns],
            )
        )
        assert summary["symmetric"] > 0
        assert summary["asymmetric"] > 0
        assert len(get_grid_patterns(PuzzleType.MINI, size, symmetric_only=True)) == (
            summary["symmetric"]
        )


class TestMidiGridPatternCatalog:
    def test_midi_9_uses_curated_feedback_patterns(self) -> None:
        patterns = get_grid_patterns(PuzzleType.MIDI, 9)

        assert len(patterns) == 47
        assert sum(pattern.weight for pattern in patterns) == 84
        assert any(pattern.symmetric for pattern in patterns)
        assert any(
            _is_vertically_mirrored(pattern.black_cells, size=9)
            and not pattern.symmetric
            for pattern in patterns
        )
        assert all(
            pattern.symmetric
            or _is_vertically_mirrored(pattern.black_cells, size=9)
            for pattern in patterns
        )
        assert all(
            _top_bottom_flipped(pattern.black_cells, size=9)
            in {other.black_cells for other in patterns}
            for pattern in patterns
        )

    def test_midi_9_patterns_avoid_corner_perimeter_black_triples(self) -> None:
        assert any(
            _has_perimeter_black_run(pattern.black_cells, size=9, run_length=3)
            for pattern in get_grid_patterns(PuzzleType.MIDI, 9)
        )
        for pattern in get_grid_patterns(PuzzleType.MIDI, 9):
            assert not _has_corner_perimeter_black_run(
                pattern.black_cells,
                size=9,
                run_length=3,
            )

    def test_midi_9_patterns_are_structurally_valid(self) -> None:
        patterns = [
            (list(pattern.black_cells), pattern.weight)
            for pattern in get_grid_patterns(PuzzleType.MIDI, 9)
        ]
        results = validate_weighted_patterns(9, patterns)
        assert all(result.valid for result in results)

    def test_midi_9_patterns_keep_long_slot_pressure_bounded(self) -> None:
        assert any(
            _long_slot_count(pattern.black_cells, size=9) == 4
            for pattern in get_grid_patterns(PuzzleType.MIDI, 9)
        )
        for pattern in get_grid_patterns(PuzzleType.MIDI, 9):
            assert _long_slot_count(pattern.black_cells, size=9) <= 4


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


def _is_vertically_mirrored(
    black_cells: tuple[tuple[int, int], ...],
    *,
    size: int,
) -> bool:
    black = set(black_cells)
    return all((r, size - 1 - c) in black for r, c in black)


def _top_bottom_flipped(
    black_cells: tuple[tuple[int, int], ...],
    *,
    size: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((size - 1 - r, c) for r, c in black_cells))


def _has_perimeter_black_run(
    black_cells: tuple[tuple[int, int], ...],
    *,
    size: int,
    run_length: int,
) -> bool:
    black = set(black_cells)
    for start in range(size - run_length + 1):
        if all((0, c) in black for c in range(start, start + run_length)):
            return True
        if all((size - 1, c) in black for c in range(start, start + run_length)):
            return True
        if all((r, 0) in black for r in range(start, start + run_length)):
            return True
        if all((r, size - 1) in black for r in range(start, start + run_length)):
            return True
    return False


def _has_corner_perimeter_black_run(
    black_cells: tuple[tuple[int, int], ...],
    *,
    size: int,
    run_length: int,
) -> bool:
    black = set(black_cells)
    edge_runs = [
        ((0, c) for c in range(run_length)),
        ((r, 0) for r in range(run_length)),
        ((0, c) for c in range(size - run_length, size)),
        ((r, size - 1) for r in range(run_length)),
        ((size - 1, c) for c in range(run_length)),
        ((r, 0) for r in range(size - run_length, size)),
        ((size - 1, c) for c in range(size - run_length, size)),
        ((r, size - 1) for r in range(size - run_length, size)),
    ]
    return any(all(cell in black for cell in run) for run in edge_runs)


def _long_slot_count(
    black_cells: tuple[tuple[int, int], ...],
    *,
    size: int,
) -> int:
    black = set(black_cells)
    count = 0

    for r in range(size):
        c = 0
        while c < size:
            if (r, c) in black:
                c += 1
                continue
            start = c
            while c < size and (r, c) not in black:
                c += 1
            if c - start in {8, 9}:
                count += 1

    for c in range(size):
        r = 0
        while r < size:
            if (r, c) in black:
                r += 1
                continue
            start = r
            while r < size and (r, c) not in black:
                r += 1
            if r - start in {8, 9}:
                count += 1

    return count
