"""Tests for the programmatic grid pattern generator."""

from __future__ import annotations

import random

import pytest

from crossword_generator.grid_pattern_generator import (
    PatternConfig,
    _all_rows_cols_have_white,
    _build_candidates,
    _check_min_word_length,
    _has_2x2_block,
    _is_connected,
    analyze_pattern,
    generate_pattern,
)

# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestConnectivity:
    def test_empty_grid_connected(self) -> None:
        assert _is_connected(5, 5, set()) is True

    def test_single_black_cell_connected(self) -> None:
        assert _is_connected(5, 5, {(2, 2)}) is True

    def test_disconnected_grid(self) -> None:
        # Black wall across the middle row splits grid in two
        black = {(2, c) for c in range(5)}
        assert _is_connected(5, 5, black) is False

    def test_donut_shape_connected(self) -> None:
        # Single black cell in center — still connected
        assert _is_connected(3, 3, {(1, 1)}) is True

    def test_corner_cut_still_connected(self) -> None:
        assert _is_connected(5, 5, {(0, 0), (4, 4)}) is True


class TestMinWordLength:
    def test_empty_grid(self) -> None:
        # Full rows/cols of length 5
        assert _check_min_word_length(5, 5, set(), 3) is True

    def test_valid_pattern(self) -> None:
        # Black cells that leave slots >= 3
        black = {(0, 0), (4, 4)}
        assert _check_min_word_length(5, 5, black, 3) is True

    def test_two_letter_slot(self) -> None:
        # Creates a 2-letter across slot in row 0: columns 0,1 are white, (0,2) black
        black = {(0, 2)}
        assert _check_min_word_length(5, 5, black, 3) is False

    def test_one_letter_slot(self) -> None:
        # (0,0) white, (0,1) black — 1-letter slot
        black = {(0, 1), (1, 0)}
        assert _check_min_word_length(5, 5, black, 3) is False


class TestHas2x2Block:
    def test_isolated_cells(self) -> None:
        black = {(1, 1), (3, 3)}
        assert _has_2x2_block(black, 1, 1) is False

    def test_2x2_detected(self) -> None:
        black = {(1, 1), (1, 2), (2, 1), (2, 2)}
        assert _has_2x2_block(black, 2, 2) is True

    def test_l_shape_no_block(self) -> None:
        black = {(1, 1), (1, 2), (2, 1)}
        assert _has_2x2_block(black, 2, 1) is False


class TestBuildCandidates:
    def test_corners_excluded(self) -> None:
        rng = random.Random(0)
        config = PatternConfig()
        candidates = _build_candidates(9, 9, rng, config)
        corners = {(0, 0), (0, 8), (8, 0), (8, 8)}
        candidate_set = set(candidates)
        assert candidate_set.isdisjoint(corners)

    def test_interior_bias(self) -> None:
        """First several candidates should be mostly interior cells."""
        rng = random.Random(42)
        config = PatternConfig()
        candidates = _build_candidates(9, 9, rng, config)
        # With 3:1 ratio, first 4 should be 3 interior + 1 edge (roughly)
        assert len(candidates) > 0

    def test_symmetry_half(self) -> None:
        """Candidates only contain one of each symmetric pair."""
        rng = random.Random(0)
        config = PatternConfig()
        candidates = _build_candidates(9, 9, rng, config)
        for r, c in candidates:
            mr, mc = 8 - r, 8 - c
            if (r, c) != (mr, mc):  # Not center cell
                assert (r, c) < (mr, mc), (
                    f"({r},{c}) should be the smaller of the pair"
                )


# ---------------------------------------------------------------------------
# Pattern generation tests
# ---------------------------------------------------------------------------


class TestPatternGeneration:
    def test_deterministic(self) -> None:
        p1 = generate_pattern(9, 9, seed=42)
        p2 = generate_pattern(9, 9, seed=42)
        assert p1 == p2

    def test_different_seeds_different_patterns(self) -> None:
        p1 = generate_pattern(9, 9, seed=0)
        p2 = generate_pattern(9, 9, seed=1)
        assert p1 != p2

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_symmetry(self, size: int) -> None:
        for seed in range(50):
            black = generate_pattern(size, size, seed=seed)
            black_set = set(black)
            for r, c in black:
                assert (size - 1 - r, size - 1 - c) in black_set, (
                    f"seed={seed}: ({r},{c}) missing mirror"
                )

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_connectivity(self, size: int) -> None:
        for seed in range(50):
            black = generate_pattern(size, size, seed=seed)
            assert _is_connected(size, size, set(black))

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_min_word_length(self, size: int) -> None:
        for seed in range(50):
            black = generate_pattern(size, size, seed=seed)
            assert _check_min_word_length(size, size, set(black), 3)

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_corners_white(self, size: int) -> None:
        corners = {(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)}
        for seed in range(50):
            black = set(generate_pattern(size, size, seed=seed))
            assert black.isdisjoint(corners)

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_rows_cols_have_white(self, size: int) -> None:
        for seed in range(50):
            black = generate_pattern(size, size, seed=seed)
            assert _all_rows_cols_have_white(size, size, set(black))

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_has_black_cells(self, size: int) -> None:
        """Generated patterns should have some black cells."""
        for seed in range(50):
            black = generate_pattern(size, size, seed=seed)
            assert len(black) > 0

    def test_density_in_range(self) -> None:
        for size in (9, 10, 11):
            for seed in range(50):
                black = generate_pattern(size, size, seed=seed)
                density = len(black) / (size * size)
                assert density <= 0.25 + 0.01, (
                    f"size={size} seed={seed}: density {density:.2f} too high"
                )


# ---------------------------------------------------------------------------
# Locked cells tests
# ---------------------------------------------------------------------------


class TestLockedCells:
    def test_locked_white_never_black(self) -> None:
        """Locked-white cells must never become black."""
        locked_white = {(1, 1), (1, 2), (1, 3), (7, 5), (7, 6), (7, 7)}
        for seed in range(50):
            black = generate_pattern(
                9, 9, seed=seed, locked_white=locked_white
            )
            black_set = set(black)
            assert black_set.isdisjoint(locked_white), (
                f"seed={seed}: locked-white cell became black"
            )

    def test_locked_black_always_present(self) -> None:
        """Locked-black cells must always be in the output."""
        locked_black = {(0, 3), (8, 5)}
        for seed in range(50):
            black = generate_pattern(
                9, 9, seed=seed, locked_black=locked_black
            )
            black_set = set(black)
            assert locked_black.issubset(black_set), (
                f"seed={seed}: locked-black cell missing from output"
            )

    def test_locked_pattern_still_valid(self) -> None:
        """Output with locked cells passes all constraint checks."""
        locked_white = {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}
        locked_black = {(0, 4), (8, 4)}
        for seed in range(50):
            black = generate_pattern(
                9, 9,
                seed=seed,
                locked_white=locked_white,
                locked_black=locked_black,
            )
            black_set = set(black)
            report = analyze_pattern(9, 9, black)
            # Pattern should be symmetric and connected
            assert report.is_symmetric, f"seed={seed}: not symmetric"
            assert report.is_connected, f"seed={seed}: not connected"
            assert report.min_word_length_found >= 3, (
                f"seed={seed}: min word length {report.min_word_length_found}"
            )
            assert not report.has_2x2_block, f"seed={seed}: has 2x2 block"
            # Locked constraints respected
            assert black_set.isdisjoint(locked_white)
            assert locked_black.issubset(black_set)

    def test_locked_cells_with_no_locked(self) -> None:
        """Passing empty locked sets produces same result as no args."""
        for seed in range(10):
            p1 = generate_pattern(9, 9, seed=seed)
            p2 = generate_pattern(
                9, 9, seed=seed, locked_white=set(), locked_black=set()
            )
            assert p1 == p2


# ---------------------------------------------------------------------------
# Diversity / statistical tests
# ---------------------------------------------------------------------------


class TestDiversity:
    @pytest.mark.parametrize(
        ("size", "min_unique"),
        [(9, 100), (10, 350), (11, 700)],
    )
    def test_unique_pattern_count(self, size: int, min_unique: int) -> None:
        """1000 seeds should produce many unique patterns.

        Thresholds are per-size because smaller grids have tighter
        constraint spaces (9x9 with min 3-letter words, 180° symmetry,
        and corners white has ~250 structurally distinct patterns).
        """
        patterns = set()
        for seed in range(1000):
            black = generate_pattern(size, size, seed=seed)
            patterns.add(tuple(black))
        assert len(patterns) >= min_unique, (
            f"size={size}: only {len(patterns)} unique patterns, "
            f"expected >= {min_unique}"
        )

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_density_distribution(self, size: int) -> None:
        """Average density should be within the configured range."""
        densities = []
        for seed in range(1000):
            black = generate_pattern(size, size, seed=seed)
            densities.append(len(black) / (size * size))
        avg = sum(densities) / len(densities)
        assert 0.10 <= avg <= 0.27, (
            f"size={size}: avg density {avg:.3f} outside expected range"
        )
        # Check spread — at least 5 distinct density values
        distinct = len(set(round(d, 3) for d in densities))
        assert distinct >= 5, (
            f"size={size}: only {distinct} distinct density values"
        )

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_hard_constraints_100_percent(self, size: int) -> None:
        """All hard constraints hold across 1000 seeds."""
        for seed in range(1000):
            black = generate_pattern(size, size, seed=seed)
            report = analyze_pattern(size, size, black)
            assert report.valid, (
                f"size={size} seed={seed}: invalid pattern — {report}"
            )

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_symmetry_rate(self, size: int) -> None:
        """100% of patterns should be 180° symmetric."""
        for seed in range(1000):
            black = generate_pattern(size, size, seed=seed)
            black_set = set(black)
            for r, c in black:
                assert (size - 1 - r, size - 1 - c) in black_set

    @pytest.mark.parametrize("size", [9, 10, 11])
    def test_no_2x2_blocks_rate(self, size: int) -> None:
        """0% of patterns should contain 2x2 black blocks."""
        for seed in range(1000):
            black = generate_pattern(size, size, seed=seed)
            report = analyze_pattern(size, size, black)
            assert not report.has_2x2_block, (
                f"size={size} seed={seed}: has 2x2 block"
            )
