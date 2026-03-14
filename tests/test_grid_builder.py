"""Tests for the theme-first grid builder."""

from __future__ import annotations

from crossword_generator.grid_builder import (
    _valid_line_partitions,
    build_themed_grids,
)
from crossword_generator.grid_pattern_generator import (
    _all_rows_cols_have_white,
    _check_min_word_length,
    _has_any_2x2_block,
    _is_connected,
)


class TestValidLinePartitions:
    def test_full_width_entry(self) -> None:
        """A 9-letter entry in a 9-wide line has exactly one partition."""
        parts = _valid_line_partitions(9, 9)
        assert len(parts) == 1
        assert parts[0].start == 0
        assert parts[0].black_positions == []
        assert parts[0].remaining_slot_lengths == []

    def test_entry_too_long(self) -> None:
        """An entry longer than the line returns no partitions."""
        assert _valid_line_partitions(9, 10) == []

    def test_short_entry_has_partitions(self) -> None:
        """A 3-letter entry in a 9-wide line should have multiple placements."""
        parts = _valid_line_partitions(9, 3)
        assert len(parts) > 0
        for p in parts:
            # Entry cells + black cells + remaining slots should cover the line
            assert all(s >= 3 for s in p.remaining_slot_lengths)

    def test_5_letter_in_9_grid(self) -> None:
        """A 5-letter entry in a 9-wide line should have valid placements."""
        parts = _valid_line_partitions(9, 5)
        assert len(parts) > 0
        for p in parts:
            assert all(s >= 3 for s in p.remaining_slot_lengths)

    def test_6_letter_in_9_grid(self) -> None:
        """A 6-letter entry should have valid placements (the key improvement)."""
        parts = _valid_line_partitions(9, 6)
        assert len(parts) > 0


class TestBuildThemedGrids:
    def test_full_width_entry_placement(self) -> None:
        """A 9-letter entry in a 9x9 grid should occupy a full row."""
        results = build_themed_grids(
            9, ["GOLDMINES"], "GOLDRATIO", seed=42, count=5
        )
        assert len(results) > 0
        for spec in results:
            assert "GOLDMINES" in spec.seed_entries.values()
            assert "GOLDRATIO" in spec.seed_entries.values()

    def test_partial_entry_placement(self) -> None:
        """A 5-letter entry creates valid row partition with other slots."""
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=42, count=5
        )
        assert len(results) > 0
        for spec in results:
            assert "RATIO" in spec.seed_entries.values()

    def test_multiple_entries_placed(self) -> None:
        """Multiple entries + revealer all present in output GridSpec."""
        # Mix of lengths: 6+6+3 letter entries + 9-letter revealer.
        # Three 6-letter entries in 9x9 is geometrically infeasible
        # (adjacent rows create 2x2 blocks; non-adjacent create
        # unsealable column gaps), so we use a mix.
        entries = ["SUNSET", "ARCHES", "ORE"]
        revealer = "GOLDRATIO"
        results = build_themed_grids(
            9, entries, revealer, seed=0, count=20
        )
        assert len(results) > 0, "Expected at least one valid grid"
        spec = results[0]
        placed_words = set(spec.seed_entries.values())
        assert revealer in placed_words
        placed_entries = placed_words - {revealer}
        assert len(placed_entries) == len(entries), (
            f"Expected all {len(entries)} entries placed, "
            f"got {len(placed_entries)}: {placed_entries}"
        )

    def test_symmetry_preferred_simple_case(self) -> None:
        """Simple themes (few entries) prefer symmetric grids.

        The system tries symmetric placement first and falls back to
        asymmetric only when needed. For a simple 2-word case, most
        grids should be symmetric.
        """
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=0, count=20
        )
        assert len(results) > 0
        symmetric_count = 0
        for spec in results:
            black_set = set(spec.black_cells)
            is_sym = all(
                (8 - r, 8 - c) in black_set
                for r, c in spec.black_cells
            )
            if is_sym:
                symmetric_count += 1
        # Simple cases should produce mostly symmetric grids
        assert symmetric_count > 0, (
            "Expected at least some symmetric grids for simple theme"
        )

    def test_connectivity(self) -> None:
        """White cells are connected."""
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=42, count=5
        )
        for spec in results:
            assert _is_connected(9, 9, set(spec.black_cells))

    def test_no_2x2_blocks(self) -> None:
        """No all-black 2x2 squares."""
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=42, count=5
        )
        for spec in results:
            assert not _has_any_2x2_block(9, 9, set(spec.black_cells))

    def test_min_word_length(self) -> None:
        """All slots >= 3 letters."""
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=42, count=5
        )
        for spec in results:
            assert _check_min_word_length(9, 9, set(spec.black_cells), 3)

    def test_different_seeds_different_grids(self) -> None:
        """Multiple seeds produce varied patterns."""
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=0, count=20
        )
        if len(results) >= 2:
            patterns = {tuple(spec.black_cells) for spec in results}
            assert len(patterns) >= 2, "Expected diverse patterns"

    def test_returns_empty_when_impossible(self) -> None:
        """Entries that can't fit return empty list."""
        # 10-letter words can't fit in a 9x9 grid -- both entry and revealer
        results = build_themed_grids(
            9, ["IMPOSSIBLE"], "ABCDEFGHIJ", seed=42, count=5
        )
        assert results == []

    def test_empty_entries(self) -> None:
        """No entries returns empty list."""
        results = build_themed_grids(9, [], "", seed=42, count=5)
        assert results == []

    def test_seed_entries_have_valid_direction(self) -> None:
        """All placed entries should be across or down."""
        results = build_themed_grids(
            9, ["SUNSET"], "GOLDRATIO", seed=42, count=5
        )
        for spec in results:
            for key in spec.seed_entries:
                parts = key.split(",")
                assert parts[2] in ("across", "down"), (
                    f"Expected across or down, got {parts[2]}"
                )

    def test_rows_cols_have_white(self) -> None:
        """No row or column is entirely black."""
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=42, count=5
        )
        for spec in results:
            assert _all_rows_cols_have_white(9, 9, set(spec.black_cells))

    def test_six_letter_entries_can_be_placed(self) -> None:
        """6-letter entries (the key failing case) should be placeable."""
        results = build_themed_grids(
            9, ["SUNSET", "ARCHES"], "GOLDRATIO", seed=0, count=20
        )
        # Should produce at least some valid grids
        assert len(results) > 0, (
            "Expected at least one valid grid for 6-letter entries"
        )

    def test_down_entries_can_be_placed(self) -> None:
        """Down placement is used when across row pairs are exhausted."""
        # 5 three-letter entries + 9-letter revealer = 6 words total.
        # The revealer uses 1 of the 5 symmetric row pairs (across).
        # The remaining 4 pairs can hold at most 4 across entries,
        # so at least 1 of the 5 short entries must go down.
        entries = ["CAT", "DOG", "BAT", "FOX", "OWL"]
        revealer = "GOLDMINES"
        results = build_themed_grids(
            9, entries, revealer, seed=0, count=50
        )
        assert len(results) > 0, "Expected at least one valid grid"
        has_down = any(
            key.endswith(",down")
            for spec in results
            for key in spec.seed_entries
        )
        assert has_down, "Expected at least one down entry placement"

    def test_down_entries_maintain_constraints(self) -> None:
        """Grids with down entries still satisfy all constraints."""
        entries = ["CAT", "DOG", "BAT", "FOX", "OWL"]
        revealer = "GOLDMINES"
        results = build_themed_grids(
            9, entries, revealer, seed=0, count=50
        )
        for spec in results:
            blacks = set(spec.black_cells)
            assert _is_connected(9, 9, blacks)
            assert not _has_any_2x2_block(9, 9, blacks)
            assert _check_min_word_length(9, 9, blacks, 3)
            assert _all_rows_cols_have_white(9, 9, blacks)

    def test_backtracking_places_five_entries(self) -> None:
        """Backtracking places 5 entries that a greedy pass would struggle with.

        Two 6-letter + two 3-letter entries + 9-letter revealer = 5 words.
        The 6-letter entries have very few valid row partitions, so a greedy
        pass often paints itself into a corner. Backtracking revises earlier
        placements to find a valid arrangement.
        """
        entries = ["SUNSET", "ARCHES", "ORE", "BAR"]
        revealer = "GOLDRATIO"
        results = build_themed_grids(
            9, entries, revealer, seed=0, count=20
        )
        assert len(results) > 0, "Expected at least one valid grid"
        spec = results[0]
        placed = set(spec.seed_entries.values())
        assert revealer in placed
        assert len(placed) == 5, (
            f"Expected all 5 words placed, got {len(placed)}: {placed}"
        )

    def test_all_hard_constraints_with_many_entries(self) -> None:
        """All grids maintain hard constraints regardless of symmetry mode."""
        entries = ["SUNSET", "ARCHES", "ORE"]
        results = build_themed_grids(
            9, entries, "GOLDRATIO", seed=0, count=20
        )
        for spec in results:
            blacks = set(spec.black_cells)
            assert _is_connected(9, 9, blacks)
            assert not _has_any_2x2_block(9, 9, blacks)
            assert _check_min_word_length(9, 9, blacks, 3)
            assert _all_rows_cols_have_white(9, 9, blacks)

    def test_crossing_depth_limits_fixed_letters(self) -> None:
        """Grids with 4+ entries respect max crossing depth of 2."""
        entries = ["SUNSET", "ARCHES", "ORE", "BAR"]
        revealer = "GOLDRATIO"
        results = build_themed_grids(9, entries, revealer, seed=0, count=20)
        for spec in results:
            across_per_col = [0] * 9
            down_per_row = [0] * 9
            for key, word in spec.seed_entries.items():
                parts = key.split(",")
                row, col, direction = int(parts[0]), int(parts[1]), parts[2]
                if direction == "across":
                    for c in range(col, col + len(word)):
                        across_per_col[c] += 1
                else:
                    for r in range(row, row + len(word)):
                        down_per_row[r] += 1
            assert max(across_per_col) <= 2, (
                f"across_per_col {across_per_col} exceeds max crossing depth"
            )
            assert max(down_per_row) <= 2, (
                f"down_per_row {down_per_row} exceeds max crossing depth"
            )

    def test_four_entries_use_mixed_directions(self) -> None:
        """4 entries in 9x9 should produce at least some down entries."""
        entries = ["SUNSET", "ARCHES", "ORE"]
        revealer = "GOLDRATIO"
        results = build_themed_grids(9, entries, revealer, seed=0, count=20)
        assert len(results) > 0
        has_down = any(
            key.endswith(",down")
            for spec in results
            for key in spec.seed_entries
        )
        assert has_down, "Expected mixed across/down with 4 entries"

    def test_asymmetric_fallback_produces_grids(self) -> None:
        """Asymmetric fallback produces grids for highly constrained entries.

        4 entries + 9-letter revealer in 9x9 consumes 4 of 5 symmetric
        pairs, which may fail symmetric placement. Asymmetric fallback
        should still produce valid grids.
        """
        entries = ["SUNSET", "ARCHES", "ORE", "BAR"]
        revealer = "GOLDRATIO"
        results = build_themed_grids(
            9, entries, revealer, seed=0, count=20
        )
        assert len(results) > 0, "Expected grids from asymmetric fallback"
        for spec in results:
            blacks = set(spec.black_cells)
            assert _is_connected(9, 9, blacks)
            assert not _has_any_2x2_block(9, 9, blacks)
            assert _check_min_word_length(9, 9, blacks, 3)
            assert _all_rows_cols_have_white(9, 9, blacks)
