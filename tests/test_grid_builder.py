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
        entries = ["SUNSET", "ARCHES", "FLEECE"]
        revealer = "GOLDRATIO"
        results = build_themed_grids(
            9, entries, revealer, seed=42, count=10
        )
        # At least some should succeed
        if results:
            spec = results[0]
            placed_words = set(spec.seed_entries.values())
            # Revealer (9 letters) and entries (6 letters each) should be placed
            assert revealer in placed_words
            # At least some entries should be placed
            placed_entries = placed_words - {revealer}
            assert len(placed_entries) > 0

    def test_symmetry_maintained(self) -> None:
        """Black cells are 180-degree symmetric."""
        results = build_themed_grids(
            9, ["RATIO"], "GOLDRATIO", seed=42, count=5
        )
        for spec in results:
            black_set = set(spec.black_cells)
            for r, c in spec.black_cells:
                assert (8 - r, 8 - c) in black_set, (
                    f"({r},{c}) missing symmetric partner"
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
            9, ["SUNSET", "ARCHES"], "GOLDRATIO", seed=0, count=100
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
            9, entries, revealer, seed=0, count=500
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
            9, entries, revealer, seed=0, count=500
        )
        for spec in results:
            blacks = set(spec.black_cells)
            assert _is_connected(9, 9, blacks)
            assert not _has_any_2x2_block(9, 9, blacks)
            assert _check_min_word_length(9, 9, blacks, 3)
            assert _all_rows_cols_have_white(9, 9, blacks)
            # Symmetry
            for r, c in spec.black_cells:
                assert (8 - r, 8 - c) in blacks
