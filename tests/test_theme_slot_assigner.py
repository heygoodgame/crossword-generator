"""Tests for the theme slot assigner."""

from __future__ import annotations

import pytest

from crossword_generator.fillers.csp import Slot, extract_slots
from crossword_generator.steps.theme_slot_assigner import (
    assign_seed_entries_to_slots,
)


def _make_slot(
    index: int,
    row: int,
    col: int,
    direction: str,
    length: int,
) -> Slot:
    """Create a Slot with appropriate cells."""
    if direction == "across":
        cells = [(row, col + i) for i in range(length)]
    else:
        cells = [(row + i, col) for i in range(length)]
    return Slot(
        index=index,
        row=row,
        col=col,
        direction=direction,
        length=length,
        cells=cells,
    )


class TestAssignSeedEntriesToSlots:
    def test_assigns_matching_length_slots(self) -> None:
        slots = [
            _make_slot(0, 0, 0, "across", 5),
            _make_slot(1, 1, 0, "across", 4),
            _make_slot(2, 2, 0, "across", 3),
        ]
        assignments = assign_seed_entries_to_slots(
            seed_entries=["EAGLE", "KITE"],
            revealer="ACE",
            slots=slots,
        )
        words = {a.word for a in assignments}
        assert words == {"EAGLE", "KITE", "ACE"}
        # Check length matching
        for a in assignments:
            assert a.length == len(a.word)

    def test_prefers_across_over_down(self) -> None:
        slots = [
            _make_slot(0, 0, 0, "down", 5),
            _make_slot(1, 0, 0, "across", 5),
        ]
        assignments = assign_seed_entries_to_slots(
            seed_entries=["EAGLE"],
            revealer="",
            slots=slots,
        )
        assert len(assignments) == 1
        assert assignments[0].direction == "across"

    def test_longest_first_assignment(self) -> None:
        # Two across slots of length 5 and 4; two words of length 5 and 4
        slots = [
            _make_slot(0, 0, 0, "across", 5),
            _make_slot(1, 1, 0, "across", 4),
        ]
        assignments = assign_seed_entries_to_slots(
            seed_entries=["KITE"],  # 4 letters
            revealer="EAGLE",  # 5 letters
            slots=slots,
        )
        # EAGLE (5) should be assigned first to slot 0 (length 5)
        eagle_assignment = next(a for a in assignments if a.word == "EAGLE")
        kite_assignment = next(a for a in assignments if a.word == "KITE")
        assert eagle_assignment.length == 5
        assert kite_assignment.length == 4

    def test_raises_on_no_matching_slot(self) -> None:
        slots = [
            _make_slot(0, 0, 0, "across", 3),
        ]
        with pytest.raises(ValueError, match="No available slot of length 5"):
            assign_seed_entries_to_slots(
                seed_entries=["EAGLE"],
                revealer="",
                slots=slots,
            )

    def test_no_double_assignment(self) -> None:
        # Two words of the same length need distinct slots
        slots = [
            _make_slot(0, 0, 0, "across", 4),
            _make_slot(1, 1, 0, "across", 4),
        ]
        assignments = assign_seed_entries_to_slots(
            seed_entries=["KITE", "HAWK"],
            revealer="",
            slots=slots,
        )
        assert len(assignments) == 2
        slot_indices = {
            next(
                s.index for s in slots
                if s.row == a.row and s.col == a.col and s.direction == a.direction
            )
            for a in assignments
        }
        assert len(slot_indices) == 2  # Distinct slots

    def test_revealer_included_in_assignment(self) -> None:
        slots = [
            _make_slot(0, 0, 0, "across", 5),
            _make_slot(1, 1, 0, "across", 4),
        ]
        assignments = assign_seed_entries_to_slots(
            seed_entries=["KITE"],
            revealer="EAGLE",
            slots=slots,
        )
        words = {a.word for a in assignments}
        assert "EAGLE" in words
        assert "KITE" in words

    def test_empty_revealer_ok(self) -> None:
        slots = [
            _make_slot(0, 0, 0, "across", 5),
        ]
        assignments = assign_seed_entries_to_slots(
            seed_entries=["EAGLE"],
            revealer="",
            slots=slots,
        )
        assert len(assignments) == 1
        assert assignments[0].word == "EAGLE"

    def test_with_real_grid_slots(self) -> None:
        # Use extract_slots on a 9x9 grid with some black cells
        black = {(0, 0), (0, 8), (4, 4), (8, 0), (8, 8)}
        slots = extract_slots(9, 9, black)

        # Find available lengths
        lengths = sorted({s.length for s in slots})
        assert len(lengths) > 0

        # Pick words that match available lengths
        # Just use simple uppercase words of matching length
        seed_entries = []
        used_lengths: set[int] = set()
        for length in lengths:
            if length >= 3 and length not in used_lengths:
                seed_entries.append("A" * length)
                used_lengths.add(length)
                if len(seed_entries) >= 2:
                    break

        if seed_entries:
            assignments = assign_seed_entries_to_slots(
                seed_entries=seed_entries,
                revealer="",
                slots=slots,
            )
            assert len(assignments) == len(seed_entries)
