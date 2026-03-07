"""Tests for crossword clue numbering utility."""

from crossword_generator.exporters.numbering import compute_numbering

# Simple 3x3 grid with no black squares:
# A B C
# D E F
# G H I
GRID_3X3_OPEN = [
    ["A", "B", "C"],
    ["D", "E", "F"],
    ["G", "H", "I"],
]

# 5x5 grid with black squares:
# O H . . .
# F . P . F
# . H O W L
# A . S . I
# S L E E P
GRID_5X5 = [
    ["O", "H", ".", ".", "."],
    ["F", ".", "P", ".", "F"],
    [".", "H", "O", "W", "L"],
    ["A", ".", "S", ".", "I"],
    ["S", "L", "E", "E", "P"],
]


class TestComputeNumbering:
    def test_open_3x3_grid(self) -> None:
        entries = compute_numbering(GRID_3X3_OPEN)
        # Only 1-Across (ABC) and 1-Down (ADG) start at (0,0)
        # because all other cells have a letter to the left or above
        numbers = {e.number for e in entries}
        assert 1 in numbers

    def test_open_3x3_across(self) -> None:
        entries = compute_numbering(GRID_3X3_OPEN)
        across = [e for e in entries if e.direction == "across"]
        # Open 3x3: each row starts an across word (left edge)
        assert len(across) == 3
        assert across[0].answer == "ABC"
        assert across[0].length == 3

    def test_open_3x3_down(self) -> None:
        entries = compute_numbering(GRID_3X3_OPEN)
        down = [e for e in entries if e.direction == "down"]
        # 1-Down: ADG (col 0)
        assert any(e.answer == "ADG" for e in down)

    def test_5x5_numbering(self) -> None:
        entries = compute_numbering(GRID_5X5)
        across = [e for e in entries if e.direction == "across"]
        down = [e for e in entries if e.direction == "down"]

        # Check some known words
        across_answers = {e.answer for e in across}
        assert "OH" in across_answers
        assert "HOWL" in across_answers
        assert "SLEEP" in across_answers

        down_answers = {e.answer for e in down}
        assert "OF" in down_answers

    def test_numbering_sequential(self) -> None:
        entries = compute_numbering(GRID_5X5)
        numbers = [e.number for e in entries]
        # Numbers should be monotonically non-decreasing
        for i in range(1, len(numbers)):
            assert numbers[i] >= numbers[i - 1]

    def test_across_before_down_at_same_number(self) -> None:
        entries = compute_numbering(GRID_5X5)
        # Where a cell starts both across and down, across comes first
        seen: dict[int, list[str]] = {}
        for e in entries:
            seen.setdefault(e.number, []).append(e.direction)
        for num, dirs in seen.items():
            if len(dirs) == 2:
                assert dirs[0] == "across"
                assert dirs[1] == "down"

    def test_entry_positions(self) -> None:
        entries = compute_numbering(GRID_5X5)
        # First entry should be at (0,0)
        first = entries[0]
        assert first.row == 0
        assert first.col == 0

    def test_empty_grid(self) -> None:
        assert compute_numbering([]) == []
        assert compute_numbering([[]]) == []

    def test_single_cell(self) -> None:
        # Single cell can't form a word (min length 2)
        entries = compute_numbering([["A"]])
        assert entries == []

    def test_two_cells_across(self) -> None:
        entries = compute_numbering([["A", "B"]])
        assert len(entries) == 1
        assert entries[0].direction == "across"
        assert entries[0].answer == "AB"

    def test_two_cells_down(self) -> None:
        entries = compute_numbering([["A"], ["B"]])
        assert len(entries) == 1
        assert entries[0].direction == "down"
        assert entries[0].answer == "AB"
