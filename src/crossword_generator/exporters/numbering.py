"""Crossword clue numbering utility."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NumberedEntry:
    """A numbered clue entry in the crossword grid."""

    number: int
    direction: str  # "across" or "down"
    row: int
    col: int
    length: int
    answer: str


def compute_numbering(
    grid: list[list[str]], *, black: str = "."
) -> list[NumberedEntry]:
    """Compute standard American crossword numbering for a grid.

    Scans left-to-right, top-to-bottom. A cell gets a number if it starts
    an across word (no letter to its left) or a down word (no letter above).
    Minimum word length is 2.

    Args:
        grid: 2D grid with uppercase letters and black-square sentinel.
        black: The black-square sentinel character (default ".").

    Returns:
        List of NumberedEntry sorted by number then direction (across before down).
    """
    if not grid or not grid[0]:
        return []

    rows = len(grid)
    cols = len(grid[0])
    entries: list[NumberedEntry] = []
    current_number = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == black:
                continue

            starts_across = False
            starts_down = False

            # Starts across: no letter to the left, and at least one letter to the right
            if (c == 0 or grid[r][c - 1] == black) and (
                c + 1 < cols and grid[r][c + 1] != black
            ):
                starts_across = True

            # Starts down: no letter above, and at least one letter below
            if (r == 0 or grid[r - 1][c] == black) and (
                r + 1 < rows and grid[r + 1][c] != black
            ):
                starts_down = True

            if not starts_across and not starts_down:
                continue

            current_number += 1

            if starts_across:
                # Measure across word length
                length = 0
                answer_chars: list[str] = []
                cc = c
                while cc < cols and grid[r][cc] != black:
                    answer_chars.append(grid[r][cc])
                    length += 1
                    cc += 1
                entries.append(
                    NumberedEntry(
                        number=current_number,
                        direction="across",
                        row=r,
                        col=c,
                        length=length,
                        answer="".join(answer_chars),
                    )
                )

            if starts_down:
                # Measure down word length
                length = 0
                answer_chars = []
                rr = r
                while rr < rows and grid[rr][c] != black:
                    answer_chars.append(grid[rr][c])
                    length += 1
                    rr += 1
                entries.append(
                    NumberedEntry(
                        number=current_number,
                        direction="down",
                        row=r,
                        col=c,
                        length=length,
                        answer="".join(answer_chars),
                    )
                )

    return entries


def compute_crossing_words(
    entries: list[NumberedEntry], grid: list[list[str]]
) -> dict[tuple[int, str], list[str]]:
    """Compute which words cross each entry in the grid.

    For each entry, find all entries that share at least one cell.
    Returns a mapping of (number, direction) -> list of crossing answer words.
    """
    # Build a cell -> entry index for fast lookup
    cell_to_entries: dict[tuple[int, int], list[int]] = {}
    for idx, entry in enumerate(entries):
        if entry.direction == "across":
            for offset in range(entry.length):
                cell = (entry.row, entry.col + offset)
                cell_to_entries.setdefault(cell, []).append(idx)
        else:  # down
            for offset in range(entry.length):
                cell = (entry.row + offset, entry.col)
                cell_to_entries.setdefault(cell, []).append(idx)

    # For each entry, collect crossing entry answers
    crossing_words: dict[tuple[int, str], list[str]] = {}
    for idx, entry in enumerate(entries):
        crossings: list[str] = []
        if entry.direction == "across":
            cells = [(entry.row, entry.col + offset) for offset in range(entry.length)]
        else:
            cells = [(entry.row + offset, entry.col) for offset in range(entry.length)]

        seen: set[int] = set()
        for cell in cells:
            for other_idx in cell_to_entries.get(cell, []):
                if other_idx != idx and other_idx not in seen:
                    seen.add(other_idx)
                    crossings.append(entries[other_idx].answer)

        crossing_words[(entry.number, entry.direction)] = crossings

    return crossing_words
