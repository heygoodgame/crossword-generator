"""Parser for go-crossword output formats (compact text and JSON)."""

from __future__ import annotations

import json
import re

from crossword_generator.fillers.base import FilledGrid

# U+2588 FULL BLOCK — used by go-crossword for black squares
_BLACK_CHAR = "\u2588"

# Black-square sentinel in our grid representation (matches puzpy convention)
BLACK = "."

# ANSI escape code pattern
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class ParseError(Exception):
    """Raised when go-crossword output cannot be parsed."""


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _extract_grid_lines(lines: list[str]) -> list[str]:
    """Extract the grid lines from between header/footer markers."""
    grid_lines: list[str] = []
    in_grid = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_grid:
                # Empty line after grid started means grid ended
                break
            continue

        # Skip known header/footer lines
        if stripped.startswith("Generating") or stripped.startswith(
            "Crossword generated"
        ):
            if in_grid:
                break
            continue
        if stripped.startswith("Seed:"):
            break

        # A grid line contains letters and/or black squares separated by spaces
        cells = stripped.split()
        if all((len(c) == 1 and c.isalpha()) or c == _BLACK_CHAR for c in cells):
            in_grid = True
            grid_lines.append(stripped)
        elif in_grid:
            # Non-grid line after grid started
            break

    return grid_lines


def _extract_words(grid: list[list[str]], direction: str) -> list[str]:
    """Extract across or down words from a grid."""
    words: list[str] = []
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if direction == "across":
        for r in range(rows):
            current: list[str] = []
            for c in range(cols):
                if grid[r][c] != BLACK:
                    current.append(grid[r][c])
                else:
                    if len(current) >= 2:
                        words.append("".join(current))
                    current = []
            if len(current) >= 2:
                words.append("".join(current))
    elif direction == "down":
        for c in range(cols):
            current = []
            for r in range(rows):
                if grid[r][c] != BLACK:
                    current.append(grid[r][c])
                else:
                    if len(current) >= 2:
                        words.append("".join(current))
                    current = []
            if len(current) >= 2:
                words.append("".join(current))

    return words


def parse_compact_output(raw: str) -> FilledGrid:
    """Parse go-crossword compact output into a FilledGrid.

    Args:
        raw: Raw stdout from go-crossword with -compact flag.

    Returns:
        FilledGrid with uppercase letters and "." for black squares.

    Raises:
        ParseError: If the output cannot be parsed into a valid grid.
    """
    cleaned = _strip_ansi(raw)
    lines = cleaned.splitlines()

    grid_lines = _extract_grid_lines(lines)
    if not grid_lines:
        raise ParseError(f"No grid found in output:\n{raw[:500]}")

    # Parse grid cells
    grid: list[list[str]] = []
    expected_cols: int | None = None

    for i, line in enumerate(grid_lines):
        cells = line.split()
        row: list[str] = []
        for cell in cells:
            if cell == _BLACK_CHAR:
                row.append(BLACK)
            elif len(cell) == 1 and cell.isalpha():
                row.append(cell.upper())
            else:
                raise ParseError(f"Unexpected cell value {cell!r} on grid line {i + 1}")
        if expected_cols is None:
            expected_cols = len(row)
        elif len(row) != expected_cols:
            raise ParseError(
                f"Inconsistent row width: expected {expected_cols}, "
                f"got {len(row)} on line {i + 1}"
            )
        grid.append(row)

    words_across = _extract_words(grid, "across")
    words_down = _extract_words(grid, "down")

    return FilledGrid(
        grid=grid,
        words_across=words_across,
        words_down=words_down,
    )


def parse_json_output(raw: str) -> FilledGrid:
    """Parse go-crossword JSON output into a FilledGrid.

    Args:
        raw: Raw stdout from go-crossword with -format json flag.

    Returns:
        FilledGrid with uppercase letters and "." for black squares.

    Raises:
        ParseError: If the output cannot be parsed into a valid grid.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON output: {e}") from e

    for field in ("grid", "rows", "cols"):
        if field not in data:
            raise ParseError(f"Missing required field: {field!r}")

    rows = data["rows"]
    cols = data["cols"]
    raw_grid = data["grid"]

    if len(raw_grid) != rows:
        raise ParseError(
            f"Grid row count mismatch: expected {rows}, got {len(raw_grid)}"
        )

    grid: list[list[str]] = []
    for i, raw_row in enumerate(raw_grid):
        if len(raw_row) != cols:
            raise ParseError(
                f"Grid column count mismatch on row {i}: "
                f"expected {cols}, got {len(raw_row)}"
            )
        row: list[str] = []
        for cell in raw_row:
            if cell == BLACK:
                row.append(BLACK)
            else:
                row.append(cell.upper())
        grid.append(row)

    words_across = [entry["word"].upper() for entry in data.get("words_across", [])]
    words_down = [entry["word"].upper() for entry in data.get("words_down", [])]

    return FilledGrid(
        grid=grid,
        words_across=words_across,
        words_down=words_down,
    )
