"""Grid specification catalog for supported puzzle types and sizes."""

from __future__ import annotations

import random

from crossword_generator.fillers.base import GridSpec
from crossword_generator.models import PuzzleType

# Valid (puzzle_type, grid_size) combinations
_VALID_SPECS: dict[tuple[PuzzleType, int], tuple[int, int]] = {
    (PuzzleType.MINI, 5): (5, 5),
    (PuzzleType.MINI, 7): (7, 7),
    (PuzzleType.MIDI, 9): (9, 9),
    (PuzzleType.MIDI, 10): (10, 10),
    (PuzzleType.MIDI, 11): (11, 11),
}

# Black cell patterns for each (puzzle_type, grid_size).
# Each pattern is a list of (row, col) positions for black cells.
# Mini puzzles use black cells to break up the constraint graph and make CSP
# filling tractable, following real NYT mini conventions.
_GRID_PATTERNS: dict[tuple[PuzzleType, int], list[list[tuple[int, int]]]] = {
    # 5x5: only corner positions avoid creating sub-3-letter slots.
    (PuzzleType.MINI, 5): [
        # Pattern 0: two opposing corners
        [(0, 4), (4, 0)],
        # Pattern 1: other diagonal pair
        [(0, 0), (4, 4)],
        # Pattern 2: all four corners (all slots = 3)
        [(0, 0), (0, 4), (4, 0), (4, 4)],
    ],
    # 7x7: 8-12 black cells, matching real NYT mini conventions.
    # Corner L-shapes and edge-midpoint pairs keep all slots >= 3.
    (PuzzleType.MINI, 7): [
        # Pattern 0: corner L-shapes at opposite corners, 8 black cells
        [(0, 0), (0, 1), (0, 6), (1, 0), (5, 6), (6, 0), (6, 5), (6, 6)],
        # Pattern 1: cross — paired bars at edge midpoints, 8 black cells
        [(0, 3), (1, 3), (3, 0), (3, 1), (3, 5), (3, 6), (5, 3), (6, 3)],
        # Pattern 2: extended corner pairs, 10 black cells
        [
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (5, 6), (6, 0), (6, 1), (6, 5), (6, 6),
        ],
        # Pattern 3: full corner blocks, 12 black cells
        [
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0), (1, 6),
            (5, 0), (5, 6), (6, 0), (6, 1), (6, 5), (6, 6),
        ],
    ],
}


def get_grid_spec(
    puzzle_type: PuzzleType | str,
    grid_size: int,
    *,
    seed: int | None = None,
) -> GridSpec:
    """Return a GridSpec for the given puzzle type and size.

    Args:
        puzzle_type: "mini" or "midi" (or PuzzleType enum).
        grid_size: Grid dimension (e.g., 5, 7, 9, 10, 11).
        seed: Optional seed to randomly select a black cell pattern.
              When None, uses the first pattern as default.

    Returns:
        GridSpec with the appropriate rows, cols, and black cells.

    Raises:
        ValueError: If the puzzle_type/grid_size combination is not supported.
    """
    pt = PuzzleType(puzzle_type)
    key = (pt, grid_size)

    if key not in _VALID_SPECS:
        valid = [f"{t.value}/{s}" for (t, s) in _VALID_SPECS]
        raise ValueError(
            f"Unsupported puzzle_type/grid_size: {pt.value}/{grid_size}. "
            f"Valid combinations: {', '.join(valid)}"
        )

    rows, cols = _VALID_SPECS[key]

    patterns = _GRID_PATTERNS.get(key)
    if patterns:
        if seed is not None:
            rng = random.Random(seed)
            black_cells = rng.choice(patterns)
        else:
            black_cells = patterns[0]
    else:
        black_cells = []

    return GridSpec(rows=rows, cols=cols, black_cells=black_cells)
