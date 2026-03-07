"""Grid specification catalog for supported puzzle types and sizes."""

from __future__ import annotations

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


def get_grid_spec(puzzle_type: PuzzleType | str, grid_size: int) -> GridSpec:
    """Return a GridSpec for the given puzzle type and size.

    Args:
        puzzle_type: "mini" or "midi" (or PuzzleType enum).
        grid_size: Grid dimension (e.g., 5, 7, 9, 10, 11).

    Returns:
        GridSpec with the appropriate rows and cols.

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
    return GridSpec(rows=rows, cols=cols)
