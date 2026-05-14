"""Validation helpers for structured crossword grid patterns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class GridPatternValidation:
    """Validation result for one weighted black-cell pattern."""

    size: int
    index: int
    weight: int
    black_cells: tuple[tuple[int, int], ...]
    symmetric: bool
    connected: bool
    min_slot_length: int
    valid: bool
    errors: tuple[str, ...]


def validate_pattern(
    size: int,
    black_cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    weight: int = 1,
    index: int = 0,
    min_slot_length: int = 3,
) -> GridPatternValidation:
    """Validate one square crossword black-cell pattern."""
    normalized = tuple(sorted(black_cells))
    black = set(normalized)
    errors: list[str] = []

    if size <= 0:
        errors.append("size must be positive")

    if weight <= 0:
        errors.append("weight must be positive")

    if len(black) != len(normalized):
        errors.append("duplicate black-cell coordinates")

    for r, c in normalized:
        if r < 0 or c < 0 or r >= size or c >= size:
            errors.append(f"black cell out of bounds: {(r, c)}")

    connected = _white_cells_connected(size, black)
    if not connected:
        errors.append("white cells are not connected")

    min_found = _minimum_slot_length(size, black)
    if 0 < min_found < min_slot_length:
        errors.append(
            f"short slot length {min_found}; expected at least {min_slot_length}"
        )

    symmetric = is_rotationally_symmetric(size, normalized)

    return GridPatternValidation(
        size=size,
        index=index,
        weight=weight,
        black_cells=normalized,
        symmetric=symmetric,
        connected=connected,
        min_slot_length=min_found,
        valid=not errors,
        errors=tuple(errors),
    )


def validate_weighted_patterns(
    size: int,
    patterns: list[tuple[list[tuple[int, int]], int]]
    | tuple[tuple[list[tuple[int, int]], int], ...],
) -> list[GridPatternValidation]:
    """Validate all weighted patterns for one grid size."""
    return [
        validate_pattern(size, black_cells, weight=weight, index=index)
        for index, (black_cells, weight) in enumerate(patterns, start=1)
    ]


def is_rotationally_symmetric(
    size: int,
    black_cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bool:
    """Return whether the pattern has 180-degree rotational symmetry."""
    black = set(black_cells)
    return all((size - 1 - r, size - 1 - c) in black for r, c in black)


def summarize_validations(results: list[GridPatternValidation]) -> dict[str, int]:
    """Return compact counts for a set of pattern validation results."""
    return {
        "patterns": len(results),
        "total_weight": sum(result.weight for result in results),
        "valid": sum(1 for result in results if result.valid),
        "invalid": sum(1 for result in results if not result.valid),
        "symmetric": sum(1 for result in results if result.symmetric),
        "asymmetric": sum(1 for result in results if not result.symmetric),
    }


def _white_cells_connected(size: int, black: set[tuple[int, int]]) -> bool:
    start: tuple[int, int] | None = None
    white_count = 0
    for r in range(size):
        for c in range(size):
            if (r, c) not in black:
                white_count += 1
                if start is None:
                    start = (r, c)

    if start is None:
        return False

    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if (
                0 <= nxt[0] < size
                and 0 <= nxt[1] < size
                and nxt not in black
                and nxt not in seen
            ):
                seen.add(nxt)
                queue.append(nxt)

    return len(seen) == white_count


def _minimum_slot_length(size: int, black: set[tuple[int, int]]) -> int:
    lengths: list[int] = []

    for r in range(size):
        run = 0
        for c in range(size):
            if (r, c) in black:
                if run:
                    lengths.append(run)
                run = 0
            else:
                run += 1
        if run:
            lengths.append(run)

    for c in range(size):
        run = 0
        for r in range(size):
            if (r, c) in black:
                if run:
                    lengths.append(run)
                run = 0
            else:
                run += 1
        if run:
            lengths.append(run)

    return min(lengths) if lengths else 0
