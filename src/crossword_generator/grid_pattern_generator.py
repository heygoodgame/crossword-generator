"""Programmatic grid pattern generator for midi crosswords.

Generates thousands of unique, structurally valid black-cell patterns
using seed-based symmetric random placement with constraint validation.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass
class PatternConfig:
    """Configuration for pattern generation."""

    min_density: float = 0.12
    max_density: float = 0.25
    min_word_length: int = 3
    interior_to_edge_ratio: int = 3  # 3:1 interior-to-edge


@dataclass
class PatternReport:
    """Diagnostic info about a generated pattern."""

    rows: int
    cols: int
    black_cell_count: int
    density: float
    is_symmetric: bool
    is_connected: bool
    min_word_length_found: int
    has_2x2_block: bool
    all_rows_cols_have_white: bool
    corners_white: bool
    valid: bool


def generate_pattern(
    rows: int,
    cols: int,
    *,
    seed: int = 0,
    config: PatternConfig | None = None,
    locked_white: set[tuple[int, int]] | None = None,
    locked_black: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Generate a black-cell pattern for a grid.

    Uses seeded RNG for deterministic, reproducible results.
    Different seeds produce different patterns.

    Args:
        rows: Grid height.
        cols: Grid width.
        seed: RNG seed for reproducibility.
        config: Optional generation parameters.
        locked_white: Positions that must remain white (e.g. theme entry cells).
        locked_black: Positions that must be black (e.g. theme delimiters).

    Returns:
        List of (row, col) positions of black cells.
    """
    if config is None:
        config = PatternConfig()

    locked_white = locked_white or set()
    locked_black = locked_black or set()

    rng = random.Random(seed)
    total_cells = rows * cols
    max_black = int(total_cells * config.max_density)

    half_positions = _get_half_positions(
        rows, cols, locked_white=locked_white, locked_black=locked_black
    )
    n_pos = len(half_positions)

    # Step 1: Per-cell independent inclusion decision.
    # Each cell uses its own RNG keyed by (seed, position_index)
    # for truly independent decisions across seeds.
    target_factor = rng.uniform(0.25, 0.65)
    desired: list[tuple[int, int]] = []
    remainder: list[tuple[int, int]] = []
    for i, pos in enumerate(half_positions):
        cell_rng = random.Random(seed * n_pos + i)
        if cell_rng.random() < target_factor:
            desired.append(pos)
        else:
            remainder.append(pos)

    rng.shuffle(desired)
    rng.shuffle(remainder)

    # Step 2: Build pattern from desired cells, then fill from remainder.
    # Start with locked-black cells already placed.
    black: set[tuple[int, int]] = set(locked_black)

    for candidates in (desired, remainder):
        for r, c in candidates:
            if len(black) >= max_black:
                break

            mirror_r, mirror_c = rows - 1 - r, cols - 1 - c
            if (r, c) in black:
                continue

            is_center = (r, c) == (mirror_r, mirror_c)
            new_cells = (
                [(r, c)] if is_center else [(r, c), (mirror_r, mirror_c)]
            )

            if len(black) + len(new_cells) > max_black:
                continue

            trial = black | set(new_cells)

            # Hard constraint checks
            if _has_2x2_block_any(rows, cols, trial, new_cells):
                continue
            if not _check_min_word_length(
                rows, cols, trial, config.min_word_length
            ):
                continue
            if not _all_rows_cols_have_white(rows, cols, trial):
                continue
            if not _is_connected(rows, cols, trial):
                continue

            black = trial

    return sorted(black)


def analyze_pattern(
    rows: int,
    cols: int,
    black_cells: list[tuple[int, int]],
) -> PatternReport:
    """Analyze a pattern and return diagnostic info."""
    black = set(black_cells)
    total = rows * cols
    count = len(black_cells)

    # Check symmetry
    is_symmetric = all(
        (rows - 1 - r, cols - 1 - c) in black for r, c in black
    )

    # Find min word length
    min_wl = _find_min_word_length(rows, cols, black)

    # Check corners
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    corners_white = all(c not in black for c in corners)

    has_2x2 = _has_any_2x2_block(rows, cols, black)

    return PatternReport(
        rows=rows,
        cols=cols,
        black_cell_count=count,
        density=count / total if total > 0 else 0.0,
        is_symmetric=is_symmetric,
        is_connected=_is_connected(rows, cols, black),
        min_word_length_found=min_wl,
        has_2x2_block=has_2x2,
        all_rows_cols_have_white=_all_rows_cols_have_white(rows, cols, black),
        corners_white=corners_white,
        valid=(
            is_symmetric
            and _is_connected(rows, cols, black)
            and min_wl >= 3
            and not has_2x2
            and _all_rows_cols_have_white(rows, cols, black)
            and corners_white
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_connected(rows: int, cols: int, black: set[tuple[int, int]]) -> bool:
    """Check that all white cells form a single connected component via BFS."""
    # Find first white cell
    start = None
    white_count = 0
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in black:
                white_count += 1
                if start is None:
                    start = (r, c)

    if start is None:
        return white_count == 0  # All black is technically "connected"

    # BFS
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([start])
    visited.add(start)

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and (nr, nc) not in black
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return len(visited) == white_count


def _check_min_word_length(
    rows: int,
    cols: int,
    black: set[tuple[int, int]],
    min_len: int,
) -> bool:
    """Return True if all word slots are >= min_len letters."""
    # Check across
    for r in range(rows):
        length = 0
        for c in range(cols):
            if (r, c) in black:
                if 0 < length < min_len:
                    return False
                length = 0
            else:
                length += 1
        if 0 < length < min_len:
            return False

    # Check down
    for c in range(cols):
        length = 0
        for r in range(rows):
            if (r, c) in black:
                if 0 < length < min_len:
                    return False
                length = 0
            else:
                length += 1
        if 0 < length < min_len:
            return False

    return True


def _find_min_word_length(
    rows: int, cols: int, black: set[tuple[int, int]]
) -> int:
    """Find the minimum word slot length in the grid."""
    min_len = max(rows, cols)  # Start with max possible

    for r in range(rows):
        length = 0
        for c in range(cols):
            if (r, c) in black:
                if length > 0:
                    min_len = min(min_len, length)
                length = 0
            else:
                length += 1
        if length > 0:
            min_len = min(min_len, length)

    for c in range(cols):
        length = 0
        for r in range(rows):
            if (r, c) in black:
                if length > 0:
                    min_len = min(min_len, length)
                length = 0
            else:
                length += 1
        if length > 0:
            min_len = min(min_len, length)

    return min_len


def _has_2x2_block(
    black: set[tuple[int, int]], r: int, c: int
) -> bool:
    """Check if placing a black cell at (r,c) creates any 2x2 all-black block."""
    # Check all four 2x2 squares that include (r, c)
    for dr in (-1, 0):
        for dc in (-1, 0):
            tr, tc = r + dr, c + dc
            if all(
                (tr + ir, tc + ic) in black
                for ir in range(2)
                for ic in range(2)
            ):
                return True
    return False


def _has_2x2_block_any(
    rows: int,
    cols: int,
    black: set[tuple[int, int]],
    new_cells: list[tuple[int, int]],
) -> bool:
    """Check if any newly placed cells create a 2x2 block."""
    for r, c in new_cells:
        if _has_2x2_block(black, r, c):
            return True
    return False


def _has_any_2x2_block(
    rows: int, cols: int, black: set[tuple[int, int]]
) -> bool:
    """Check if the grid has any 2x2 all-black block."""
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(
                (r + dr, c + dc) in black for dr in range(2) for dc in range(2)
            ):
                return True
    return False


def _all_rows_cols_have_white(
    rows: int, cols: int, black: set[tuple[int, int]]
) -> bool:
    """Check that no row or column is entirely black."""
    for r in range(rows):
        if all((r, c) in black for c in range(cols)):
            return False
    for c in range(cols):
        if all((r, c) in black for r in range(rows)):
            return False
    return True


def _get_half_positions(
    rows: int,
    cols: int,
    *,
    locked_white: set[tuple[int, int]] | None = None,
    locked_black: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Get all candidate half-positions (excluding corners and locked cells).

    Returns one of each symmetric pair (lexicographically smaller),
    plus the center cell for odd grids. Skips positions that are locked
    (either white or black) since they cannot be toggled.
    """
    corners = {(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)}
    locked = (locked_white or set()) | (locked_black or set())
    positions: list[tuple[int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if (r, c) in corners:
                continue
            if (r, c) in locked:
                continue
            mirror_r, mirror_c = rows - 1 - r, cols - 1 - c
            if (mirror_r, mirror_c) in locked:
                continue
            # Center cell
            if (r, c) == (mirror_r, mirror_c):
                positions.append((r, c))
            # Only the lexicographically smaller of the pair
            elif (r, c) < (mirror_r, mirror_c):
                positions.append((r, c))

    return positions


def _build_candidates(
    rows: int,
    cols: int,
    rng: random.Random,
    config: PatternConfig,
) -> list[tuple[int, int]]:
    """Build symmetry-aware candidate list with interior bias.

    Only generates "half" positions for 180° symmetry.
    Interior cells appear multiple times to bias placement inward.
    Corners are excluded.
    """
    corners = {(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)}
    edge_band = 2  # First/last 2 rows/cols are "edge"

    result: list[tuple[int, int]] = []
    center_cell: tuple[int, int] | None = None

    for r in range(rows):
        for c in range(cols):
            if (r, c) in corners:
                continue

            mirror_r, mirror_c = rows - 1 - r, cols - 1 - c

            # Center cell of odd grid
            if (r, c) == (mirror_r, mirror_c):
                center_cell = (r, c)
                continue

            # Only take the "first half" — lexicographic ordering
            if (r, c) > (mirror_r, mirror_c):
                continue

            is_edge = (
                r < edge_band
                or r >= rows - edge_band
                or c < edge_band
                or c >= cols - edge_band
            )

            # Interior cells added multiple times for bias
            copies = 1 if is_edge else config.interior_to_edge_ratio
            for _ in range(copies):
                result.append((r, c))

    # Single shuffle of all candidates (duplicates included)
    rng.shuffle(result)

    # Insert center cell at random position
    if center_cell is not None:
        pos = rng.randint(0, len(result))
        result.insert(pos, center_cell)

    return result
