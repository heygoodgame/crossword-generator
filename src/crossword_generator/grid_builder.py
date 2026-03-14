"""Theme-first grid construction.

Places theme entries first, then builds a valid black-cell pattern around
them. This inverts the default pipeline order (random grid -> fit theme)
to match the standard professional crossword construction workflow.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from crossword_generator.fillers.base import GridSpec
from crossword_generator.grid_pattern_generator import (
    PatternConfig,
    _all_rows_cols_have_white,
    _check_min_word_length,
    _has_2x2_block,
    _has_any_2x2_block,
    _is_connected,
    generate_pattern,
)

logger = logging.getLogger(__name__)

_MAX_PLACEMENT_NODES = 50_000
_MAX_CROSSING_DEPTH = 2


@dataclass
class LinePartition:
    """A way to place an entry of a given length along a line (row or column).

    For across placement, ``start`` is the column index and
    ``black_positions`` are column indices.  For down placement, ``start``
    is the row index and ``black_positions`` are row indices.

    Attributes:
        start: Position along the line where the entry begins.
        black_positions: Positions along the line that must be black.
        remaining_slot_lengths: Lengths of non-entry slots created.
    """

    start: int
    black_positions: list[int]
    remaining_slot_lengths: list[int]


def _valid_line_partitions(
    grid_size: int, entry_length: int
) -> list[LinePartition]:
    """Return all ways to place an entry of given length along a line.

    Each placement ensures remaining white segments are either 0 or >= 3
    letters (no 1- or 2-letter orphan slots). Black cells serve as
    delimiters between the entry and other segments.

    Args:
        grid_size: Length of the line (row width or column height).
        entry_length: Length of the entry to place.

    Returns:
        List of valid LinePartition options.
    """
    if entry_length > grid_size:
        return []
    if entry_length == grid_size:
        return [LinePartition(
            start=0, black_positions=[], remaining_slot_lengths=[],
        )]

    partitions: list[LinePartition] = []

    for start in range(grid_size - entry_length + 1):
        end = start + entry_length  # exclusive
        blacks: list[int] = []
        remaining: list[int] = []

        # --- Before entry: positions [0, start) ---
        if start > 0:
            # Must have a black delimiter at start-1
            blacks.append(start - 1)
            # Remaining segment: [0, start-1)
            left_len = start - 1
            if left_len == 0:
                pass  # just the delimiter, no orphan
            elif left_len < 3:
                # Orphan too short -- make those cells black too
                blacks.extend(range(left_len))
            else:
                remaining.append(left_len)

        # --- After entry: positions [end, grid_size) ---
        if end < grid_size:
            # Must have a black delimiter at end
            blacks.append(end)
            # Remaining segment: [end+1, grid_size)
            right_len = grid_size - end - 1
            if right_len == 0:
                pass  # just the delimiter
            elif right_len < 3:
                # Orphan too short -- make those cells black too
                blacks.extend(range(end + 1, grid_size))
            else:
                remaining.append(right_len)

        # Validate: entry cells must not be black
        black_set = set(blacks)
        if any(p in black_set for p in range(start, end)):
            continue

        partitions.append(
            LinePartition(
                start=start,
                black_positions=sorted(black_set),
                remaining_slot_lengths=remaining,
            )
        )

    return partitions


def _symmetric_row(row: int, grid_size: int) -> int:
    """Return the 180-degree-symmetric partner row."""
    return grid_size - 1 - row


def build_themed_grids(
    grid_size: int,
    entries: list[str],
    revealer: str,
    *,
    seed: int = 0,
    count: int = 10,
) -> list[GridSpec]:
    """Build grid patterns with theme entries pre-placed.

    Places theme entries in across or down slots, then generates black
    cells around those placements to form a valid crossword grid.
    Across placement is preferred (standard convention); down is used
    as a fallback when across slots are exhausted.

    Args:
        grid_size: Grid dimension (e.g. 9 for 9x9).
        entries: Theme entry words (uppercase).
        revealer: Revealer word (uppercase).
        seed: Base RNG seed; incremented for each variant.
        count: Number of grid variants to attempt.

    Returns:
        List of GridSpec objects with theme entries in seed_entries.
        May be empty if no valid placement is found.
    """
    all_words = list(entries)
    if revealer:
        all_words.append(revealer)

    if not all_words:
        return []

    # Filter out words that can't fit
    all_words = [w for w in all_words if len(w) <= grid_size]
    if not all_words:
        return []

    # Sort by length descending (most constrained first)
    all_words.sort(key=len, reverse=True)

    # Precompute valid partitions per entry length
    partition_cache: dict[int, list[LinePartition]] = {}
    for w in all_words:
        wlen = len(w)
        if wlen not in partition_cache:
            partition_cache[wlen] = _valid_line_partitions(grid_size, wlen)
            if not partition_cache[wlen]:
                logger.debug(
                    "No valid partitions for length %d in %dx%d grid",
                    wlen, grid_size, grid_size,
                )
                return []

    # Symmetric index pairs — used for both row pairs (across) and
    # column pairs (down).  For odd grids the center pairs with itself.
    sym_pairs: list[tuple[int, int]] = []
    for i in range(grid_size // 2):
        sym_pairs.append((i, grid_size - 1 - i))
    if grid_size % 2 == 1:
        sym_pairs.append((grid_size // 2, grid_size // 2))

    # Individual indices for asymmetric fallback — each row/column
    # is independent (no forced mirror placement).
    individual_pairs = [(idx, idx) for idx in range(grid_size)]

    results: list[GridSpec] = []

    for variant in range(count):
        rng = random.Random(seed + variant)

        # Phase 1: Try symmetric placement
        spec = _try_place_entries(
            grid_size, all_words, sym_pairs, partition_cache, rng,
            symmetric=True,
        )
        # Phase 2: Asymmetric fallback — entries placed on individual
        # rows without forcing mirror cells, giving the CSP filler
        # more degrees of freedom.
        if spec is None:
            rng2 = random.Random(seed + variant)
            spec = _try_place_entries(
                grid_size, all_words, individual_pairs, partition_cache,
                rng2, symmetric=False,
            )
        if spec is not None:
            results.append(spec)

    logger.info(
        "build_themed_grids: %d/%d variants succeeded for %d entries "
        "in %dx%d grid",
        len(results), count, len(all_words), grid_size, grid_size,
    )
    return results


def _has_unsealable_gap(
    grid_size: int,
    locked_white: set[tuple[int, int]],
    locked_black: set[tuple[int, int]],
    *,
    symmetric: bool = True,
) -> bool:
    """Quick check: any 1-2 cell gap between blacks has a locked-white cell.

    If a short gap contains a cell that is locked white (or, when symmetric,
    whose 180-degree mirror is locked white), the gap cannot be sealed by
    _seal_short_gaps.  Used as a forward-pruning check during backtracking.
    """
    for axis in ("row", "col"):
        for idx in range(grid_size):
            cells = (
                [(idx, c) for c in range(grid_size)]
                if axis == "row"
                else [(r, idx) for r in range(grid_size)]
            )
            i = 0
            while i < grid_size:
                if cells[i] in locked_black:
                    i += 1
                    continue
                run_start = i
                while i < grid_size and cells[i] not in locked_black:
                    i += 1
                if 1 <= (i - run_start) <= 2:
                    for j in range(run_start, i):
                        cell = cells[j]
                        if cell in locked_white:
                            return True
                        if symmetric:
                            mir = (
                                grid_size - 1 - cell[0],
                                grid_size - 1 - cell[1],
                            )
                            if mir in locked_white:
                                return True
    return False


def _compute_placement_cells(
    direction: str,
    line_a: int,
    line_b: int,
    part: LinePartition,
    wlen: int,
    grid_size: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Compute the white and black cells for a candidate placement.

    Returns (all_whites, all_blacks) including 180-degree mirror cells.
    """
    if direction == "across":
        blacks_a = {(line_a, c) for c in part.black_positions}
        whites_a = {
            (line_a, c) for c in range(part.start, part.start + wlen)
        }
        blacks_b: set[tuple[int, int]] = set()
        whites_b: set[tuple[int, int]] = set()
        if line_a != line_b:
            for _, c in blacks_a:
                blacks_b.add((line_b, grid_size - 1 - c))
            for _, c in whites_a:
                whites_b.add((line_b, grid_size - 1 - c))
    else:
        blacks_a = {(r, line_a) for r in part.black_positions}
        whites_a = {
            (part.start + i, line_a) for i in range(wlen)
        }
        blacks_b = set()
        whites_b = set()
        if line_a != line_b:
            for r, _ in blacks_a:
                blacks_b.add((grid_size - 1 - r, line_b))
            for r, _ in whites_a:
                whites_b.add((grid_size - 1 - r, line_b))

    return whites_a | whites_b, blacks_a | blacks_b


def _try_place_entries(
    grid_size: int,
    words: list[str],
    line_pairs: list[tuple[int, int]],
    partition_cache: dict[int, list[LinePartition]],
    rng: random.Random,
    *,
    symmetric: bool = True,
) -> GridSpec | None:
    """Try to place all words using backtracking search.

    Uses recursive backtracking to explore placement combinations.
    Across placements are tried before down (preserving convention),
    but if a later word can't be placed, earlier choices are revised.

    Args:
        line_pairs: Row/column pairs for placement. In symmetric mode
            these are (i, grid_size-1-i) pairs; in asymmetric mode
            they are (i, i) so _compute_placement_cells skips mirroring.
        symmetric: Whether to enforce symmetric gap sealing and pattern
            generation. Asymmetric mode gives the CSP filler more freedom.

    Returns a GridSpec on success, None on failure.
    """
    locked_white: set[tuple[int, int]] = set()
    locked_black: set[tuple[int, int]] = set()
    seed_entries: dict[str, str] = {}
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    across_per_col: list[int] = [0] * grid_size
    down_per_row: list[int] = [0] * grid_size

    # Pre-build shuffled candidate lists for each word.
    # Interleave across/down for each (pair, partition) combo so
    # backtracking explores mixed-direction placements early.
    candidates_per_word: list[
        list[tuple[str, int, int, LinePartition]]
    ] = []
    for word in words:
        partitions = partition_cache[len(word)]
        shuffled_pairs = list(line_pairs)
        rng.shuffle(shuffled_pairs)
        shuffled_parts = list(partitions)
        rng.shuffle(shuffled_parts)

        candidates: list[tuple[str, int, int, LinePartition]] = []
        for pair in shuffled_pairs:
            for part in shuffled_parts:
                candidates.append(("across", pair[0], pair[1], part))
                candidates.append(("down", pair[0], pair[1], part))
        candidates_per_word.append(candidates)

    nodes = [0]
    # Holds (sealed_white, sealed_black) on successful placement + seal
    sealed_state: list[
        tuple[set[tuple[int, int]], set[tuple[int, int]]] | None
    ] = [None]

    def _backtrack(word_idx: int) -> bool:
        if word_idx == len(words):
            # All words placed — try to seal gaps on a copy
            lw_copy = set(locked_white)
            lb_copy = set(locked_black)
            if _seal_short_gaps(
                grid_size, lw_copy, lb_copy, symmetric=symmetric
            ):
                sealed_state[0] = (lw_copy, lb_copy)
                return True
            return False  # Keep searching for another arrangement

        nodes[0] += 1
        if nodes[0] > _MAX_PLACEMENT_NODES:
            return False

        word = words[word_idx]
        wlen = len(word)

        for direction, line_a, line_b, part in candidates_per_word[word_idx]:
            if direction == "across" and line_a in used_rows:
                continue
            if direction == "down" and line_a in used_cols:
                continue

            all_whites, all_blacks = _compute_placement_cells(
                direction, line_a, line_b, part, wlen, grid_size,
            )

            # Conflict checks
            if all_blacks & locked_white:
                continue
            if all_whites & locked_black:
                continue
            if all_blacks & all_whites:
                continue

            trial_black = locked_black | all_blacks
            if any(
                _has_2x2_block(trial_black, r, c) for r, c in all_blacks
            ):
                continue

            # Crossing depth: limit how many seed entries cross any
            # single perpendicular line to keep the CSP feasible.
            if direction == "across":
                if any(
                    across_per_col[c] >= _MAX_CROSSING_DEPTH
                    for c in range(part.start, part.start + wlen)
                ):
                    continue
            else:
                if any(
                    down_per_row[r] >= _MAX_CROSSING_DEPTH
                    for r in range(part.start, part.start + wlen)
                ):
                    continue

            # Apply — track only the cells this placement actually adds
            added_w = all_whites - locked_white
            added_b = all_blacks - locked_black
            locked_white.update(added_w)
            locked_black.update(added_b)

            if direction == "across":
                for c in range(part.start, part.start + wlen):
                    across_per_col[c] += 1
            else:
                for r in range(part.start, part.start + wlen):
                    down_per_row[r] += 1

            # Forward check: prune if any short gap is unsealable
            if _has_unsealable_gap(
                grid_size, locked_white, locked_black,
                symmetric=symmetric,
            ):
                if direction == "across":
                    for c in range(part.start, part.start + wlen):
                        across_per_col[c] -= 1
                else:
                    for r in range(part.start, part.start + wlen):
                        down_per_row[r] -= 1
                locked_white.difference_update(added_w)
                locked_black.difference_update(added_b)
                continue

            if direction == "across":
                used_rows.add(line_a)
                if line_a != line_b:
                    used_rows.add(line_b)
                key = f"{line_a},{part.start},across"
            else:
                used_cols.add(line_a)
                if line_a != line_b:
                    used_cols.add(line_b)
                key = f"{part.start},{line_a},down"
            seed_entries[key] = word.upper()

            if _backtrack(word_idx + 1):
                return True

            # Undo
            locked_white.difference_update(added_w)
            locked_black.difference_update(added_b)
            if direction == "across":
                for c in range(part.start, part.start + wlen):
                    across_per_col[c] -= 1
                used_rows.discard(line_a)
                if line_a != line_b:
                    used_rows.discard(line_b)
            else:
                for r in range(part.start, part.start + wlen):
                    down_per_row[r] -= 1
                used_cols.discard(line_a)
                if line_a != line_b:
                    used_cols.discard(line_b)
            del seed_entries[key]

        return False

    if not _backtrack(0):
        logger.debug(
            "Backtracking exhausted (%d nodes) for %d words in %dx%d grid",
            nodes[0], len(words), grid_size, grid_size,
        )
        return None

    # Use the sealed locked cells from backtracking
    final_white, final_black = sealed_state[0]  # type: ignore[misc]

    # Phase B: Generate black cells around the placements
    black_cells = _generate_constrained_pattern(
        grid_size, final_white, final_black, rng, symmetric=symmetric
    )
    if black_cells is None:
        return None

    spec = GridSpec(
        rows=grid_size,
        cols=grid_size,
        black_cells=black_cells,
        seed_entries=seed_entries,
    )
    return spec


def _seal_short_gaps(
    grid_size: int,
    locked_white: set[tuple[int, int]],
    locked_black: set[tuple[int, int]],
    *,
    symmetric: bool = True,
) -> bool:
    """Seal 1- and 2-cell gaps in rows/columns between locked blacks.

    Only seals a gap if doing so does not create a 2x2 all-black block.
    When symmetric, seals both the gap cell and its 180-degree mirror.
    Returns True if all gaps were sealed successfully, False if any
    gap remains that cannot be sealed without creating 2x2 blocks.

    Modifies locked_white and locked_black in place.
    """
    changed = True
    while changed:
        changed = False
        for axis in ("col", "row"):
            limit = grid_size
            for idx in range(limit):
                cells = (
                    [(idx, c) for c in range(grid_size)]
                    if axis == "row"
                    else [(r, idx) for r in range(grid_size)]
                )
                i = 0
                while i < grid_size:
                    if cells[i] in locked_black:
                        i += 1
                        continue
                    run_start = i
                    while i < grid_size and cells[i] not in locked_black:
                        i += 1
                    run_len = i - run_start
                    if 1 <= run_len <= 2:
                        # Try sealing
                        can_seal = True
                        to_seal: list[tuple[int, int]] = []
                        for j in range(run_start, i):
                            cell = cells[j]
                            if symmetric:
                                mir = (
                                    grid_size - 1 - cell[0],
                                    grid_size - 1 - cell[1],
                                )
                                targets = [cell, mir]
                            else:
                                targets = [cell]
                            for c in targets:
                                if c in locked_white:
                                    can_seal = False
                                    break
                                trial = locked_black | set(to_seal) | {c}
                                if _has_2x2_block(trial, c[0], c[1]):
                                    can_seal = False
                                    break
                                to_seal.append(c)
                            if not can_seal:
                                break

                        if can_seal:
                            for c in to_seal:
                                locked_black.add(c)
                                locked_white.discard(c)
                            changed = True

    # Verify no short gaps remain
    for axis in ("col", "row"):
        for idx in range(grid_size):
            cells = (
                [(idx, c) for c in range(grid_size)]
                if axis == "row"
                else [(r, idx) for r in range(grid_size)]
            )
            i = 0
            while i < grid_size:
                if cells[i] in locked_black:
                    i += 1
                    continue
                run_start = i
                while i < grid_size and cells[i] not in locked_black:
                    i += 1
                if 1 <= (i - run_start) <= 2:
                    return False
    return True


def _generate_constrained_pattern(
    grid_size: int,
    locked_white: set[tuple[int, int]],
    locked_black: set[tuple[int, int]],
    rng: random.Random,
    *,
    symmetric: bool = True,
) -> list[tuple[int, int]] | None:
    """Generate a black-cell pattern respecting locked cells.

    Uses the existing pattern generator with locked cell constraints,
    then validates the result.

    Returns sorted list of black cell positions, or None if invalid.
    """
    config = PatternConfig()

    pattern = generate_pattern(
        grid_size,
        grid_size,
        seed=rng.randint(0, 2**31 - 1),
        config=config,
        locked_white=locked_white,
        locked_black=locked_black,
        symmetric=symmetric,
    )

    black = set(pattern)

    # Ensure all locked blacks are present
    if not locked_black.issubset(black):
        return None

    # Ensure no locked whites are black
    if black & locked_white:
        return None

    # Validate constraints
    if not _is_connected(grid_size, grid_size, black):
        return None
    if _has_any_2x2_block(grid_size, grid_size, black):
        return None
    if not _check_min_word_length(grid_size, grid_size, black, 3):
        return None
    if not _all_rows_cols_have_white(grid_size, grid_size, black):
        return None

    # Check density
    density = len(black) / (grid_size * grid_size)
    if density < config.min_density or density > config.max_density:
        return None

    # Check 180-degree symmetry (only when symmetric mode is requested)
    if symmetric:
        for r, c in black:
            if (grid_size - 1 - r, grid_size - 1 - c) not in black:
                return None

    return sorted(black)
