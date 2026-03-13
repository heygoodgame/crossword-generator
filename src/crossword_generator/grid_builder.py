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

    results: list[GridSpec] = []

    for variant in range(count):
        rng = random.Random(seed + variant)

        spec = _try_place_entries(
            grid_size, all_words, sym_pairs, partition_cache, rng
        )
        if spec is not None:
            results.append(spec)

    logger.info(
        "build_themed_grids: %d/%d variants succeeded for %d entries "
        "in %dx%d grid",
        len(results), count, len(all_words), grid_size, grid_size,
    )
    return results


def _try_place_entries(
    grid_size: int,
    words: list[str],
    sym_pairs: list[tuple[int, int]],
    partition_cache: dict[int, list[LinePartition]],
    rng: random.Random,
) -> GridSpec | None:
    """Try to place all words in across or down slots and build a valid grid.

    Across placements are attempted first (conventional for theme entries),
    with down as a fallback when no across slot works for a word.

    Returns a GridSpec on success, None on failure.
    """
    locked_white: set[tuple[int, int]] = set()
    locked_black: set[tuple[int, int]] = set()
    seed_entries: dict[str, str] = {}

    # Shuffle symmetric pairs for variety
    available_pairs = list(sym_pairs)
    rng.shuffle(available_pairs)

    # Track which rows/columns have been claimed by across/down entries
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    words_to_place = list(words)

    for word in words_to_place:
        placed = False
        wlen = len(word)
        partitions = partition_cache[wlen]

        # --- Try across placements first (convention) ---
        for line_a, line_b in available_pairs:
            if line_a in used_rows:
                continue

            shuffled_parts = list(partitions)
            rng.shuffle(shuffled_parts)

            for part in shuffled_parts:
                new_blacks_a = {(line_a, c) for c in part.black_positions}
                new_whites_a = {
                    (line_a, c)
                    for c in range(part.start, part.start + wlen)
                }

                # Mirror for symmetric row
                new_blacks_b: set[tuple[int, int]] = set()
                new_whites_mirror: set[tuple[int, int]] = set()
                if line_a != line_b:
                    for _, c in new_blacks_a:
                        new_blacks_b.add((line_b, grid_size - 1 - c))
                    for _, c in new_whites_a:
                        new_whites_mirror.add((line_b, grid_size - 1 - c))

                all_new_blacks = new_blacks_a | new_blacks_b
                all_new_whites = new_whites_a | new_whites_mirror

                # Conflict checks
                if all_new_blacks & locked_white:
                    continue
                if all_new_whites & locked_black:
                    continue
                if all_new_blacks & all_new_whites:
                    continue

                # Check 2x2 blocks with existing locked blacks
                trial_black = locked_black | all_new_blacks
                has_block = False
                for r, c in all_new_blacks:
                    if _has_2x2_block(trial_black, r, c):
                        has_block = True
                        break
                if has_block:
                    continue

                # Accept across placement
                locked_white |= all_new_whites
                locked_black |= all_new_blacks
                used_rows.add(line_a)
                if line_a != line_b:
                    used_rows.add(line_b)

                key = f"{line_a},{part.start},across"
                seed_entries[key] = word.upper()
                placed = True
                break

            if placed:
                break

        # --- If across failed, try down placements ---
        if not placed:
            for line_a, line_b in available_pairs:
                if line_a in used_cols:
                    continue

                shuffled_parts = list(partitions)
                rng.shuffle(shuffled_parts)

                for part in shuffled_parts:
                    new_blacks_a = {
                        (r, line_a) for r in part.black_positions
                    }
                    new_whites_a = {
                        (part.start + i, line_a) for i in range(wlen)
                    }

                    # Mirror for symmetric column
                    new_blacks_b: set[tuple[int, int]] = set()
                    new_whites_mirror: set[tuple[int, int]] = set()
                    if line_a != line_b:
                        for r, _ in new_blacks_a:
                            new_blacks_b.add(
                                (grid_size - 1 - r, line_b)
                            )
                        for r, _ in new_whites_a:
                            new_whites_mirror.add(
                                (grid_size - 1 - r, line_b)
                            )

                    all_new_blacks = new_blacks_a | new_blacks_b
                    all_new_whites = new_whites_a | new_whites_mirror

                    # Conflict checks
                    if all_new_blacks & locked_white:
                        continue
                    if all_new_whites & locked_black:
                        continue
                    if all_new_blacks & all_new_whites:
                        continue

                    # Check 2x2 blocks with existing locked blacks
                    trial_black = locked_black | all_new_blacks
                    has_block = False
                    for r, c in all_new_blacks:
                        if _has_2x2_block(trial_black, r, c):
                            has_block = True
                            break
                    if has_block:
                        continue

                    # Accept down placement
                    locked_white |= all_new_whites
                    locked_black |= all_new_blacks
                    used_cols.add(line_a)
                    if line_a != line_b:
                        used_cols.add(line_b)

                    key = f"{part.start},{line_a},down"
                    seed_entries[key] = word.upper()
                    placed = True
                    break

                if placed:
                    break

        if not placed:
            logger.debug(
                "Could not place word %r (len %d) in any row or column",
                word, wlen,
            )
            return None

    # Phase A.5: Seal short gaps created by locked blacks.
    # When blacks in nearby rows/cols leave 1- or 2-cell white gaps,
    # those gaps must become black to avoid illegal slot lengths.
    if not _seal_short_gaps(grid_size, locked_white, locked_black):
        logger.debug("Cannot seal all short gaps without 2x2 blocks")
        return None

    # Phase B: Generate black cells around the placements
    black_cells = _generate_constrained_pattern(
        grid_size, locked_white, locked_black, rng
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
) -> bool:
    """Seal 1- and 2-cell gaps in rows/columns between locked blacks.

    Only seals a gap if doing so does not create a 2x2 all-black block.
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
                            mir = (
                                grid_size - 1 - cell[0],
                                grid_size - 1 - cell[1],
                            )
                            for c in (cell, mir):
                                if c in locked_white:
                                    # Can't make a locked-white cell black
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

    # Check 180-degree symmetry
    for r, c in black:
        if (grid_size - 1 - r, grid_size - 1 - c) not in black:
            return None

    return sorted(black)
