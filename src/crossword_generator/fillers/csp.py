"""Native Python CSP crossword filler using backtracking with forward checking."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from crossword_generator.config import CSPFillerConfig, find_project_root
from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FilledGrid, FillError, GridFiller, GridSpec

logger = logging.getLogger(__name__)

BLACK = "."


@dataclass
class _Slot:
    """A word slot in the grid."""

    index: int
    row: int
    col: int
    direction: str  # "across" or "down"
    length: int
    cells: list[tuple[int, int]]
    crossings: list[tuple[int, int, int]] = field(default_factory=list)
    # crossings: (pos_in_this_slot, other_slot_index, pos_in_other_slot)


def _extract_slots(rows: int, cols: int, black: set[tuple[int, int]]) -> list[_Slot]:
    """Extract word slots from grid dimensions and black cell positions."""
    slots: list[_Slot] = []
    idx = 0

    # Across slots
    for r in range(rows):
        c = 0
        while c < cols:
            if (r, c) in black:
                c += 1
                continue
            # Start of a run
            start_c = c
            cells: list[tuple[int, int]] = []
            while c < cols and (r, c) not in black:
                cells.append((r, c))
                c += 1
            if len(cells) >= 2:
                slots.append(_Slot(
                    index=idx,
                    row=r,
                    col=start_c,
                    direction="across",
                    length=len(cells),
                    cells=cells,
                ))
                idx += 1

    # Down slots
    for c in range(cols):
        r = 0
        while r < rows:
            if (r, c) in black:
                r += 1
                continue
            start_r = r
            cells = []
            while r < rows and (r, c) not in black:
                cells.append((r, c))
                r += 1
            if len(cells) >= 2:
                slots.append(_Slot(
                    index=idx,
                    row=start_r,
                    col=c,
                    direction="down",
                    length=len(cells),
                    cells=cells,
                ))
                idx += 1

    # Build crossing map: for each cell, track which slots pass through it
    cell_to_slots: dict[tuple[int, int], list[tuple[int, int]]] = {}
    # cell -> [(slot_index, position_in_slot)]
    for slot in slots:
        for pos, cell in enumerate(slot.cells):
            if cell not in cell_to_slots:
                cell_to_slots[cell] = []
            cell_to_slots[cell].append((slot.index, pos))

    # Link crossing slots
    for cell, entries in cell_to_slots.items():
        if len(entries) == 2:
            (s1, p1), (s2, p2) = entries
            slots[s1].crossings.append((p1, s2, p2))
            slots[s2].crossings.append((p2, s1, p1))

    return slots


def _build_letter_index(
    words: list[str],
) -> dict[tuple[int, str], set[int]]:
    """Build index: (position, letter) -> set of word indices."""
    index: dict[tuple[int, str], set[int]] = {}
    for wi, word in enumerate(words):
        for pos, ch in enumerate(word):
            key = (pos, ch)
            if key not in index:
                index[key] = set()
            index[key].add(wi)
    return index


class CSPFiller(GridFiller):
    """Grid filler using constraint satisfaction with backtracking."""

    def __init__(self, config: CSPFillerConfig, dictionary: Dictionary) -> None:
        self._config = config
        self._dictionary = dictionary

    @classmethod
    def from_config(cls, config: CSPFillerConfig) -> CSPFiller:
        """Create a CSPFiller, loading the dictionary from config."""
        dict_path = Path(config.dictionary_path)
        if not dict_path.is_absolute():
            dict_path = find_project_root() / dict_path
        dictionary = Dictionary.load(
            dict_path,
            min_word_score=config.min_word_score,
            min_2letter_score=config.min_2letter_score,
        )
        return cls(config, dictionary)

    @property
    def name(self) -> str:
        return "csp"

    def fill(self, spec: GridSpec, *, seed: int | None = None) -> FilledGrid:
        """Fill a grid using CSP backtracking search.

        Args:
            spec: Grid dimensions and constraints.
            seed: Random seed for reproducibility.

        Returns:
            FilledGrid with the completed grid.

        Raises:
            FillError: If no valid fill is found within the timeout.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        rng = random.Random(seed)
        logger.info(
            "CSP filling %dx%d grid (seed=%d)", spec.rows, spec.cols, seed
        )

        black = set(spec.black_cells)
        slots = _extract_slots(spec.rows, spec.cols, black)

        if not slots:
            # Grid with no valid slots (all black or too small)
            grid = [
                [BLACK if (r, c) in black else " " for c in range(spec.cols)]
                for r in range(spec.rows)
            ]
            return FilledGrid(grid=grid)

        # Build candidate word lists and letter indices per slot length
        candidates_by_slot: list[list[str]] = []
        letter_indices: dict[int, dict[tuple[int, str], set[int]]] = {}

        for slot in slots:
            words = self._dictionary.words_by_length(slot.length)
            if not words:
                raise FillError(
                    f"No dictionary words of length {slot.length} "
                    f"for {slot.direction} slot at ({slot.row},{slot.col})"
                )
            candidates_by_slot.append(list(words))
            if slot.length not in letter_indices:
                letter_indices[slot.length] = _build_letter_index(words)

        # Domains: set of candidate indices for each slot
        domains: list[set[int]] = [
            set(range(len(cands))) for cands in candidates_by_slot
        ]

        assignment: dict[int, int] = {}  # slot_index -> word_index
        deadline = time.monotonic() + self._config.timeout
        check_interval = 500
        backtracks = 0

        def solve() -> bool:
            nonlocal backtracks

            if len(assignment) == len(slots):
                return True

            # MRV: pick unassigned slot with smallest domain
            best_slot = -1
            best_size = float("inf")
            for si in range(len(slots)):
                if si not in assignment and len(domains[si]) < best_size:
                    best_size = len(domains[si])
                    best_slot = si

            if best_slot == -1 or best_size == 0:
                return False

            slot = slots[best_slot]
            cands = candidates_by_slot[best_slot]
            # Randomized order for variety
            ordered = list(domains[best_slot])
            rng.shuffle(ordered)

            for wi in ordered:
                backtracks += 1
                if backtracks % check_interval == 0:
                    if time.monotonic() > deadline:
                        raise FillError(
                            f"CSP solver timed out after {self._config.timeout}s"
                        )

                word = cands[wi]

                # Check consistency: no other assigned slot should use same word
                conflict = False
                for assigned_si, assigned_wi in assignment.items():
                    if candidates_by_slot[assigned_si][assigned_wi] == word:
                        conflict = True
                        break
                if conflict:
                    continue

                # Forward check: prune crossing domains
                saved_domains: dict[int, set[int]] = {}
                feasible = True

                for pos_in_this, other_si, pos_in_other in slot.crossings:
                    if other_si in assignment:
                        # Already assigned — just check consistency
                        other_word = candidates_by_slot[other_si][
                            assignment[other_si]
                        ]
                        if word[pos_in_this] != other_word[pos_in_other]:
                            feasible = False
                            break
                    else:
                        # Prune other slot's domain
                        letter = word[pos_in_this]
                        li = letter_indices[slots[other_si].length]
                        matching = li.get((pos_in_other, letter), set())

                        if other_si not in saved_domains:
                            saved_domains[other_si] = domains[other_si]
                        domains[other_si] = domains[other_si] & matching

                        if not domains[other_si]:
                            feasible = False
                            break

                if feasible:
                    assignment[best_slot] = wi
                    if solve():
                        return True
                    del assignment[best_slot]

                # Restore domains
                for si, saved in saved_domains.items():
                    domains[si] = saved

            return False

        if not solve():
            raise FillError(
                f"CSP solver could not fill {spec.rows}x{spec.cols} grid"
            )

        # Build grid from assignment
        grid: list[list[str]] = [
            [BLACK if (r, c) in black else " " for c in range(spec.cols)]
            for r in range(spec.rows)
        ]

        for si, wi in assignment.items():
            slot = slots[si]
            word = candidates_by_slot[si][wi]
            for pos, (r, c) in enumerate(slot.cells):
                grid[r][c] = word[pos]

        words_across = [
            candidates_by_slot[si][wi]
            for si, wi in sorted(assignment.items())
            if slots[si].direction == "across"
        ]
        words_down = [
            candidates_by_slot[si][wi]
            for si, wi in sorted(assignment.items())
            if slots[si].direction == "down"
        ]

        logger.info(
            "CSP fill complete: %d across, %d down words (%d backtracks)",
            len(words_across),
            len(words_down),
            backtracks,
        )

        return FilledGrid(
            grid=grid,
            words_across=words_across,
            words_down=words_down,
        )
