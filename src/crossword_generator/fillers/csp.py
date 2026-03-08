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
            start_c = c
            cells: list[tuple[int, int]] = []
            while c < cols and (r, c) not in black:
                cells.append((r, c))
                c += 1
            if len(cells) >= 2:
                slots.append(_Slot(
                    index=idx, row=r, col=start_c,
                    direction="across", length=len(cells), cells=cells,
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
                    index=idx, row=start_r, col=c,
                    direction="down", length=len(cells), cells=cells,
                ))
                idx += 1

    # Build crossing map
    cell_to_slots: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for slot in slots:
        for pos, cell in enumerate(slot.cells):
            if cell not in cell_to_slots:
                cell_to_slots[cell] = []
            cell_to_slots[cell].append((slot.index, pos))

    for entries in cell_to_slots.values():
        if len(entries) == 2:
            (s1, p1), (s2, p2) = entries
            slots[s1].crossings.append((p1, s2, p2))
            slots[s2].crossings.append((p2, s1, p1))

    return slots


def _build_letter_index_flat(
    words: list[str], word_length: int,
) -> list[int]:
    """Build flat array: [pos * 26 + letter_ord] -> bitset of word indices.

    Returns a list of length word_length * 26.
    """
    arr = [0] * (word_length * 26)
    for wi, word in enumerate(words):
        bit = 1 << wi
        for pos, ch in enumerate(word):
            arr[pos * 26 + ord(ch) - 65] |= bit
    return arr


def _iter_bits(n: int) -> list[int]:
    """Extract set bit positions from a bitset."""
    bits: list[int] = []
    while n:
        b = n & -n
        bits.append(b.bit_length() - 1)
        n ^= b
    return bits


def _shuffle_within_tiers(
    indices: list[int],
    scores: list[int],
    rng: random.Random,
    tier_size: int = 10,
) -> list[int]:
    """Sort indices by score descending, then shuffle within 10-point tiers."""
    pairs = [(i, scores[i]) for i in indices]
    pairs.sort(key=lambda p: p[1], reverse=True)

    result: list[int] = []
    tier: list[int] = []
    tier_floor = None

    for idx, score in pairs:
        floor = (score // tier_size) * tier_size
        if tier_floor is None:
            tier_floor = floor
        if floor != tier_floor:
            rng.shuffle(tier)
            result.extend(tier)
            tier = []
            tier_floor = floor
        tier.append(idx)

    if tier:
        rng.shuffle(tier)
        result.extend(tier)

    return result


class _PrefixTrie:
    """Simple trie for prefix existence checks."""

    def __init__(self) -> None:
        self._root: dict[str, dict] = {}

    def insert(self, word: str) -> None:
        node = self._root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]

    def has_prefix(self, prefix: str) -> bool:
        node = self._root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True


def _arc_revise(
    domains_si: int,
    domains_other: int,
    li_si: list[int],
    li_other: list[int],
    base_i: int,
    base_j: int,
) -> int:
    """Compute supported words in other_si given si's domain.

    Uses flat arrays for fast lookup. Returns the supported bitset.
    """
    supported = 0
    for k in range(26):
        wl = li_si[base_i + k]
        if domains_si & wl:
            supported |= (domains_other & li_other[base_j + k])
    return supported


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
        """Fill a grid using CSP backtracking with random restarts."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        rng = random.Random(seed)
        logger.info(
            "CSP filling %dx%d grid (seed=%d)", spec.rows, spec.cols, seed
        )

        black = set(spec.black_cells)
        slots = _extract_slots(spec.rows, spec.cols, black)

        if not slots:
            grid = [
                [BLACK if (r, c) in black else " " for c in range(spec.cols)]
                for r in range(spec.rows)
            ]
            return FilledGrid(grid=grid)

        # Build candidate word lists and flat letter-index arrays per length
        candidates_by_slot: list[list[str]] = []
        li_flat: dict[int, list[int]] = {}  # word_length -> flat array

        for slot in slots:
            words = self._dictionary.words_by_length(slot.length)
            if not words:
                raise FillError(
                    f"No dictionary words of length {slot.length} "
                    f"for {slot.direction} slot at ({slot.row},{slot.col})"
                )
            candidates_by_slot.append(list(words))
            if slot.length not in li_flat:
                li_flat[slot.length] = _build_letter_index_flat(
                    words, slot.length
                )

        # Per-slot flat array reference (avoids dict lookup in hot path)
        slot_li: list[list[int]] = [
            li_flat[slot.length] for slot in slots
        ]

        # Build prefix tries per word length
        tries: dict[int, _PrefixTrie] = {}
        for length in li_flat:
            trie = _PrefixTrie()
            for slot in slots:
                if slot.length == length:
                    for w in candidates_by_slot[slot.index]:
                        trie.insert(w)
                    break
            tries[length] = trie

        # Pre-compute word scores for value ordering
        scores_by_slot: list[list[int]] = []
        for slot_cands in candidates_by_slot:
            scores_by_slot.append(
                [self._dictionary.score(w) or 0 for w in slot_cands]
            )

        # Domains as bitsets
        initial_domains: list[int] = [
            (1 << len(cands)) - 1 for cands in candidates_by_slot
        ]

        # Initial arc consistency using flat arrays
        self._initial_ac3_flat(slots, initial_domains, slot_li)
        for si, dom in enumerate(initial_domains):
            if not dom:
                raise FillError(
                    f"CSP solver: infeasible after initial AC-3 "
                    f"(slot {si} empty domain)"
                )

        # Resolve timeout
        grid_max = max(spec.rows, spec.cols)
        if self._config.timeout_by_size and grid_max in self._config.timeout_by_size:
            timeout = self._config.timeout_by_size[grid_max]
        else:
            timeout = self._config.timeout

        deadline = time.monotonic() + timeout

        if time.monotonic() > deadline:
            raise FillError(f"CSP solver timed out after {timeout}s")

        # Mutable state (reset per restart attempt)
        domains: list[int] = list(initial_domains)
        assignment: dict[int, int] = {}
        used_words: set[str] = set()
        placed: dict[tuple[int, int], str] = {}
        check_interval = 1000
        backtracks = 0
        backtrack_limit = 10_000

        class _BacktrackLimitError(Exception):
            pass

        def _degree(si: int) -> int:
            count = 0
            for _, other_si, _ in slots[si].crossings:
                if other_si not in assignment:
                    count += 1
            return count

        def solve() -> bool:
            nonlocal backtracks

            if len(assignment) == len(slots):
                return True

            # MRV with degree tie-breaking
            best_slot = -1
            best_size = float("inf")
            best_degree = -1
            for si in range(len(slots)):
                if si in assignment:
                    continue
                dsize = domains[si].bit_count()
                deg = _degree(si)
                if dsize < best_size or (
                    dsize == best_size and deg > best_degree
                ):
                    best_size = dsize
                    best_slot = si
                    best_degree = deg

            if best_slot == -1 or best_size == 0:
                return False

            slot = slots[best_slot]
            cands = candidates_by_slot[best_slot]

            domain_indices = _iter_bits(domains[best_slot])
            ordered = _shuffle_within_tiers(
                domain_indices,
                scores_by_slot[best_slot],
                rng,
            )

            for wi in ordered:
                backtracks += 1
                if backtracks >= backtrack_limit:
                    raise _BacktrackLimitError
                if backtracks % check_interval == 0:
                    if time.monotonic() > deadline:
                        raise _BacktrackLimitError

                word = cands[wi]

                if word in used_words:
                    continue

                # Prefix pruning
                prefix_dead = False
                for pos_in_this, other_si, pos_in_other in slot.crossings:
                    if other_si in assignment:
                        continue
                    other_slot = slots[other_si]
                    prefix = []
                    for cell in other_slot.cells:
                        if cell == slot.cells[pos_in_this]:
                            prefix.append(word[pos_in_this])
                        elif cell in placed:
                            prefix.append(placed[cell])
                        else:
                            break
                    if len(prefix) >= 2 and not tries[
                        other_slot.length
                    ].has_prefix("".join(prefix)):
                        prefix_dead = True
                        break
                if prefix_dead:
                    continue

                # Forward check: prune crossing domains
                saved_domains: dict[int, int] = {}
                feasible = True
                fc_reduced: list[int] = []

                for pos_in_this, other_si, pos_in_other in slot.crossings:
                    if other_si in assignment:
                        other_word = candidates_by_slot[other_si][
                            assignment[other_si]
                        ]
                        if word[pos_in_this] != other_word[pos_in_other]:
                            feasible = False
                            break
                    else:
                        letter = word[pos_in_this]
                        matching = slot_li[other_si][
                            pos_in_other * 26 + ord(letter) - 65
                        ]
                        if other_si not in saved_domains:
                            saved_domains[other_si] = domains[other_si]
                        domains[other_si] = domains[other_si] & matching
                        if not domains[other_si]:
                            feasible = False
                            break
                        fc_reduced.append(other_si)

                # One-level propagation: for each FC-reduced slot,
                # check its other crossings for support
                if feasible:
                    for rsi in fc_reduced:
                        r_slot = slots[rsi]
                        r_li = slot_li[rsi]
                        for rpos, nsi, npos in r_slot.crossings:
                            if nsi == best_slot or nsi in assignment:
                                continue
                            n_li = slot_li[nsi]
                            supported = _arc_revise(
                                domains[rsi], domains[nsi],
                                r_li, n_li,
                                rpos * 26, npos * 26,
                            )
                            if supported != domains[nsi]:
                                if nsi not in saved_domains:
                                    saved_domains[nsi] = domains[nsi]
                                domains[nsi] = supported
                                if not domains[nsi]:
                                    feasible = False
                                    break
                        if not feasible:
                            break

                if feasible:
                    assignment[best_slot] = wi
                    used_words.add(word)
                    newly_placed: list[tuple[int, int]] = []
                    for pos, cell in enumerate(slot.cells):
                        if cell not in placed:
                            placed[cell] = word[pos]
                            newly_placed.append(cell)
                    if solve():
                        return True
                    del assignment[best_slot]
                    used_words.discard(word)
                    for cell in newly_placed:
                        del placed[cell]

                # Restore domains
                for si, saved in saved_domains.items():
                    domains[si] = saved

            return False

        # Random restart loop
        attempt = 0
        total_backtracks = 0
        while time.monotonic() < deadline:
            attempt += 1
            assignment.clear()
            used_words.clear()
            placed.clear()
            backtracks = 0
            domains[:] = list(initial_domains)

            try:
                if solve():
                    total_backtracks += backtracks
                    logger.info(
                        "CSP solved on attempt %d (%d backtracks, "
                        "%d total)",
                        attempt, backtracks, total_backtracks,
                    )
                    break
            except _BacktrackLimitError:
                total_backtracks += backtracks
                logger.debug(
                    "Attempt %d: %d backtracks, restarting",
                    attempt, backtracks,
                )
                rng = random.Random(seed + attempt)
                continue

            # solve() returned False — exhausted search
            raise FillError(
                f"CSP solver could not fill {spec.rows}x{spec.cols} grid"
            )
        else:
            raise FillError(
                f"CSP solver timed out after {timeout}s "
                f"({attempt} attempts, {total_backtracks} backtracks)"
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
            total_backtracks,
        )

        return FilledGrid(
            grid=grid,
            words_across=words_across,
            words_down=words_down,
        )

    @staticmethod
    def _initial_ac3_flat(
        slots: list[_Slot],
        domains: list[int],
        slot_li: list[list[int]],
    ) -> None:
        """Run full AC-3 using flat letter-index arrays."""
        from collections import deque

        queue: deque[tuple[int, int, int, int]] = deque()
        for slot in slots:
            for pos_in_this, other_si, pos_in_other in slot.crossings:
                queue.append(
                    (slot.index, pos_in_this, other_si, pos_in_other)
                )

        while queue:
            si, pos_in_this, other_si, pos_in_other = queue.popleft()
            supported = _arc_revise(
                domains[si], domains[other_si],
                slot_li[si], slot_li[other_si],
                pos_in_this * 26, pos_in_other * 26,
            )
            if supported != domains[other_si]:
                domains[other_si] = supported
                if not domains[other_si]:
                    return
                for pos_o, neighbor_si, pos_n in slots[other_si].crossings:
                    if neighbor_si != si:
                        queue.append(
                            (other_si, pos_o, neighbor_si, pos_n)
                        )
