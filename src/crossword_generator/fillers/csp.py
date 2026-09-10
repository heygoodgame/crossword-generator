"""Native Python CSP crossword filler using backtracking with forward checking."""

from __future__ import annotations

import logging
import math
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
class Slot:
    """A word slot in the grid."""

    index: int
    row: int
    col: int
    direction: str  # "across" or "down"
    length: int
    cells: list[tuple[int, int]]
    crossings: list[tuple[int, int, int]] = field(default_factory=list)
    # crossings: (pos_in_this_slot, other_slot_index, pos_in_other_slot)


# Backward-compatible alias
_Slot = Slot


def _first_across_slot(slots: list[Slot]) -> int | None:
    """Index of 1-Across: the across slot whose start cell numbers first."""
    across = [s for s in slots if s.direction == "across"]
    if not across:
        return None
    return min(across, key=lambda s: (s.row, s.col)).index


def extract_slots(rows: int, cols: int, black: set[tuple[int, int]]) -> list[Slot]:
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


# Backward-compatible alias
_extract_slots = extract_slots


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


def usage_weight(usage_count: int, penalty: float) -> float:
    """Relative draw weight of a candidate given its recent schedule usage.

    ``(1 + count) ** -penalty``: with the default penalty of 1.0 a word used
    once in the count window is half as likely as a fresh word to be tried
    first, a word used 7 times is 8x less likely, 15 times 16x less likely.

    The weight drives a *weighted shuffle inside each raw-score tier* rather
    than a deterministic score deduction. That distinction matters because
    the production HGG lists are flat (every Easy word scores 50): a
    deduction there degenerates into a strict sort by usage, and since ~95%
    of the 3-letter pool carries a count, the CSP would try the 15 never-used
    glue words first and time out on tight grids (the open 5x5 failed on
    every seed). A weighted shuffle keeps overused words *less likely* to be
    tried first without ever making them strictly last, and tier eligibility
    still uses the raw dictionary score, so nothing becomes unfillable.
    """
    if usage_count <= 0 or penalty <= 0:
        return 1.0
    return (1.0 + usage_count) ** -penalty


def _weighted_shuffle(
    items: list[int],
    weights: list[float],
    rng: random.Random,
) -> list[int]:
    """Order ``items`` by weighted sampling without replacement.

    Efraimidis-Spirakis: key = log(u) / w with u uniform(0, 1]; descending
    keys give a draw order where each item's chance of coming next is
    proportional to its weight.
    """
    keyed = []
    for item, weight in zip(items, weights, strict=True):
        u = rng.random() or 1e-12
        keyed.append((math.log(u) / weight, item))
    keyed.sort(reverse=True)
    return [item for _, item in keyed]


def _shuffle_within_tiers(
    indices: list[int],
    scores: list[int],
    rng: random.Random,
    tier_size: int = 10,
    usage_counts: list[int] | None = None,
    usage_penalty: float = 0.0,
) -> list[int]:
    """Sort indices by score descending, then shuffle within 10-point tiers.

    When ``usage_counts`` (parallel to ``scores``) and a positive
    ``usage_penalty`` are given, the shuffle inside each tier is weighted by
    :func:`usage_weight`, so recently overused answers tend to be tried
    later than fresh answers of the same quality.
    """
    pairs = [(i, scores[i]) for i in indices]
    pairs.sort(key=lambda p: p[1], reverse=True)

    weighted = usage_counts is not None and usage_penalty > 0

    def _flush(tier: list[int]) -> list[int]:
        if not weighted:
            rng.shuffle(tier)
            return tier
        assert usage_counts is not None
        weights = [usage_weight(usage_counts[i], usage_penalty) for i in tier]
        return _weighted_shuffle(tier, weights, rng)

    result: list[int] = []
    tier: list[int] = []
    tier_floor = None

    for idx, score in pairs:
        floor = (score // tier_size) * tier_size
        if tier_floor is None:
            tier_floor = floor
        if floor != tier_floor:
            result.extend(_flush(tier))
            tier = []
            tier_floor = floor
        tier.append(idx)

    if tier:
        result.extend(_flush(tier))

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


def map_seed_entries_to_slots(
    seed_entries: dict[str, str],
    slots: list[Slot],
) -> dict[int, str]:
    """Map seed entry specifications to slot indices.

    Args:
        seed_entries: Keys are "row,col,direction", values are uppercase words.
        slots: Extracted grid slots.

    Returns:
        Dict mapping slot index to seed word.

    Raises:
        FillError: If a seed entry doesn't match any slot.
    """
    result: dict[int, str] = {}

    for key, word in seed_entries.items():
        parts = key.split(",")
        if len(parts) != 3:
            raise FillError(f"Invalid seed entry key format: {key!r}")
        row, col, direction = int(parts[0]), int(parts[1]), parts[2]

        matched = False
        for slot in slots:
            if slot.row == row and slot.col == col and slot.direction == direction:
                if slot.length != len(word):
                    raise FillError(
                        f"Seed entry {word!r} length {len(word)} doesn't match "
                        f"slot length {slot.length} at ({row},{col},{direction})"
                    )
                result[slot.index] = word
                matched = True
                break

        if not matched:
            raise FillError(
                f"No slot found at ({row},{col},{direction}) for seed entry {word!r}"
            )

    return result


class CSPFiller(GridFiller):
    """Grid filler using constraint satisfaction with backtracking."""

    def __init__(
        self,
        config: CSPFillerConfig,
        dictionary: Dictionary,
        *,
        answer_usage_counts: dict[str, int] | None = None,
        answer_usage_penalty: float | None = None,
        hard_word_set: frozenset[str] | None = None,
        proper_noun_set: frozenset[str] | None = None,
        max_proper_noun_ratio: float = 0.15,
        min_proper_noun_allowance: int = 2,
    ) -> None:
        self._config = config
        self._dictionary = dictionary
        # The fill grader's board-level hard fails, enforced during search
        # when ``enforce_grid_rules`` is on: no two Hard-list words may cross,
        # at most max(allowance, floor(ratio * slots)) proper-noun answers,
        # and 1-Across is never a proper noun. Same sets and cap formula as
        # FillGrader so the search never converges on a board it rejects.
        enforce = config.enforce_grid_rules
        self._hard_word_set = hard_word_set if enforce else None
        self._proper_noun_set = proper_noun_set if enforce else None
        self._max_proper_noun_ratio = max_proper_noun_ratio
        self._min_proper_noun_allowance = min_proper_noun_allowance
        # Recent-schedule usage per answer (uppercase). Drives a soft
        # value-ordering penalty so globally overused fill is tried last.
        self._usage_counts = {
            str(answer).strip().upper(): int(count)
            for answer, count in (answer_usage_counts or {}).items()
            if int(count) > 0
        }
        self._usage_penalty = (
            config.answer_usage_penalty
            if answer_usage_penalty is None
            else answer_usage_penalty
        )

    @classmethod
    def from_config(cls, config: CSPFillerConfig) -> CSPFiller:
        """Create a CSPFiller, loading the dictionary from config."""
        project_root = find_project_root()
        paths = [Path(config.dictionary_path), *(
            Path(path) for path in config.additional_dictionary_paths
        )]
        resolved_paths = [
            path if path.is_absolute() else project_root / path
            for path in paths
        ]
        dictionary = Dictionary.load_many(
            resolved_paths,
            min_word_score=config.min_word_score,
            min_2letter_score=config.min_2letter_score,
            additional_min_length=config.additional_dictionary_min_length,
            additional_max_length=config.additional_dictionary_max_length,
        )
        return cls(config, dictionary)

    @property
    def name(self) -> str:
        return "csp"

    @staticmethod
    def _candidate_masks(
        candidates_by_slot: list[list[str]],
        word_set: frozenset[str] | None,
        seed_assignments: dict[int, str],
    ) -> list[int] | None:
        """Per-slot bitmask of candidate indices whose word is in ``word_set``.

        Seeded (theme) slots get an empty mask: intentional themed names
        never count, matching the grader.
        """
        if not word_set:
            return None
        masks: list[int] = []
        for si, cands in enumerate(candidates_by_slot):
            if si in seed_assignments:
                masks.append(0)
                continue
            mask = 0
            for wi, word in enumerate(cands):
                if word in word_set:
                    mask |= 1 << wi
            masks.append(mask)
        return masks

    def fill(self, spec: GridSpec, *, seed: int | None = None) -> FilledGrid:
        """Fill a grid using CSP backtracking with quality-tier passes."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        logger.info(
            "CSP filling %dx%d grid (seed=%d)", spec.rows, spec.cols, seed
        )

        black = set(spec.black_cells)
        slots = extract_slots(spec.rows, spec.cols, black)

        if not slots:
            grid = [
                [BLACK if (r, c) in black else " " for c in range(spec.cols)]
                for r in range(spec.rows)
            ]
            return FilledGrid(grid=grid)

        # Resolve timeout
        grid_max = max(spec.rows, spec.cols)
        if self._config.timeout_by_size and grid_max in self._config.timeout_by_size:
            timeout = self._config.timeout_by_size[grid_max]
        else:
            timeout = self._config.timeout

        deadline = time.monotonic() + timeout

        if time.monotonic() > deadline:
            raise FillError(f"CSP solver timed out after {timeout}s")

        # Parse seed entries from spec
        seed_assignments: dict[int, str] = {}
        if spec.seed_entries:
            seed_slot_map = map_seed_entries_to_slots(spec.seed_entries, slots)
            seed_assignments = seed_slot_map
            logger.info(
                "CSP: %d seed entries to place", len(seed_assignments)
            )

        # Quality tier loop: try high-score words first, then fall back
        tiers = self._config.quality_tiers
        for tier_idx, tier_min_score in enumerate(tiers):
            is_last_tier = tier_idx == len(tiers) - 1
            logger.info(
                "CSP tier %d/%d: min_score=%d",
                tier_idx + 1, len(tiers), tier_min_score,
            )

            # Build candidate word lists for this tier
            candidates_by_slot: list[list[str]] = []
            li_flat: dict[int, list[int]] = {}
            skip_tier = False

            for slot in slots:
                # Seed-assigned slots get a single-word candidate list
                if slot.index in seed_assignments:
                    seed_word = seed_assignments[slot.index]
                    candidates_by_slot.append([seed_word])
                    continue

                min_score = max(
                    tier_min_score,
                    self._config.min_score_by_length.get(slot.length, 0),
                )
                words = self._dictionary.words_by_length(
                    slot.length, min_score=min_score
                )
                if not words:
                    # Fall back to all loaded words only when this length has
                    # no explicit score floor.
                    if slot.length not in self._config.min_score_by_length:
                        words = self._dictionary.words_by_length(slot.length)
                if not words:
                    if is_last_tier:
                        raise FillError(
                            f"No dictionary words of length {slot.length} "
                            f"for {slot.direction} slot at ({slot.row},{slot.col})"
                        )
                    skip_tier = True
                    break
                candidates_by_slot.append(list(words))
                if slot.length not in li_flat:
                    li_flat[slot.length] = _build_letter_index_flat(
                        words, slot.length
                    )

            if skip_tier:
                continue

            # Per-slot flat array reference — seed slots get their own arrays
            slot_li: list[list[int]] = []
            for slot in slots:
                if slot.index in seed_assignments:
                    slot_li.append(
                        _build_letter_index_flat(
                            candidates_by_slot[slot.index], slot.length
                        )
                    )
                else:
                    slot_li.append(li_flat[slot.length])

            # Build prefix tries per word length (skip seed slots)
            tries: dict[int, _PrefixTrie] = {}
            for length in li_flat:
                trie = _PrefixTrie()
                for slot in slots:
                    if (
                        slot.length == length
                        and slot.index not in seed_assignments
                    ):
                        for w in candidates_by_slot[slot.index]:
                            trie.insert(w)
                        break
                tries[length] = trie
            # Ensure seed words are also in tries
            for si, word in seed_assignments.items():
                wlen = len(word)
                if wlen not in tries:
                    tries[wlen] = _PrefixTrie()
                tries[wlen].insert(word)

            # Pre-compute word scores for value ordering
            scores_by_slot: list[list[int]] = []
            usage_by_slot: list[list[int]] | None = None
            if self._usage_counts and self._usage_penalty > 0:
                usage_by_slot = []
            for slot_cands in candidates_by_slot:
                scores_by_slot.append(
                    [self._dictionary.score(w) or 0 for w in slot_cands]
                )
                if usage_by_slot is not None:
                    usage_by_slot.append(
                        [self._usage_counts.get(w, 0) for w in slot_cands]
                    )

            # Domains as bitsets
            initial_domains: list[int] = [
                (1 << len(cands)) - 1 for cands in candidates_by_slot
            ]

            # Grid-rule masks: which candidate indices per slot are Hard-list
            # words / proper nouns. Seeded slots are exempt (theme entries).
            hard_masks = self._candidate_masks(
                candidates_by_slot, self._hard_word_set, seed_assignments
            )
            proper_masks = self._candidate_masks(
                candidates_by_slot, self._proper_noun_set, seed_assignments
            )
            proper_cap = (
                max(
                    self._min_proper_noun_allowance,
                    math.floor(len(slots) * self._max_proper_noun_ratio),
                )
                if proper_masks is not None
                else None
            )
            if proper_masks is not None:
                first_across = _first_across_slot(slots)
                if first_across is not None:
                    initial_domains[first_across] &= ~proper_masks[first_across]

            # Initial arc consistency
            self._initial_ac3_flat(slots, initial_domains, slot_li)

            # Check if any domain is empty after AC-3
            infeasible = False
            for si, dom in enumerate(initial_domains):
                if not dom:
                    infeasible = True
                    break
            if infeasible:
                s = slots[si]
                logger.debug(
                    "AC-3 infeasible: slot %d (%s at %d,%d, len=%d) "
                    "has empty domain",
                    si, s.direction, s.row, s.col, s.length,
                )
                if seed_assignments and logger.isEnabledFor(logging.DEBUG):
                    for seed_si, seed_word in seed_assignments.items():
                        ss = slots[seed_si]
                        logger.debug(
                            "  Seed: slot %d (%s at %d,%d) = %s",
                            seed_si, ss.direction, ss.row, ss.col,
                            seed_word,
                        )
                if is_last_tier:
                    raise FillError(
                        f"CSP solver: infeasible after initial AC-3 "
                        f"(slot {si} empty domain)"
                    )
                logger.info(
                    "Tier %d infeasible after AC-3, trying next tier",
                    tier_idx + 1,
                )
                continue

            result = self._solve_with_restarts(
                spec, slots, candidates_by_slot, slot_li, tries,
                scores_by_slot, initial_domains, black, seed, deadline,
                timeout, seed_assignments,
                usage_by_slot=usage_by_slot,
                hard_masks=hard_masks,
                proper_masks=proper_masks,
                proper_cap=proper_cap,
            )
            if result is not None:
                return result

            # This tier failed — try next
            if not is_last_tier:
                logger.info(
                    "Tier %d (min_score=%d) failed, trying next tier",
                    tier_idx + 1, tier_min_score,
                )

        raise FillError(
            f"CSP solver could not fill {spec.rows}x{spec.cols} grid "
            f"(exhausted all quality tiers)"
        )

    def _solve_with_restarts(
        self,
        spec: GridSpec,
        slots: list[_Slot],
        candidates_by_slot: list[list[str]],
        slot_li: list[list[int]],
        tries: dict[int, _PrefixTrie],
        scores_by_slot: list[list[int]],
        initial_domains: list[int],
        black: set[tuple[int, int]],
        seed: int,
        deadline: float,
        timeout: int,
        seed_assignments: dict[int, str] | None = None,
        *,
        usage_by_slot: list[list[int]] | None = None,
        hard_masks: list[int] | None = None,
        proper_masks: list[int] | None = None,
        proper_cap: int | None = None,
    ) -> FilledGrid | None:
        """Run the random-restart solve loop. Returns FilledGrid or None."""
        if seed_assignments is None:
            seed_assignments = {}
        rng = random.Random(seed)
        usage_penalty = self._usage_penalty if usage_by_slot is not None else 0.0

        # Mutable state (reset per restart attempt)
        domains: list[int] = list(initial_domains)
        assignment: dict[int, int] = {}
        used_words: set[str] = set()
        placed: dict[tuple[int, int], str] = {}
        check_interval = 1000
        backtracks = 0
        backtrack_limit = 10_000
        # Proper-noun answers placed so far (grid rule: at most proper_cap).
        proper_count = 0

        class _BacktrackLimitError(Exception):
            pass

        def _degree(si: int) -> int:
            count = 0
            for _, other_si, _ in slots[si].crossings:
                if other_si not in assignment:
                    count += 1
            return count

        def solve() -> bool:
            nonlocal backtracks, proper_count

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
                usage_counts=(
                    usage_by_slot[best_slot] if usage_by_slot is not None else None
                ),
                usage_penalty=usage_penalty,
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

                is_hard = bool(hard_masks and hard_masks[best_slot] >> wi & 1)
                is_proper = bool(
                    proper_masks and proper_masks[best_slot] >> wi & 1
                )
                if is_proper and proper_cap is not None and proper_count >= proper_cap:
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
                        # hard_cross: two Hard-list words may not cross.
                        if (
                            is_hard
                            and hard_masks is not None
                            and hard_masks[other_si] >> assignment[other_si] & 1
                        ):
                            feasible = False
                            break
                    else:
                        letter = word[pos_in_this]
                        matching = slot_li[other_si][
                            pos_in_other * 26 + ord(letter) - 65
                        ]
                        if is_hard and hard_masks is not None:
                            matching &= ~hard_masks[other_si]
                        if other_si not in saved_domains:
                            saved_domains[other_si] = domains[other_si]
                        domains[other_si] = domains[other_si] & matching
                        if not domains[other_si]:
                            feasible = False
                            break
                        fc_reduced.append(other_si)

                # proper_noun_cap: once this word fills the last allowed
                # proper-noun slot, no unassigned slot may take one.
                if (
                    feasible
                    and is_proper
                    and proper_masks is not None
                    and proper_cap is not None
                    and proper_count + 1 >= proper_cap
                ):
                    for other_si in range(len(slots)):
                        if other_si == best_slot or other_si in assignment:
                            continue
                        pruned = domains[other_si] & ~proper_masks[other_si]
                        if pruned != domains[other_si]:
                            if other_si not in saved_domains:
                                saved_domains[other_si] = domains[other_si]
                            domains[other_si] = pruned
                            if not pruned:
                                feasible = False
                                break

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
                    if is_proper:
                        proper_count += 1
                    newly_placed: list[tuple[int, int]] = []
                    for pos, cell in enumerate(slot.cells):
                        if cell not in placed:
                            placed[cell] = word[pos]
                            newly_placed.append(cell)
                    if solve():
                        return True
                    del assignment[best_slot]
                    used_words.discard(word)
                    if is_proper:
                        proper_count -= 1
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
            proper_count = 0
            domains[:] = list(initial_domains)

            # Pre-assign seed entry slots
            for si, word in seed_assignments.items():
                assignment[si] = 0  # index 0 in single-element candidate list
                used_words.add(word)
                for pos, cell in enumerate(slots[si].cells):
                    placed[cell] = word[pos]

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

            # solve() returned False — exhausted search, tier failed
            return None
        else:
            # Timed out — let the tier loop try the next tier
            return None

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
                    s = slots[other_si]
                    cause = slots[si]
                    logger.debug(
                        "AC-3 wipeout: slot %d (%s %d,%d len=%d) "
                        "emptied by arc from slot %d (%s %d,%d len=%d) "
                        "at crossing pos %d->%d",
                        other_si, s.direction, s.row, s.col, s.length,
                        si, cause.direction, cause.row, cause.col,
                        cause.length,
                        pos_in_this, pos_in_other,
                    )
                    return
                for pos_o, neighbor_si, pos_n in slots[other_si].crossings:
                    if neighbor_si != si:
                        queue.append(
                            (other_si, pos_o, neighbor_si, pos_n)
                        )
