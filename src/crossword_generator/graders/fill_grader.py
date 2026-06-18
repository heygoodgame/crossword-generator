"""Rule-based fill quality grader against the Jeff Chen word list."""

from __future__ import annotations

import logging
from collections import defaultdict

from crossword_generator.dictionary import Dictionary
from crossword_generator.exporters.numbering import (
    NumberedEntry,
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.models import FillGradeReport, WordGrade

logger = logging.getLogger(__name__)

_MIN_SHARED_ETYMOLOGY_PART_LENGTH = 4

# Minimum length of a shared dictionary stem for the shared-root rule below.
_MIN_SHARED_ROOT_LENGTH = 4

# Inflectional/derivational suffixes that may attach to a stem at a morpheme
# boundary. The short, ambiguous endings ("s", "y", "ed") are INCLUDED here on
# purpose — the shared-root rule additionally requires the stem itself to be a
# dictionary word AND both answers to be stem+suffix, and that anchoring keeps
# coincidences (START/STAR) out while still catching STINKY/STINKS. The price is
# a rare false positive such as READS/READY; a wrongly rejected fill is simply
# regenerated, which is preferable to shipping the same root twice.
_SHARED_ROOT_SUFFIXES = (
    "iest",
    "ier",
    "ies",
    "ness",
    "less",
    "able",
    "ers",
    "est",
    "ful",
    "ing",
    "es",
    "ed",
    "er",
    "ly",
    "s",
    "d",
    "y",
)


class FillGrader:
    """Scores a filled crossword grid against a scored dictionary.

    Per-word scoring starts with the dictionary score (or 0 if unknown),
    then subtracts penalties. Grid-level scoring is a length-weighted mean
    of adjusted word scores, minus grid-level penalties.
    """

    def __init__(
        self,
        dictionary: Dictionary,
        *,
        min_passing_score: int = 51,
        exact_score_count_length: int | None = None,
        exact_score_count_min_score: int | None = None,
        exact_score_count: int | None = None,
        hard_word_set: frozenset[str] | None = None,
    ) -> None:
        self._dictionary = dictionary
        self._min_passing_score = min_passing_score
        self._exact_score_count_length = exact_score_count_length
        self._exact_score_count_min_score = exact_score_count_min_score
        self._exact_score_count = exact_score_count
        # Words from Jeff's Hard list. When set (Hard puzzles), two rules
        # apply: (1) a grid must contain at least one of these entries, else
        # it is really an Easy puzzle (see _hard_entry_count); and (2) no two
        # of them may cross each other (see _hard_cross_count). Many Hard-list
        # entries are proper names; crossing two of them can force an
        # unsatisfying total guess when a solver knows neither.
        self._hard_word_set = hard_word_set

    def grade(self, grid: list[list[str]]) -> FillGradeReport:
        """Grade a filled grid and return a report."""
        entries = compute_numbering(grid)
        if not entries:
            return FillGradeReport(
                overall_score=0.0,
                word_count=0,
                passing=False,
                summary="Empty grid — no words to grade.",
            )

        grid_size = len(grid)
        word_grades = [self._grade_word(entry) for entry in entries]
        overall_score, grid_penalties = self._compute_aggregate(
            word_grades, grid_size=grid_size
        )

        if self._hard_word_set and self._hard_entry_count(entries) == 0:
            grid_penalties["no_hard_entry"] = 100.0
            overall_score = max(
                0.0, overall_score - grid_penalties["no_hard_entry"]
            )

        hard_cross_pairs = self._hard_cross_count(entries, grid)
        if hard_cross_pairs > 0:
            grid_penalties["hard_cross"] = 100.0 * hard_cross_pairs
            overall_score = max(
                0.0, overall_score - grid_penalties["hard_cross"]
            )

        passing = overall_score >= self._min_passing_score
        if "terminal_s_variants" in grid_penalties:
            passing = False
        if "shared_etymology" in grid_penalties:
            passing = False
        if "shared_root" in grid_penalties:
            passing = False
        if "exact_score_count" in grid_penalties:
            passing = False
        if "no_hard_entry" in grid_penalties:
            passing = False
        if "hard_cross" in grid_penalties:
            passing = False

        summary_parts = [
            f"{len(word_grades)} words,",
            f"score {overall_score:.1f}/100",
            f"({'PASS' if passing else 'FAIL'})",
        ]
        if grid_penalties:
            penalty_strs = [f"{k}: -{v:.0f}" for k, v in grid_penalties.items()]
            summary_parts.append(f"[grid penalties: {', '.join(penalty_strs)}]")

        return FillGradeReport(
            overall_score=overall_score,
            word_count=len(word_grades),
            passing=passing,
            word_grades=word_grades,
            penalties_applied=grid_penalties,
            summary=" ".join(summary_parts),
        )

    def _grade_word(self, entry: NumberedEntry) -> WordGrade:
        """Score a single word entry."""
        dict_score = self._dictionary.score(entry.answer)
        penalties: dict[str, float] = {}

        if dict_score is None:
            base = 0.0
        else:
            base = float(dict_score)

        if entry.length == 2:
            penalties["two_letter"] = 5.0

        # short_glue penalty removed: 3-letter words with score < 55 are
        # structurally unavoidable in 5x5–11x11 grids.

        adjusted = max(0.0, min(100.0, base - sum(penalties.values())))

        return WordGrade(
            word=entry.answer,
            length=entry.length,
            direction=entry.direction,
            number=entry.number,
            dictionary_score=dict_score,
            penalties=penalties,
            adjusted_score=adjusted,
        )

    def _hard_entry_count(self, entries: list[NumberedEntry]) -> int:
        """Count entries that appear in the configured Hard list.

        Used to require at least one Hard-list entry in every Hard puzzle;
        an all-Easy board is really an Easy puzzle. Returns 0 when no hard
        word set is configured.
        """
        if not self._hard_word_set:
            return 0
        return sum(1 for e in entries if e.answer in self._hard_word_set)

    def _hard_cross_count(
        self, entries: list[NumberedEntry], grid: list[list[str]]
    ) -> int:
        """Count crossings between two Hard-list entries.

        A crossing is a single shared cell between an across entry and a
        down entry. We count each across/down entry pair at most once even
        if Jeff's lists ever contained a word twice. Returns 0 when no hard
        word set is configured (e.g. Easy puzzles).
        """
        if not self._hard_word_set:
            return 0

        crossing_words = compute_crossing_words(entries, grid)
        counted: set[tuple[str, str]] = set()
        by_key = {(e.number, e.direction): e for e in entries}

        for (number, direction), crossings in crossing_words.items():
            entry = by_key[(number, direction)]
            if entry.answer not in self._hard_word_set:
                continue
            for other in crossings:
                if other not in self._hard_word_set:
                    continue
                pair = tuple(sorted((entry.answer, other)))
                counted.add(pair)

        return len(counted)

    def _compute_aggregate(
        self, word_grades: list[WordGrade], *, grid_size: int = 0
    ) -> tuple[float, dict[str, float]]:
        """Compute length-weighted mean and grid-level penalties."""
        total_weight = sum(wg.length for wg in word_grades)
        if total_weight == 0:
            return 0.0, {}

        weighted_sum = sum(wg.adjusted_score * wg.length for wg in word_grades)
        raw_score = weighted_sum / total_weight

        grid_penalties: dict[str, float] = {}

        # Duplicate words
        seen: dict[str, int] = {}
        for wg in word_grades:
            seen[wg.word] = seen.get(wg.word, 0) + 1
        duplicate_pairs = sum(count - 1 for count in seen.values() if count > 1)
        if duplicate_pairs > 0:
            grid_penalties["duplicate_words"] = 30.0 * duplicate_pairs

        terminal_s_pairs = _terminal_s_variant_count(seen)
        if terminal_s_pairs > 0:
            grid_penalties["terminal_s_variants"] = 100.0 * terminal_s_pairs

        shared_etymology_pairs = _shared_etymology_pair_count(
            seen, self._dictionary
        )
        if shared_etymology_pairs > 0:
            grid_penalties["shared_etymology"] = 100.0 * shared_etymology_pairs

        shared_root_pairs = _shared_root_pair_count(seen, self._dictionary)
        if shared_root_pairs > 0:
            grid_penalties["shared_root"] = 100.0 * shared_root_pairs

        # High unknown ratio
        unknown_count = sum(1 for wg in word_grades if wg.dictionary_score is None)
        if len(word_grades) > 0 and unknown_count / len(word_grades) > 0.2:
            grid_penalties["high_unknown_ratio"] = 10.0

        if self._exact_score_count is not None:
            target_length = self._exact_score_count_length
            min_score = self._exact_score_count_min_score
            if target_length is None or min_score is None:
                raise ValueError(
                    "exact_score_count requires exact_score_count_length "
                    "and exact_score_count_min_score"
                )

            matching_count = sum(
                1
                for wg in word_grades
                if wg.length == target_length
                and wg.dictionary_score is not None
                and wg.dictionary_score >= min_score
            )
            if matching_count != self._exact_score_count:
                grid_penalties["exact_score_count"] = 100.0

        # excessive_short_glue penalty removed: 3-letter words are
        # structurally unavoidable in 5x5–11x11 grids.

        overall = max(0.0, min(100.0, raw_score - sum(grid_penalties.values())))
        return overall, grid_penalties


def _terminal_s_variant_count(word_counts: dict[str, int]) -> int:
    """Count answer pairs that only differ by one terminal S.

    This intentionally handles only the simple plural-like case Jeff called
    out, such as OPAH/OPAHS. Irregular morphology like EAT/ATE is out of scope.
    """
    words = set(word_counts)
    count = 0
    for word in words:
        if len(word) <= 1 or not word.endswith("S"):
            continue
        if word[:-1] in words:
            count += word_counts[word] * word_counts[word[:-1]]
    return count


def _shared_etymology_pair_count(
    word_counts: dict[str, int], dictionary: Dictionary
) -> int:
    """Count answer pairs that reuse a dictionary-recognized lexical part.

    The source word lists do not include etymology metadata, so this implements
    Jeff's rule as a conservative compound/root check: full entries and
    dictionary-recognized compound pieces of length 4+ may not recur in another
    entry. Requiring a valid split keeps unrelated substrings such as STAR in
    START from being rejected.
    """
    words_by_part: dict[str, set[str]] = defaultdict(set)
    for word in word_counts:
        for part in _etymology_parts(word, dictionary):
            words_by_part[part].add(word)

    pairs: set[tuple[str, str]] = set()
    for words in words_by_part.values():
        if len(words) < 2:
            continue
        sorted_words = sorted(words)
        for index, left in enumerate(sorted_words):
            for right in sorted_words[index + 1:]:
                pairs.add((left, right))
    return len(pairs)


def _etymology_parts(word: str, dictionary: Dictionary) -> set[str]:
    """Return reusable dictionary roots/compound parts for one answer."""
    normalized = word.upper()
    parts: set[str] = set()

    full_part = _dictionary_part(normalized, dictionary)
    if (
        full_part is not None
        and len(full_part) >= _MIN_SHARED_ETYMOLOGY_PART_LENGTH
    ):
        parts.add(full_part)

    for split_at in range(3, len(normalized) - 2):
        left = _dictionary_part(normalized[:split_at], dictionary)
        right = _dictionary_part(normalized[split_at:], dictionary)
        if left is None or right is None:
            continue
        if len(left) >= _MIN_SHARED_ETYMOLOGY_PART_LENGTH:
            parts.add(left)
        if len(right) >= _MIN_SHARED_ETYMOLOGY_PART_LENGTH:
            parts.add(right)

    return parts


def _dictionary_part(word: str, dictionary: Dictionary) -> str | None:
    """Return the dictionary form of a possible component."""
    if len(word) < 3:
        return None
    if dictionary.contains(word):
        return word
    if word.endswith("S") and len(word) > _MIN_SHARED_ETYMOLOGY_PART_LENGTH:
        singular = word[:-1]
        if dictionary.contains(singular):
            return singular
    return None


def _shared_root_pair_count(
    word_counts: dict[str, int], dictionary: Dictionary
) -> int:
    """Count answer pairs that are both derivatives of one dictionary stem.

    Jeff rejects grids that pair two words from the same family even when
    neither is a compound of the other (STINKY/STINKS, PLAYER/PLAYS,
    RAINS/RAINY). The compound-based ``_shared_etymology_pair_count`` misses
    these because the shared stem (STINK) is not a compound *piece* of either
    answer, and the Snowball stemmer splits STINKY/STINKS to different stems.

    This rule keys each answer by every dictionary stem it derives from — a
    stem >= 4 chars that is itself a dictionary entry and to which the answer
    adds exactly one known suffix (see ``_SHARED_ROOT_SUFFIXES``). Two answers
    sharing such a stem are a same-family pair. Requiring BOTH to be a real
    derivative of a real word keeps coincidental shared prefixes out: START is
    not STAR + a suffix, so START/STAR does not collide on STAR.
    """
    words_by_stem: dict[str, set[str]] = defaultdict(set)
    for word in word_counts:
        for stem in _shared_root_stems(word, dictionary):
            words_by_stem[stem].add(word)

    pairs: set[tuple[str, str]] = set()
    for words in words_by_stem.values():
        if len(words) < 2:
            continue
        sorted_words = sorted(words)
        for index, left in enumerate(sorted_words):
            for right in sorted_words[index + 1:]:
                pairs.add((left, right))
    return len(pairs)


def _shared_root_stems(word: str, dictionary: Dictionary) -> set[str]:
    """Return dictionary stems this answer derives from via one known suffix.

    A stem qualifies when it is >= 4 chars, is a dictionary entry, is shorter
    than the answer, and the answer equals the stem plus one suffix from
    ``_SHARED_ROOT_SUFFIXES``. The answer itself is excluded so an exact stem
    entry (handled by the duplicate rule) is not double-counted here.
    """
    normalized = word.upper()
    stems: set[str] = set()
    for suffix in _SHARED_ROOT_SUFFIXES:
        if not normalized.endswith(suffix.upper()):
            continue
        stem = normalized[: -len(suffix)]
        if len(stem) < _MIN_SHARED_ROOT_LENGTH:
            continue
        if dictionary.contains(stem):
            stems.add(stem)
    return stems
