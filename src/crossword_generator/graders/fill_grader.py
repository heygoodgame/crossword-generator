"""Rule-based fill quality grader against the Jeff Chen word list."""

from __future__ import annotations

import logging

from crossword_generator.dictionary import Dictionary
from crossword_generator.exporters.numbering import (
    NumberedEntry,
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.models import FillGradeReport, WordGrade

logger = logging.getLogger(__name__)


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
