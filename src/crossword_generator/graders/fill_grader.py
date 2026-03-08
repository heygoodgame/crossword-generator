"""Rule-based fill quality grader against the Jeff Chen word list."""

from __future__ import annotations

import logging

from crossword_generator.dictionary import Dictionary
from crossword_generator.exporters.numbering import NumberedEntry, compute_numbering
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
        min_passing_score: int = 70,
    ) -> None:
        self._dictionary = dictionary
        self._min_passing_score = min_passing_score

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

        word_grades = [self._grade_word(entry) for entry in entries]
        overall_score, grid_penalties = self._compute_aggregate(word_grades)
        passing = overall_score >= self._min_passing_score

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

        if entry.length == 3 and (dict_score is None or dict_score < 55):
            penalties["short_glue"] = 5.0

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

    def _compute_aggregate(
        self, word_grades: list[WordGrade]
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

        # High unknown ratio
        unknown_count = sum(1 for wg in word_grades if wg.dictionary_score is None)
        if len(word_grades) > 0 and unknown_count / len(word_grades) > 0.2:
            grid_penalties["high_unknown_ratio"] = 10.0

        # Excessive short glue
        short_glue_count = sum(
            1
            for wg in word_grades
            if wg.length == 3
            and (wg.dictionary_score is None or wg.dictionary_score < 55)
        )
        if len(word_grades) > 0 and short_glue_count / len(word_grades) > 0.3:
            grid_penalties["excessive_short_glue"] = 5.0

        overall = max(0.0, min(100.0, raw_score - sum(grid_penalties.values())))
        return overall, grid_penalties
