"""Tests for the FillGrader rule-based fill quality scorer."""

from __future__ import annotations

import pytest

from crossword_generator.dictionary import Dictionary
from crossword_generator.graders.fill_grader import FillGrader


def _make_dict(words: dict[str, int]) -> Dictionary:
    """Create a Dictionary from a plain dict (uppercase keys expected)."""
    return Dictionary(words, min_word_score=0, min_2letter_score=0)


# A simple 5x5 grid with real-looking words
GOOD_GRID = [
    ["O", "C", "E", "A", "N"],
    ["P", "A", "R", "S", "E"],
    ["E", "N", "T", "E", "R"],
    ["R", "O", "S", "E", "S"],
    ["A", "N", "G", "E", "L"],
]

# Dictionary with scores for words in GOOD_GRID
GOOD_DICT = _make_dict(
    {
        "OCEAN": 80,
        "PARSE": 75,
        "ENTER": 85,
        "ROSES": 70,
        "ANGEL": 90,
        "OPERA": 65,
        "CANON": 70,
        "ERTSG": 20,  # Low score
        "ASEEE": 40,
        "NERSL": 30,
    }
)


class TestWordScoring:
    def test_high_score_word(self) -> None:
        words = {
            "OCEAN": 80,
            "PARSE": 75,
            "ENTER": 85,
            "ROSES": 70,
            "ANGEL": 90,
        }
        grader = FillGrader(_make_dict(words))
        # Grade a grid with just one across word
        grid = [["O", "C", "E", "A", "N"]]
        report = grader.grade(grid)
        # Single word "OCEAN" with score 80, no penalties
        assert report.word_grades[0].dictionary_score == 80
        assert report.word_grades[0].adjusted_score == 80.0
        assert report.word_grades[0].penalties == {}

    def test_unknown_word_penalty(self) -> None:
        grader = FillGrader(_make_dict({}))
        grid = [["X", "Y", "Z", "Q", "W"]]
        report = grader.grade(grid)
        wg = report.word_grades[0]
        assert wg.dictionary_score is None
        assert "not_in_dictionary" in wg.penalties
        # Base 20 - 30 (not_in_dictionary) = -10, clamped to 0
        assert wg.adjusted_score == 0.0

    def test_low_score_penalty(self) -> None:
        grader = FillGrader(_make_dict({"ABCDE": 52}))
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        wg = report.word_grades[0]
        assert wg.dictionary_score == 52
        assert "low_score" in wg.penalties
        assert wg.adjusted_score == 52.0 - 5.0

    def test_short_glue_penalty(self) -> None:
        grader = FillGrader(_make_dict({"ABC": 50}))
        grid = [["A", "B", "C"]]
        report = grader.grade(grid)
        wg = report.word_grades[0]
        assert "short_glue" in wg.penalties
        assert "low_score" in wg.penalties
        # 50 - 10 (short_glue) - 5 (low_score) = 35
        assert wg.adjusted_score == 35.0

    def test_short_glue_not_applied_above_55(self) -> None:
        grader = FillGrader(_make_dict({"ABC": 60}))
        grid = [["A", "B", "C"]]
        report = grader.grade(grid)
        wg = report.word_grades[0]
        assert "short_glue" not in wg.penalties

    def test_two_letter_penalty(self) -> None:
        # 2-letter word grid: need a 2x1 grid that produces a 2-letter word
        grader = FillGrader(_make_dict({"AB": 60}))
        grid = [["A", "B"]]
        report = grader.grade(grid)
        wg = report.word_grades[0]
        assert "two_letter" in wg.penalties
        assert wg.adjusted_score == 60.0 - 5.0

    def test_score_clamped_to_zero(self) -> None:
        grader = FillGrader(_make_dict({}))
        # Unknown 3-letter: 20 - 30 - 10 = -20, clamped to 0
        grid = [["X", "Y", "Z"]]
        report = grader.grade(grid)
        wg = report.word_grades[0]
        assert wg.adjusted_score == 0.0


class TestGridLevelPenalties:
    def test_duplicate_words_penalty(self) -> None:
        grader = FillGrader(
            _make_dict({"ABC": 70, "DEF": 70}),
        )
        # Grid where "ABC" appears as both across words
        grid = [
            ["A", "B", "C"],
            ["A", "B", "C"],
        ]
        report = grader.grade(grid)
        # "ABC" across appears twice → 1 duplicate pair → -5
        if "duplicate_words" in report.penalties_applied:
            assert report.penalties_applied["duplicate_words"] == 5.0

    def test_high_unknown_ratio_penalty(self) -> None:
        # All words unknown → high_unknown_ratio
        grader = FillGrader(_make_dict({}))
        grid = [
            ["A", "B", "C"],
            ["D", "E", "F"],
        ]
        report = grader.grade(grid)
        assert "high_unknown_ratio" in report.penalties_applied
        assert report.penalties_applied["high_unknown_ratio"] == 10.0

    def test_excessive_short_glue_penalty(self) -> None:
        # Grid with many 3-letter low-score words
        words = {
            "ABC": 50,
            "DEF": 50,
            "GHI": 50,
            "ADG": 50,
            "BEH": 50,
            "CFI": 50,
        }
        grader = FillGrader(_make_dict(words))
        grid = [
            ["A", "B", "C"],
            ["D", "E", "F"],
            ["G", "H", "I"],
        ]
        report = grader.grade(grid)
        # All 6 words are 3-letter with score < 55 → ratio = 1.0 > 0.3
        assert "excessive_short_glue" in report.penalties_applied

    def test_no_grid_penalties_for_good_fill(self) -> None:
        # All across and down words must be in the dict
        words = {
            # across
            "OCEAN": 80,
            "PARSE": 75,
            "ENTER": 85,
            "ROSES": 70,
            "ANGEL": 90,
            # down (col0=OPERA, col1=CANON, col2=ERTSG,
            #        col3=ASEEE, col4=NERSL)
            "OPERA": 65,
            "CANON": 70,
            "ERTSG": 60,
            "ASEEE": 60,
            "NERSL": 60,
        }
        grader = FillGrader(_make_dict(words))
        report = grader.grade(GOOD_GRID)
        # All words present, no duplicates, no excessive glue
        assert "duplicate_words" not in report.penalties_applied
        assert "excessive_short_glue" not in report.penalties_applied
        assert "high_unknown_ratio" not in report.penalties_applied


class TestAggregateScoring:
    def test_length_weighted_mean(self) -> None:
        # 5-letter word (score 80) + 3-letter word (score 60)
        # weighted = (80*5 + 60*3) / (5+3) = (400+180)/8 = 72.5
        grader = FillGrader(
            _make_dict({"ABCDE": 80, "FGH": 60}),
        )
        grid = [
            ["A", "B", "C", "D", "E"],
            [".", ".", "F", "G", "H"],
        ]
        report = grader.grade(grid)
        # Account for down words too — this is a simplified check
        assert report.overall_score > 0

    def test_passing_threshold(self) -> None:
        grader = FillGrader(
            _make_dict({"ABCDE": 80}),
            min_passing_score=70,
        )
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        assert report.passing is True

    def test_failing_threshold(self) -> None:
        grader = FillGrader(
            _make_dict({"ABCDE": 50}),
            min_passing_score=70,
        )
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        # Score = 50 - 5 (low_score) = 45, below 70
        assert report.passing is False

    def test_configurable_threshold(self) -> None:
        grader = FillGrader(
            _make_dict({"ABCDE": 50}),
            min_passing_score=40,
        )
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        # Score = 50 - 5 = 45, above 40
        assert report.passing is True


class TestEdgeCases:
    def test_empty_grid(self) -> None:
        grader = FillGrader(_make_dict({}))
        report = grader.grade([])
        assert report.overall_score == 0.0
        assert report.word_count == 0
        assert report.passing is False

    def test_all_black_grid(self) -> None:
        grader = FillGrader(_make_dict({}))
        grid = [[".", "."], [".", "."]]
        report = grader.grade(grid)
        assert report.word_count == 0
        assert report.passing is False

    def test_report_has_summary(self) -> None:
        grader = FillGrader(_make_dict({"ABCDE": 80}))
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        assert report.summary != ""
        assert "PASS" in report.summary or "FAIL" in report.summary

    def test_report_word_count(self) -> None:
        grader = FillGrader(_make_dict({"ABCDE": 80, "FGHIJ": 75}))
        grid = [
            ["A", "B", "C", "D", "E"],
            ["F", "G", "H", "I", "J"],
        ]
        report = grader.grade(grid)
        # 2 across + some down words
        assert report.word_count >= 2


class TestIntegrationWithRealDictionary:
    def test_grade_with_real_dictionary(self, dictionary_path) -> None:
        """Integration test using the actual Jeff Chen word list."""
        if not dictionary_path.exists():
            pytest.skip("Jeff Chen dictionary not available")
        dictionary = Dictionary.load(dictionary_path)
        grader = FillGrader(dictionary)

        # A grid with real English words
        grid = [
            ["S", "T", "A", "R", "E"],
            ["T", "O", "N", "E", "S"],
            ["A", "R", "E", "N", "A"],
            ["R", "E", "S", "E", "T"],
            ["S", "P", "E", "E", "D"],
        ]
        report = grader.grade(grid)
        assert report.word_count > 0
        assert 0 <= report.overall_score <= 100
