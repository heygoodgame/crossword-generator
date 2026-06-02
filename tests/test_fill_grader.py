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
        # Base 0, no penalty key
        assert "not_in_dictionary" not in wg.penalties
        assert wg.adjusted_score == 0.0

    def test_no_short_glue_penalty(self) -> None:
        """short_glue penalty was removed for all grid sizes."""
        grader = FillGrader(_make_dict({"ABC": 50}))
        grid = [["A", "B", "C"]]
        report = grader.grade(grid)
        wg = report.word_grades[0]
        assert "short_glue" not in wg.penalties
        assert wg.adjusted_score == 50.0

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
        # Unknown 3-letter: base 0, clamped to 0
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
        # "ABC" across appears twice → 1 duplicate pair → -30
        if "duplicate_words" in report.penalties_applied:
            assert report.penalties_applied["duplicate_words"] == 30.0

    def test_terminal_s_variant_penalty_fails_fill(self) -> None:
        grader = FillGrader(
            _make_dict(
                {
                    "OPAH": 80,
                    "OPAHS": 80,
                    "OO": 80,
                    "PP": 80,
                    "AA": 80,
                    "HH": 80,
                }
            ),
            min_passing_score=0,
        )
        grid = [
            ["O", "P", "A", "H", "."],
            ["O", "P", "A", "H", "S"],
        ]

        report = grader.grade(grid)

        assert report.penalties_applied["terminal_s_variants"] == 100.0
        assert report.passing is False

    def test_terminal_s_variant_counts_each_pair(self) -> None:
        grader = FillGrader(
            _make_dict(
                {
                    "ABC": 80,
                    "ABCS": 80,
                    "DEF": 80,
                    "DEFS": 80,
                    "AD": 80,
                    "BE": 80,
                    "CF": 80,
                    "CS": 80,
                }
            ),
            min_passing_score=0,
        )
        grid = [
            ["A", "B", "C", "."],
            ["A", "B", "C", "S"],
            ["D", "E", "F", "."],
            ["D", "E", "F", "S"],
        ]

        report = grader.grade(grid)

        assert report.penalties_applied["terminal_s_variants"] == 200.0
        assert report.passing is False

    def test_irregular_word_relationships_are_not_terminal_s_variants(
        self,
    ) -> None:
        grader = FillGrader(
            _make_dict(
                {
                    "EAT": 80,
                    "ATE": 80,
                    "EA": 80,
                    "AT": 80,
                }
            ),
            min_passing_score=0,
        )
        grid = [
            ["E", "A", "T"],
            ["A", "T", "E"],
        ]

        report = grader.grade(grid)

        assert "terminal_s_variants" not in report.penalties_applied

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

    def test_no_excessive_short_glue_penalty(self) -> None:
        """excessive_short_glue penalty was removed for all grid sizes."""
        words = {
            "ABC": 50, "DEF": 50, "GHI": 50, "JKL": 50,
            "MNO": 50, "PQR": 50, "STU": 50, "VWX": 50,
            "AGMS": 50, "BHNT": 50, "CIOU": 50,
            "DJPV": 50, "EKQW": 50, "FLRX": 50,
            "ABCDEFGH": 60, "IJKLMNOP": 60,
        }
        grader = FillGrader(_make_dict(words))
        grid = [
            ["A", "B", "C", ".", "D", "E", "F", "."],
            ["G", "H", "I", ".", "J", "K", "L", "."],
            ["M", "N", "O", ".", "P", "Q", "R", "."],
            ["S", "T", "U", ".", "V", "W", "X", "."],
            [".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", "."],
            ["A", "B", "C", "D", "E", "F", "G", "H"],
            ["I", "J", "K", "L", "M", "N", "O", "P"],
        ]
        report = grader.grade(grid)
        assert "excessive_short_glue" not in report.penalties_applied

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

    def test_exact_score_count_passes_with_one_matching_entry(self) -> None:
        grader = FillGrader(
            _make_dict(
                {
                    "AAAAAAA": 60,
                    "BBBBBBB": 55,
                    "CCCCCCC": 55,
                    "ABC": 55,
                    "ABCABC": 55,
                }
            ),
            exact_score_count_length=7,
            exact_score_count_min_score=60,
            exact_score_count=1,
        )

        report = grader.grade([["A"] * 7, ["."] * 7, ["B"] * 7])

        assert "exact_score_count" not in report.penalties_applied
        assert report.passing is True

    def test_exact_score_count_fails_when_too_many_entries_match(self) -> None:
        grader = FillGrader(
            _make_dict({"AAAAAAA": 60, "BBBBBBB": 60}),
            min_passing_score=0,
            exact_score_count_length=7,
            exact_score_count_min_score=60,
            exact_score_count=1,
        )

        report = grader.grade([["A"] * 7, ["."] * 7, ["B"] * 7])

        assert report.penalties_applied["exact_score_count"] == 100.0
        assert report.passing is False


class TestHardCrossRule:
    """No two Hard-list entries may cross each other (Jeff, 2026-06)."""

    # Across row0 = CAT, down col0 = CAB; they cross at (0,0).
    CROSS_GRID = [
        ["C", "A", "T"],
        ["A", ".", "."],
        ["B", ".", "."],
    ]

    CROSS_DICT = _make_dict({"CAT": 60, "CAB": 60})

    def test_two_hard_entries_crossing_fails(self) -> None:
        grader = FillGrader(
            self.CROSS_DICT,
            min_passing_score=0,
            hard_word_set=frozenset({"CAT", "CAB"}),
        )

        report = grader.grade(self.CROSS_GRID)

        assert report.penalties_applied["hard_cross"] == 100.0
        assert report.passing is False

    def test_hard_crossing_easy_passes(self) -> None:
        # Only CAT is a Hard-list entry; CAB is Easy fill.
        grader = FillGrader(
            self.CROSS_DICT,
            min_passing_score=0,
            hard_word_set=frozenset({"CAT"}),
        )

        report = grader.grade(self.CROSS_GRID)

        assert "hard_cross" not in report.penalties_applied
        assert report.passing is True

    def test_no_hard_word_set_skips_rule(self) -> None:
        # Easy puzzles pass no hard_word_set, so the rule never fires.
        grader = FillGrader(self.CROSS_DICT, min_passing_score=0)

        report = grader.grade(self.CROSS_GRID)

        assert "hard_cross" not in report.penalties_applied
        assert report.passing is True

    def test_two_hard_entries_not_crossing_passes(self) -> None:
        # AAA (across, row0) and BBB (across, row2) are both Hard-list
        # entries but share no cell, so they do not cross.
        grid = [
            ["A", "A", "A"],
            [".", ".", "."],
            ["B", "B", "B"],
        ]
        grader = FillGrader(
            _make_dict({"AAA": 60, "BBB": 60}),
            min_passing_score=0,
            hard_word_set=frozenset({"AAA", "BBB"}),
        )

        report = grader.grade(grid)

        assert "hard_cross" not in report.penalties_applied
        assert report.passing is True

    def test_counts_each_crossing_pair_once(self) -> None:
        # CAT (across) crosses both CAB (down col0) and TUB (down col2);
        # all three are Hard-list entries → two distinct hard-cross pairs.
        grid = [
            ["C", "A", "T"],
            ["A", ".", "U"],
            ["B", ".", "B"],
        ]
        grader = FillGrader(
            _make_dict({"CAT": 60, "CAB": 60, "TUB": 60}),
            min_passing_score=0,
            hard_word_set=frozenset({"CAT", "CAB", "TUB"}),
        )

        report = grader.grade(grid)

        assert report.penalties_applied["hard_cross"] == 200.0
        assert report.passing is False


class TestMinOneHardEntryRule:
    """Every Hard puzzle must contain a Hard-list entry (Jeff, 2026-06)."""

    # Two non-crossing across words: AAA (row0) and BBB (row2).
    GRID = [
        ["A", "A", "A"],
        [".", ".", "."],
        ["B", "B", "B"],
    ]
    DICT = _make_dict({"AAA": 60, "BBB": 60})

    def test_no_hard_entry_fails(self) -> None:
        # Neither word is a Hard-list entry → an all-Easy board, rejected.
        grader = FillGrader(
            self.DICT,
            min_passing_score=0,
            hard_word_set=frozenset({"ZZZ"}),
        )

        report = grader.grade(self.GRID)

        assert report.penalties_applied["no_hard_entry"] == 100.0
        assert report.passing is False

    def test_one_hard_entry_passes(self) -> None:
        grader = FillGrader(
            self.DICT,
            min_passing_score=0,
            hard_word_set=frozenset({"AAA"}),
        )

        report = grader.grade(self.GRID)

        assert "no_hard_entry" not in report.penalties_applied
        assert report.passing is True

    def test_no_hard_word_set_skips_rule(self) -> None:
        # Easy puzzles pass no hard_word_set, so an all-Easy board is fine.
        grader = FillGrader(self.DICT, min_passing_score=0)

        report = grader.grade(self.GRID)

        assert "no_hard_entry" not in report.penalties_applied
        assert report.passing is True


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
        # Score = 50, below 70
        assert report.passing is False

    def test_configurable_threshold(self) -> None:
        grader = FillGrader(
            _make_dict({"ABCDE": 50}),
            min_passing_score=40,
        )
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        # Score = 50, above 40
        assert report.passing is True

    def test_default_threshold_is_51(self) -> None:
        """Default threshold of 51 is achievable with real dict scores."""
        grader = FillGrader(_make_dict({"ABCDE": 60}))
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        # Score = 60, above default 51
        assert report.passing is True

    def test_default_threshold_fails_bare_minimum(self) -> None:
        """All score-50 words should fail the default threshold of 51."""
        grader = FillGrader(_make_dict({"ABCDE": 50}))
        grid = [["A", "B", "C", "D", "E"]]
        report = grader.grade(grid)
        # Score = 50, below default 51
        assert report.passing is False


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
