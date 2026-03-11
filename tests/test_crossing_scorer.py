"""Tests for the crossing scorer module."""

from __future__ import annotations

from crossword_generator.dictionary import Dictionary
from crossword_generator.steps.crossing_scorer import (
    build_letter_position_index,
    rank_candidates,
    score_word,
)


def _make_dict(words: dict[str, int]) -> Dictionary:
    return Dictionary(words, min_word_score=0, min_2letter_score=0)


class TestBuildLetterPositionIndex:
    def test_index_keys_are_word_lengths(self) -> None:
        dictionary = _make_dict({"CAT": 50, "DOG": 50, "EAGLE": 50})
        index = build_letter_position_index(dictionary, grid_size=9)
        assert 3 in index
        assert 5 in index
        assert 4 not in index  # no 4-letter words

    def test_index_counts_letters_at_positions(self) -> None:
        dictionary = _make_dict({"CAT": 50, "CAR": 50, "DOG": 50})
        index = build_letter_position_index(dictionary, grid_size=5)
        # Position 0 of length 3: C appears in CAT and CAR (2), D in DOG (1)
        pos0 = index[3][0]
        assert pos0["C"] == 2
        assert pos0["D"] == 1

    def test_empty_dictionary_gives_empty_index(self) -> None:
        dictionary = _make_dict({})
        index = build_letter_position_index(dictionary, grid_size=9)
        assert index == {}

    def test_respects_grid_size_limit(self) -> None:
        dictionary = _make_dict({"CAT": 50, "ELEPHANT": 50})
        index = build_letter_position_index(dictionary, grid_size=5)
        assert 3 in index
        assert 8 not in index  # ELEPHANT is 8 letters, > grid_size 5


class TestScoreWord:
    def test_common_letters_score_higher(self) -> None:
        # Build a dictionary with many words using common letters
        words = {
            "ATE": 50, "ARE": 50, "ACE": 50, "AGE": 50, "APE": 50,
            "SET": 50, "SIT": 50, "SAT": 50, "SOT": 50, "SUN": 50,
        }
        dictionary = _make_dict(words)
        index = build_letter_position_index(dictionary, grid_size=5)

        # "ATE" has very common letters (A, T, E appear in many words)
        score_ate = score_word("ATE", index, grid_size=5)
        # "ZQX" would have zero support
        score_zqx = score_word("ZQX", index, grid_size=5)
        assert score_ate > score_zqx

    def test_zero_for_unsupported_letters(self) -> None:
        dictionary = _make_dict({"CAT": 50, "DOG": 50})
        index = build_letter_position_index(dictionary, grid_size=5)
        # Z doesn't appear in any word
        score = score_word("ZZZ", index, grid_size=5)
        assert score == 0.0

    def test_word_too_short_scores_zero(self) -> None:
        dictionary = _make_dict({"CAT": 50})
        index = build_letter_position_index(dictionary, grid_size=5)
        assert score_word("AB", index, grid_size=5) == 0.0

    def test_word_too_long_scores_zero(self) -> None:
        dictionary = _make_dict({"CAT": 50})
        index = build_letter_position_index(dictionary, grid_size=5)
        assert score_word("TOOLONG", index, grid_size=5) == 0.0

    def test_positive_score_for_valid_word(self) -> None:
        words = {"CAT": 50, "CAN": 50, "COP": 50, "ATE": 50, "THE": 50}
        dictionary = _make_dict(words)
        index = build_letter_position_index(dictionary, grid_size=5)
        assert score_word("CAT", index, grid_size=5) > 0.0


class TestRankCandidates:
    def test_ranks_by_score_descending(self) -> None:
        words = {
            "ATE": 50, "ARE": 50, "ACE": 50, "AGE": 50, "APE": 50,
            "SET": 50, "SIT": 50, "SAT": 50,
        }
        dictionary = _make_dict(words)
        ranked = rank_candidates(
            ["ZQX", "ATE", "SET"], "REVEAL", dictionary, grid_size=5
        )
        # ATE and SET should rank above ZQX
        words_ranked = [w for w, _ in ranked]
        assert words_ranked.index("ATE") < words_ranked.index("ZQX")

    def test_excludes_revealer(self) -> None:
        dictionary = _make_dict({"CAT": 50, "DOG": 50, "BAT": 50})
        ranked = rank_candidates(
            ["CAT", "DOG", "BAT"], "CAT", dictionary, grid_size=5
        )
        words_ranked = [w for w, _ in ranked]
        assert "CAT" not in words_ranked

    def test_empty_candidates(self) -> None:
        dictionary = _make_dict({"CAT": 50})
        ranked = rank_candidates([], "REVEAL", dictionary, grid_size=5)
        assert ranked == []
