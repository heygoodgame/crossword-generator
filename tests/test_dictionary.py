"""Tests for the dictionary module."""

from pathlib import Path

import pytest

from crossword_generator.dictionary import Dictionary, DictionaryError


@pytest.fixture
def small_dict_file(tmp_path: Path) -> Path:
    """Create a small dictionary file for testing."""
    content = "ocean;50\ncat;60\ndog;40\nab;30\ncd;10\nhi;50\n"
    p = tmp_path / "words.txt"
    p.write_text(content)
    return p


class TestDictionaryLoad:
    """Test Dictionary.load with various inputs."""

    def test_load_and_query(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        assert d.contains("ocean")
        assert d.score("ocean") == 50
        assert d.contains("cat")
        assert d.score("cat") == 60

    def test_filter_by_min_word_score(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        # dog has score 40, below min_word_score=50
        assert not d.contains("dog")

    def test_2letter_threshold_independent(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        # ab;30 meets min_2letter_score=30
        assert d.contains("ab")
        assert d.score("ab") == 30
        # cd;10 does not meet min_2letter_score=30
        assert not d.contains("cd")
        # hi;50 meets min_2letter_score=30
        assert d.contains("hi")

    def test_case_insensitive(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        assert d.contains("OCEAN")
        assert d.contains("Ocean")
        assert d.contains("ocean")
        assert d.score("OCEAN") == 50
        assert d.score("Ocean") == 50

    def test_words_by_length(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        twos = d.words_by_length(2)
        assert "AB" in twos
        assert "HI" in twos
        assert "CD" not in twos  # filtered out
        threes = d.words_by_length(3)
        assert "CAT" in threes

    def test_dunder_contains(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        assert "ocean" in d
        assert "dog" not in d

    def test_dunder_len(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        # ocean, cat, ab, hi pass filters (dog and cd excluded)
        assert len(d) == 4

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DictionaryError, match="not found"):
            Dictionary.load(tmp_path / "nonexistent.txt")

    def test_empty_after_filtering_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "low.txt"
        p.write_text("bad;10\nworse;5\n")
        with pytest.raises(DictionaryError, match="empty after filtering"):
            Dictionary.load(p, min_word_score=50)

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "messy.txt"
        p.write_text("good;50\nbadline\nalso;bad;line\n;50\n")
        d = Dictionary.load(p, min_word_score=50)
        assert len(d) == 1
        assert d.contains("good")

    def test_score_returns_none_for_missing(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        assert d.score("zzz") is None


class TestRealDictionary:
    """Tests against the actual Jeff Chen word list."""

    def test_ocean_score(self, dictionary_path: Path) -> None:
        d = Dictionary.load(dictionary_path)
        assert d.score("OCEAN") == 50
        assert d.contains("ocean")

    def test_no_2letter_words_at_default_threshold(self, dictionary_path: Path) -> None:
        d = Dictionary.load(dictionary_path)
        twos = d.words_by_length(2)
        assert len(twos) == 0

    def test_23_2letter_words_at_score_10(self, dictionary_path: Path) -> None:
        d = Dictionary.load(dictionary_path, min_2letter_score=10)
        twos = d.words_by_length(2)
        assert len(twos) == 23
