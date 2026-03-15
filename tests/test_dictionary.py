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


class TestWordsByLengthMinScore:
    """Test words_by_length with min_score filtering."""

    def test_min_score_filters(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        # cat has score 60, ocean has score 50
        threes = d.words_by_length(3, min_score=60)
        assert "CAT" in threes
        # ocean is length 5 with score 50, not returned at min_score=60
        fives = d.words_by_length(5, min_score=60)
        assert "OCEAN" not in fives

    def test_min_score_none_returns_all(self, small_dict_file: Path) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        threes = d.words_by_length(3)
        assert "CAT" in threes

    def test_min_score_returns_empty_when_none_qualify(
        self, small_dict_file: Path
    ) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=50, min_2letter_score=30)
        # No 5-letter words with score >= 70
        fives = d.words_by_length(5, min_score=70)
        assert fives == []


class TestExportPlain:
    """Test Dictionary.export_plain()."""

    def test_export_plain_writes_words(
        self, small_dict_file: Path, tmp_path: Path
    ) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=0, min_2letter_score=0)
        out = tmp_path / "export.txt"
        count = d.export_plain(out, min_score=50)
        lines = out.read_text().strip().split("\n")
        assert count == len(lines)
        # All words with score >= 50: ocean(50), cat(60), hi(50)
        assert "ocean" in lines
        assert "cat" in lines
        assert "hi" in lines
        # dog(40), ab(30), cd(10) excluded
        assert "dog" not in lines

    def test_export_plain_respects_min_score(
        self, small_dict_file: Path, tmp_path: Path
    ) -> None:
        d = Dictionary.load(small_dict_file, min_word_score=0, min_2letter_score=0)
        out = tmp_path / "export.txt"
        count = d.export_plain(out, min_score=60)
        lines = out.read_text().strip().split("\n")
        assert count == 1
        assert "cat" in lines
        # ocean(50) excluded at min_score=60
        assert "ocean" not in lines


class TestAddWords:
    """Test Dictionary.add_words()."""

    def test_adds_new_words(self) -> None:
        d = Dictionary({"CAT": 60})
        d.add_words({"MUCUS": 60, "SLIME": 55})
        assert d.contains("MUCUS")
        assert d.score("MUCUS") == 60
        assert d.contains("SLIME")
        assert d.score("SLIME") == 55
        assert len(d) == 3

    def test_does_not_overwrite_existing(self) -> None:
        d = Dictionary({"CAT": 60})
        d.add_words({"CAT": 99})
        assert d.score("CAT") == 60

    def test_case_insensitive_add(self) -> None:
        d = Dictionary({"CAT": 60})
        d.add_words({"mucus": 60})
        assert d.contains("MUCUS")
        assert d.contains("mucus")

    def test_indexed_by_length(self) -> None:
        d = Dictionary({"CAT": 60})
        d.add_words({"MUCUS": 60})
        fives = d.words_by_length(5)
        assert "MUCUS" in fives


class TestRealDictionary:
    """Tests against the actual Jeff Chen word list."""

    def test_ocean_score(self, dictionary_path: Path) -> None:
        d = Dictionary.load(dictionary_path)
        assert d.score("OCEAN") == 58
        assert d.contains("ocean")

    def test_no_2letter_words_at_default_threshold(self, dictionary_path: Path) -> None:
        d = Dictionary.load(dictionary_path)
        twos = d.words_by_length(2)
        assert len(twos) == 0

    def test_no_2letter_words_at_score_10(self, dictionary_path: Path) -> None:
        d = Dictionary.load(dictionary_path, min_2letter_score=10)
        twos = d.words_by_length(2)
        assert len(twos) == 0
