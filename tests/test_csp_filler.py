"""Tests for the native CSP crossword filler."""

from __future__ import annotations

import pytest

from crossword_generator.config import CSPFillerConfig
from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FilledGrid, FillError, GridSpec
from crossword_generator.fillers.csp import CSPFiller, _extract_slots


@pytest.fixture
def small_dictionary() -> Dictionary:
    """A small dictionary for fast tests."""
    words = {
        # 3-letter words
        "ACE": 60,
        "ACT": 60,
        "ADD": 60,
        "AGE": 60,
        "AID": 60,
        "AIM": 60,
        "AIR": 60,
        "ALE": 60,
        "ALL": 60,
        "AND": 60,
        "ANT": 60,
        "APE": 60,
        "ARC": 60,
        "ARE": 60,
        "ARK": 60,
        "ARM": 60,
        "ART": 60,
        "ATE": 60,
        "AWE": 60,
        "AXE": 60,
        "BAD": 60,
        "BAG": 60,
        "BAN": 60,
        "BAR": 60,
        "BAT": 60,
        "BED": 60,
        "BIG": 60,
        "BIT": 60,
        "BOW": 60,
        "BOX": 60,
        "BOY": 60,
        "BUD": 60,
        "BUG": 60,
        "BUS": 60,
        "BUT": 60,
        "BUY": 60,
        "CAB": 60,
        "CAN": 60,
        "CAP": 60,
        "CAR": 60,
        "CAT": 60,
        "COP": 60,
        "COT": 60,
        "COW": 60,
        "CRY": 60,
        "CUB": 60,
        "CUP": 60,
        "CUT": 60,
        "DAD": 60,
        "DAM": 60,
        "DAY": 60,
        "DIG": 60,
        "DIM": 60,
        "DIP": 60,
        "DOC": 60,
        "DOG": 60,
        "DOT": 60,
        "DRY": 60,
        "DUG": 60,
        "DYE": 60,
        "EAR": 60,
        "EAT": 60,
        "EEL": 60,
        "EGG": 60,
        "ELF": 60,
        "ELM": 60,
        "EMU": 60,
        "END": 60,
        "ERA": 60,
        "EVE": 60,
        "EWE": 60,
        "EYE": 60,
        "FAN": 60,
        "FAR": 60,
        "FAT": 60,
        "FED": 60,
        "FIG": 60,
        "FIN": 60,
        "FIT": 60,
        "FLY": 60,
        "FOG": 60,
        "FOR": 60,
        "FOX": 60,
        "FRY": 60,
        "FUN": 60,
        "FUR": 60,
        "GAP": 60,
        "GAS": 60,
        "GOD": 60,
        "GOT": 60,
        "GUM": 60,
        "GUN": 60,
        "GUT": 60,
        "GUY": 60,
        "GYM": 60,
        "HAD": 60,
        "HAM": 60,
        "HAS": 60,
        "HAT": 60,
        "HEN": 60,
        "HER": 60,
        "HID": 60,
        "HIM": 60,
        "HIP": 60,
        "HIS": 60,
        "HIT": 60,
        "HOG": 60,
        "HOP": 60,
        "HOT": 60,
        "HOW": 60,
        "HUB": 60,
        "HUG": 60,
        "HUM": 60,
    }
    return Dictionary(words)


@pytest.fixture
def real_dictionary(project_root) -> Dictionary:
    """Load the real Jeff Chen dictionary."""
    return Dictionary.load(
        project_root / "dictionaries" / "HggCuratedCrosswordList.txt",
        min_word_score=50,
    )


@pytest.fixture
def config() -> CSPFillerConfig:
    return CSPFillerConfig(timeout=10)


@pytest.fixture
def filler(config: CSPFillerConfig, real_dictionary: Dictionary) -> CSPFiller:
    return CSPFiller(config, real_dictionary)


class TestExtractSlots:
    def test_open_3x3(self) -> None:
        slots = _extract_slots(3, 3, set())
        across = [s for s in slots if s.direction == "across"]
        down = [s for s in slots if s.direction == "down"]
        assert len(across) == 3
        assert len(down) == 3
        assert all(s.length == 3 for s in slots)

    def test_with_black_cells(self) -> None:
        # 3x3 grid with center black cell
        black = {(1, 1)}
        slots = _extract_slots(3, 3, black)
        # Across: row 0 = 3, row 1 splits into (0) and (2) = both length 1 (skipped),
        # row 2 = 3. So 2 across slots.
        across = [s for s in slots if s.direction == "across"]
        down = [s for s in slots if s.direction == "down"]
        assert len(across) == 2  # rows 0 and 2
        assert len(down) == 2  # cols 0 and 2

    def test_crossings_populated(self) -> None:
        slots = _extract_slots(3, 3, set())
        # Each across slot crosses 3 down slots
        for s in slots:
            if s.direction == "across":
                assert len(s.crossings) == 3


class TestCSPFiller:
    def test_name(self, filler: CSPFiller) -> None:
        assert filler.name == "csp"

    def test_is_available(self, filler: CSPFiller) -> None:
        assert filler.is_available() is True

    def test_fill_3x3(
        self, config: CSPFillerConfig, small_dictionary: Dictionary
    ) -> None:
        filler = CSPFiller(config, small_dictionary)
        spec = GridSpec(rows=3, cols=3)
        result = filler.fill(spec, seed=42)
        assert len(result.grid) == 3
        assert len(result.grid[0]) == 3
        # All cells should be uppercase letters
        for row in result.grid:
            for cell in row:
                assert cell.isalpha() and cell.isupper()

    def test_fill_produces_valid_words(
        self, config: CSPFillerConfig, small_dictionary: Dictionary
    ) -> None:
        filler = CSPFiller(config, small_dictionary)
        spec = GridSpec(rows=3, cols=3)
        result = filler.fill(spec, seed=42)
        for word in result.words_across:
            assert small_dictionary.contains(word)
        for word in result.words_down:
            assert small_dictionary.contains(word)

    def test_fill_5x5_open(self, filler: CSPFiller) -> None:
        spec = GridSpec(rows=5, cols=5)
        result = filler.fill(spec, seed=42)
        assert len(result.grid) == 5
        assert len(result.grid[0]) == 5
        for row in result.grid:
            for cell in row:
                assert cell.isalpha() and cell.isupper()

    def test_fill_5x5_valid_words(
        self, filler: CSPFiller, real_dictionary: Dictionary
    ) -> None:
        spec = GridSpec(rows=5, cols=5)
        result = filler.fill(spec, seed=42)
        for word in result.words_across + result.words_down:
            assert real_dictionary.contains(word), f"{word} not in dictionary"

    def test_fill_reproducible_with_seed(self, filler: CSPFiller) -> None:
        spec = GridSpec(rows=5, cols=5)
        result1 = filler.fill(spec, seed=123)
        result2 = filler.fill(spec, seed=123)
        assert result1.grid == result2.grid

    def test_fill_different_seeds(self, filler: CSPFiller) -> None:
        spec = GridSpec(rows=5, cols=5)
        result1 = filler.fill(spec, seed=1)
        result2 = filler.fill(spec, seed=2)
        assert result1.grid != result2.grid

    def test_fill_respects_black_cells(self, filler: CSPFiller) -> None:
        spec = GridSpec(rows=5, cols=5, black_cells=[(0, 4), (4, 0)])
        result = filler.fill(spec, seed=42)
        assert result.grid[0][4] == "."
        assert result.grid[4][0] == "."
        # Other cells should be letters
        assert result.grid[0][0].isalpha()

    def test_timeout_raises(self, real_dictionary: Dictionary) -> None:
        config = CSPFillerConfig(timeout=0)  # immediate timeout
        filler = CSPFiller(config, real_dictionary)
        spec = GridSpec(rows=5, cols=5)
        with pytest.raises(FillError, match="timed out"):
            filler.fill(spec, seed=42)

    def test_no_words_of_length_raises(
        self, config: CSPFillerConfig
    ) -> None:
        # Dictionary with only 3-letter words
        tiny_dict = Dictionary({"CAT": 60, "DOG": 60, "BAT": 60})
        filler = CSPFiller(config, tiny_dict)
        spec = GridSpec(rows=5, cols=5)  # needs 5-letter words
        with pytest.raises(FillError, match="No dictionary words of length 5"):
            filler.fill(spec, seed=42)

    def test_words_across_and_down_populated(self, filler: CSPFiller) -> None:
        spec = GridSpec(rows=5, cols=5)
        result = filler.fill(spec, seed=42)
        assert len(result.words_across) == 5
        assert len(result.words_down) == 5

    def test_timeout_by_size_overrides_default(
        self, real_dictionary: Dictionary
    ) -> None:
        config = CSPFillerConfig(timeout=10, timeout_by_size={5: 0})
        filler = CSPFiller(config, real_dictionary)
        spec = GridSpec(rows=5, cols=5)
        # timeout_by_size[5]=0 should cause immediate timeout
        with pytest.raises(FillError, match="timed out"):
            filler.fill(spec, seed=42)

    def test_timeout_falls_back_to_default(
        self, real_dictionary: Dictionary
    ) -> None:
        # timeout_by_size has no entry for size 5, so default timeout=10 is used
        config = CSPFillerConfig(timeout=10, timeout_by_size={7: 120})
        filler = CSPFiller(config, real_dictionary)
        spec = GridSpec(rows=5, cols=5)
        # Should succeed with default 10s timeout
        result = filler.fill(spec, seed=42)
        assert len(result.grid) == 5

    def test_fill_7x7(self, real_dictionary: Dictionary) -> None:
        config = CSPFillerConfig(timeout=30)
        filler = CSPFiller(config, real_dictionary)
        # Use black cells to break up the constraint graph
        spec = GridSpec(
            rows=7, cols=7, black_cells=[(0, 3), (3, 0), (3, 6), (6, 3)]
        )
        result = filler.fill(spec, seed=42)
        assert len(result.grid) == 7
        assert len(result.grid[0]) == 7
        for r, row in enumerate(result.grid):
            for c, cell in enumerate(row):
                if (r, c) in {(0, 3), (3, 0), (3, 6), (6, 3)}:
                    assert cell == "."
                else:
                    assert cell.isalpha() and cell.isupper()

    def test_fill_7x7_valid_words(self, real_dictionary: Dictionary) -> None:
        config = CSPFillerConfig(timeout=30)
        filler = CSPFiller(config, real_dictionary)
        spec = GridSpec(
            rows=7, cols=7, black_cells=[(0, 3), (3, 0), (3, 6), (6, 3)]
        )
        result = filler.fill(spec, seed=42)
        for word in result.words_across + result.words_down:
            assert real_dictionary.contains(word), f"{word} not in dictionary"

    def test_quality_tiers_improve_scores(
        self, real_dictionary: Dictionary
    ) -> None:
        """Fill with quality tiers [60, 50] should produce higher scores than [50]."""
        spec = GridSpec(rows=5, cols=5)

        # With tiers (tries score-60 first)
        config_tiered = CSPFillerConfig(timeout=10, quality_tiers=[60, 50])
        filler_tiered = CSPFiller(config_tiered, real_dictionary)
        result_tiered = filler_tiered.fill(spec, seed=42)

        # Without tiers (score-50 only)
        config_flat = CSPFillerConfig(timeout=10, quality_tiers=[50])
        filler_flat = CSPFiller(config_flat, real_dictionary)
        result_flat = filler_flat.fill(spec, seed=42)

        def avg_score(result: FilledGrid) -> float:
            words = result.words_across + result.words_down
            scores = [real_dictionary.score(w) or 0 for w in words]
            return sum(scores) / len(scores)

        tiered_avg = avg_score(result_tiered)
        flat_avg = avg_score(result_flat)
        # Tiered should be at least as good
        assert tiered_avg >= flat_avg

    def test_single_tier_50_works(self, real_dictionary: Dictionary) -> None:
        """Single tier [50] should still produce a valid fill."""
        config = CSPFillerConfig(timeout=10, quality_tiers=[50])
        filler = CSPFiller(config, real_dictionary)
        spec = GridSpec(rows=5, cols=5)
        result = filler.fill(spec, seed=42)
        assert len(result.grid) == 5
        for word in result.words_across + result.words_down:
            assert real_dictionary.contains(word)
