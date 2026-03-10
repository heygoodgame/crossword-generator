"""Tests for CSP filler with seed entry constraints."""

from __future__ import annotations

import pytest

from crossword_generator.config import CSPFillerConfig
from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FillError, GridSpec
from crossword_generator.fillers.csp import CSPFiller, extract_slots


@pytest.fixture
def real_dictionary(project_root) -> Dictionary:
    """Load the real Jeff Chen dictionary."""
    return Dictionary.load(
        project_root / "dictionaries" / "HggCuratedCrosswordList.txt",
        min_word_score=50,
    )


@pytest.fixture
def config() -> CSPFillerConfig:
    return CSPFillerConfig(timeout=30)


@pytest.fixture
def filler(config: CSPFillerConfig, real_dictionary: Dictionary) -> CSPFiller:
    return CSPFiller(config, real_dictionary)


class TestCSPSeedEntries:
    def test_single_seed_entry_placed(
        self, filler: CSPFiller, real_dictionary: Dictionary
    ) -> None:
        """CSP fill with a single seed entry places it correctly."""
        spec = GridSpec(rows=5, cols=5)
        # Place OCEAN in the first across slot (row 0, col 0)
        spec.seed_entries = {"0,0,across": "OCEAN"}
        result = filler.fill(spec, seed=42)

        # Read the word from row 0
        row0_word = "".join(result.grid[0])
        assert row0_word == "OCEAN"

    def test_multiple_seed_entries(
        self, filler: CSPFiller, real_dictionary: Dictionary
    ) -> None:
        """Multiple seed entries all get placed correctly."""
        spec = GridSpec(rows=5, cols=5)

        # First fill normally to get known-good words
        baseline = filler.fill(spec, seed=42)
        word_row0 = "".join(baseline.grid[0])
        word_col0 = "".join(baseline.grid[r][0] for r in range(5))

        # Re-fill with those as seed entries
        spec2 = GridSpec(rows=5, cols=5)
        spec2.seed_entries = {
            "0,0,across": word_row0,
            "0,0,down": word_col0,
        }
        result = filler.fill(spec2, seed=99)

        row0 = "".join(result.grid[0])
        col0 = "".join(result.grid[r][0] for r in range(5))
        assert row0 == word_row0
        assert col0 == word_col0

    def test_fill_still_produces_valid_grid(
        self, filler: CSPFiller, real_dictionary: Dictionary
    ) -> None:
        """Grid fill around seed entries still produces valid dictionary words."""
        spec = GridSpec(rows=5, cols=5)
        spec.seed_entries = {"0,0,across": "OCEAN"}
        result = filler.fill(spec, seed=42)

        # All words (including non-seed) should be in dictionary
        for word in result.words_across + result.words_down:
            assert real_dictionary.contains(word), f"{word} not in dictionary"

    def test_seed_entry_with_black_cells(
        self, filler: CSPFiller, real_dictionary: Dictionary
    ) -> None:
        """Seed entries work with black cells in the grid."""
        spec = GridSpec(
            rows=5, cols=5,
            black_cells=[(0, 0), (4, 4)],
        )
        # First across slot on row 0 starts at col 1, length 4
        slots = extract_slots(5, 5, {(0, 0), (4, 4)})
        across_row0 = [
            s for s in slots
            if s.direction == "across" and s.row == 0
        ]
        assert len(across_row0) == 1
        slot = across_row0[0]

        # Find a 4-letter word in the dictionary
        words = real_dictionary.words_by_length(slot.length)
        assert len(words) > 0
        seed_word = words[0]

        spec.seed_entries = {
            f"{slot.row},{slot.col},{slot.direction}": seed_word
        }
        result = filler.fill(spec, seed=42)

        # Read the word from that slot
        placed = "".join(
            result.grid[slot.row][c]
            for c in range(slot.col, slot.col + slot.length)
        )
        assert placed == seed_word

    def test_infeasible_seed_entry_length_mismatch(
        self, filler: CSPFiller
    ) -> None:
        """Seed entry with wrong length reports error."""
        spec = GridSpec(rows=5, cols=5)
        # Row 0 across is length 5, but we give a 3-letter word
        spec.seed_entries = {"0,0,across": "CAT"}
        with pytest.raises(FillError, match="length"):
            filler.fill(spec, seed=42)

    def test_invalid_seed_entry_key(self, filler: CSPFiller) -> None:
        """Invalid seed entry key format raises error."""
        spec = GridSpec(rows=5, cols=5)
        spec.seed_entries = {"bad_key": "OCEAN"}
        with pytest.raises(FillError, match="Invalid seed entry key"):
            filler.fill(spec, seed=42)

    def test_seed_entry_no_matching_slot(self, filler: CSPFiller) -> None:
        """Seed entry for nonexistent slot raises error."""
        spec = GridSpec(rows=5, cols=5)
        # No slot starts at (0, 3) across in a 5x5 open grid
        # Actually (0,0) across covers the whole row, so (0,3,across) doesn't exist
        spec.seed_entries = {"0,3,across": "AB"}
        with pytest.raises(FillError, match="No slot found"):
            filler.fill(spec, seed=42)

    def test_down_seed_entry(
        self, filler: CSPFiller, real_dictionary: Dictionary
    ) -> None:
        """Seed entries in the down direction work correctly."""
        spec = GridSpec(rows=5, cols=5)
        # Column 0, down direction = length 5
        spec.seed_entries = {"0,0,down": "OCEAN"}
        result = filler.fill(spec, seed=42)

        # Read column 0
        col0 = "".join(result.grid[r][0] for r in range(5))
        assert col0 == "OCEAN"

        # All words valid
        for word in result.words_across + result.words_down:
            assert real_dictionary.contains(word), f"{word} not in dictionary"
