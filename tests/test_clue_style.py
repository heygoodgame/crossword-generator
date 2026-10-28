"""Tests for deterministic clue house-style fixes."""

from __future__ import annotations

import pytest

from crossword_generator.clue_style import normalize_abbreviation_tags


@pytest.mark.parametrize(
    ("clue", "expected"),
    [
        ("Knee part that can tear, briefly", "Knee part that can tear, for short"),
        ("Flight-board figure, in brief", "Flight-board figure, for short"),
        ("Hosp. area, Briefly", "Hosp. area, for short"),
        ("Genetic stuff (briefly)", "Genetic stuff (abbr.)"),
        ("Genetic stuff (In Brief)", "Genetic stuff (abbr.)"),
        ("Pal (buddy, briefly)", "Pal (buddy, for short)"),
        # Not an abbreviation tag: "briefly" as "for a short time".
        ("Stop briefly", "Stop briefly"),
        ("Pause briefly, then go", "Pause briefly, then go"),
        ("Genetic stuff (abbr.)", "Genetic stuff (abbr.)"),
        ("Trig ratio, for short", "Trig ratio, for short"),
    ],
)
def test_normalize_abbreviation_tags(clue: str, expected: str) -> None:
    assert normalize_abbreviation_tags(clue) == expected
