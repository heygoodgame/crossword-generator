"""Tests for deterministic clue house-style fixes."""

from __future__ import annotations

import pytest

from crossword_generator.clue_style import (
    is_question_mark_clue,
    normalize_abbreviation_tags,
    normalize_clue_style,
    spell_out_examples,
)


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


@pytest.mark.parametrize(
    ("clue", "expected"),
    [
        ("Sedan, e.g.", "Sedan, for example"),
        ('Said "not guilty," e.g.', 'Said "not guilty," for example'),
        ("Dungeons & Dragons, e.g. (abbr.)", "Dungeons & Dragons, for example (abbr.)"),
        ("Clairvoyance, e.g., for short", "Clairvoyance, for example, for short"),
        ("___ park (Disneyland, e.g.)", "___ park (Disneyland, for example)"),
        ("Trees (e.g., oaks)", "Trees (for example, oaks)"),
        ("E.g., a sedan", "For example, a sedan"),
        ("Sedan, e. g.", "Sedan, for example"),
        ("Sedan, for example", "Sedan, for example"),
        ("Keg party", "Keg party"),
    ],
)
def test_spell_out_examples(clue: str, expected: str) -> None:
    assert spell_out_examples(clue) == expected


def test_normalize_clue_style_applies_every_fix() -> None:
    assert normalize_clue_style("Org., e.g., briefly") == "Org., for example, for short"


@pytest.mark.parametrize(
    ("clue", "expected"),
    [
        ("Plant manager?", True),
        ("Plant manager? ", True),
        ("Coat room, for short?", True),
        ("Union member? (abbr.)", True),
        ("Party line?, for short", True),
        # Literal questions are fine.
        ("___ you okay?", False),
        ('"Are we there yet?"', False),
        ('Kid\'s "Are we there yet?"', False),
        ("Person who tends roses", False),
    ],
)
def test_is_question_mark_clue(clue: str, expected: bool) -> None:
    assert is_question_mark_clue(clue) is expected
