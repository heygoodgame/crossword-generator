"""Tests for the deterministic wrong-hyphen detector."""

from __future__ import annotations

import pytest

from crossword_generator.dictionary import Dictionary
from crossword_generator.graders.hyphen_detector import (
    detect_hyphen,
    detect_hyphens,
)
from crossword_generator.models import ClueEntry

# Holds the open-compound halves (so "Paper crane" is fixable), a couple of
# genuinely-hyphenated words as whole units, and the single-letter case.
_DICT = Dictionary(
    {
        w: 60
        for w in [
            "PAPER", "CRANE",
            "LINGERIE", "DRAWER",
            "SNOOZE", "BUTTON",
            "ICE", "CREAM",
            # Genuinely-hyphenated words present as whole units -> not suspect.
            "WELL-KNOWN", "SELF-ESTEEM", "X-RAY",
            # Halves of a genuine hyphenation whose unit is also listed.
            "WELL", "KNOWN", "SELF", "ESTEEM",
            # A word with no real partner half (so "paper-zzz" stays clean).
            "RAY",
        ]
    },
    min_word_score=0,
    min_2letter_score=0,
)


SUSPECT = [
    ("CRANE", "Paper-crane, folded", "Paper-crane", "Paper crane"),
    ("DRAWER", "Lingerie-drawer item", "Lingerie-drawer", "Lingerie drawer"),
    ("BUTTON", "Snooze-button presser", "Snooze-button", "Snooze button"),
    ("CREAM", "Ice-cream cone", "Ice-cream", "Ice cream"),
]


@pytest.mark.parametrize("answer,clue,token,suggestion", SUSPECT)
def test_suspect_hyphen_flagged(
    answer: str, clue: str, token: str, suggestion: str
) -> None:
    finding = detect_hyphen(answer, clue, _DICT)
    assert finding is not None, f"expected hyphen flag for {clue!r}"
    assert finding.token == token
    assert finding.suggestion == suggestion


CLEAN = [
    # Genuine hyphenated words (present as whole units) are left alone.
    ("AUTHOR", "Well-known author"),
    ("CONFIDENCE", "Self-esteem, informally"),
    ("SCAN", "X-ray, for one"),
    # No hyphen at all.
    ("OCEAN", "Vast body of saltwater"),
    # A half that is not a real word -> not an obvious space-substitution fix.
    ("CRANE", "Paper-zzz nonsense"),
    # A dash used as punctuation (spaces around it) is not a word-internal
    # hyphen and must not match.
    ("CITY", "Capital city - the big one"),
]


@pytest.mark.parametrize("answer,clue", CLEAN)
def test_clean_clue_not_flagged(answer: str, clue: str) -> None:
    assert detect_hyphen(answer, clue, _DICT) is None


def test_detect_hyphens_fills_number_and_direction() -> None:
    clues = [
        ClueEntry(
            number=3, direction="across", answer="CRANE", clue="Paper-crane, folded"
        ),
        ClueEntry(
            number=5, direction="down", answer="OCEAN", clue="Vast saltwater"
        ),
    ]
    findings = detect_hyphens(clues, _DICT)
    assert len(findings) == 1
    assert findings[0].number == 3
    assert findings[0].direction == "across"
    assert findings[0].token == "Paper-crane"
