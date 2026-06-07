"""Tests for the deterministic clue-leak detector."""

from __future__ import annotations

import pytest

from crossword_generator.graders.leak_detector import (
    LeakFinding,
    detect_leak,
    detect_leaks,
)
from crossword_generator.models import ClueEntry

# --- Positive cases: clues that leak their answer ---------------------------

LEAKS = [
    # (answer, clue, expected kind)
    ("OCEAN", "The ocean is vast", "exact"),
    ("RAP", "A bad rap, so to speak", "exact"),
    ("TEACHER", "One who teaches math", "shared_root"),
    ("TEACHES", "What a teacher does", "shared_root"),
    ("BAKING", "What a baker does", "shared_root"),
    ("RUNNER", "One running a race", "shared_root"),
    ("WRITER", "One who writes books", "shared_root"),
    ("EDITOR", "One who edits copy", "shared_root"),
    ("TRIMS", "Cuts the hedge; a trim job", "shared_root"),
    ("CARE", "Look after; careful now", "shared_root"),
    ("HEAT", "Warm up; a heated debate", "shared_root"),
    ("HOPE", "A hopeless cause", "shared_root"),
    ("WIFE", "Plot of a housewife", "shared_root"),
    ("CHILD", "A group of children", "irregular"),
    ("WIVES", "In a wifely manner", "irregular"),
    ("EST", "Eastern Standard Time, for short", "abbrev_expansion"),
    ("OPS", "Military operations, briefly", "abbrev_expansion"),
    ("POL", "A political figure, informally", "abbrev_expansion"),
    ("NASA", "National Aeronautics and Space Administration", "abbrev_expansion"),
]


@pytest.mark.parametrize("answer,clue,kind", LEAKS)
def test_detects_leak(answer: str, clue: str, kind: str) -> None:
    finding = detect_leak(answer, clue)
    assert finding is not None, f"expected leak for {answer} :: {clue}"
    assert finding.kind == kind


# --- Negative cases: clean clues and coincidental substrings ----------------

CLEAN = [
    ("ACHE", "Dull, persistent pain"),
    ("RAP", "Hip-hop music genre"),
    ("SEE", "Grasp the meaning of"),
    ("EDGE", "Border or margin"),
    ("BEAR", "Grin and ___ it"),
    ("ART", "Painting or sculpture"),
    # Coincidental substrings that must NOT flag.
    ("CARD", "Cardiac ward, informally"),
    ("READ", "A ready response"),
    ("PART", "Departure lounge"),
    ("RATE", "Berate harshly"),
    ("STAR", "Started the engine"),
    ("MAIN", "Remaining tasks"),
    ("OVER", "Discover the truth"),
    ("PAIN", "A painter at work"),
    # Stopword-length answers must never crash or false-flag.
    ("AN", "Indefinite article"),
    ("IT", "Pronoun for an object"),
]


@pytest.mark.parametrize("answer,clue", CLEAN)
def test_clean_clue_not_flagged(answer: str, clue: str) -> None:
    assert detect_leak(answer, clue) is None


def test_empty_clue_is_clean() -> None:
    assert detect_leak("OCEAN", "") is None


def test_case_insensitive() -> None:
    assert detect_leak("ocean", "The OCEAN is deep") is not None


# --- detect_leaks over a clue list ------------------------------------------


def test_detect_leaks_returns_findings_with_positions() -> None:
    clues = [
        ClueEntry(number=1, direction="across", answer="OCEAN", clue="Vast ocean"),
        ClueEntry(number=2, direction="down", answer="ACHE", clue="A dull pain"),
        ClueEntry(
            number=3,
            direction="across",
            answer="EST",
            clue="Eastern Standard Time abbr.",
        ),
    ]
    findings = detect_leaks(clues)
    assert len(findings) == 2
    by_answer = {f.answer: f for f in findings}
    assert by_answer["OCEAN"].number == 1
    assert by_answer["OCEAN"].direction == "across"
    assert by_answer["EST"].kind == "abbrev_expansion"
    assert all(isinstance(f, LeakFinding) for f in findings)


def test_detect_leaks_skips_empty_clues() -> None:
    clues = [ClueEntry(number=1, direction="across", answer="OCEAN", clue="")]
    assert detect_leaks(clues) == []


def test_initialism_leak() -> None:
    # Clue whose leading letters spell the answer.
    finding = detect_leak("UCLA", "University of California Los Angeles, informally")
    assert finding is not None
    assert finding.kind == "abbrev_expansion"
