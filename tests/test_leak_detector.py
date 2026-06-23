"""Tests for the deterministic clue-leak detector."""

from __future__ import annotations

import pytest

from crossword_generator.dictionary import Dictionary
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
    # Abbreviation expansion-WORD leaks (full strictness — any expansion word,
    # even a generic one, is banned). These shipped in the first batch.
    ("ETA", "Arrival time abbreviation", "abbrev_expansion_word"),
    ("ETA", "Arrival time, for short", "abbrev_expansion_word"),
    ("ETA", "Departure time, roughly", "abbrev_expansion_word"),  # 'time' alone
    ("MCL", "Knee ligament, briefly", "abbrev_expansion_word"),
    ("CPR", "Resuscitation technique, for short", "abbrev_expansion_word"),
    ("GPA", "Student's average, for short", "abbrev_expansion_word"),
    ("ATM", "Cash machine, in brief", "abbrev_expansion_word"),
    ("DOB", "Birth date, on a form", "abbrev_expansion_word"),
    # Shared-prefix / etymology / spelling-fragment leaks (aggressive rule).
    # Jeff's reported cases:
    ("NAVAL", "Of the navy", "shared_prefix"),
    ("TRI", 'Start of "triangle"', "shared_prefix"),
    # Same class:
    ("KNEE", "Joint that bends when you kneel", "shared_prefix"),
    ("PRE", "Prefix meaning before", "shared_prefix"),
    ("ART", "Painting at an artist's studio", "shared_prefix"),
    ("SUN", "Sunday, informally", "shared_prefix"),
    # Accepted false positive (max-recall policy): read/ready dominate.
    ("READ", "A ready response", "shared_prefix"),
    # Compound-containment leaks: a clue word is the leading or trailing
    # component of the answer. Jeff's reported case from the 2026-06 batch:
    ("SCOFFLAW", "One who flouts laws", "compound_containment"),
    # Same class:
    ("SCOFFLAW", "Habitual breaker of the law", "compound_containment"),
    ("LAWSUIT", "Court filing over a broken law", "compound_containment"),
    ("OUTLAWS", "Bandits outside the law", "compound_containment"),
    ("CARPET", "Floor covering a pet might nap on", "compound_containment"),
    # Embedded-substring leaks: a short answer hidden inside a longer,
    # related clue word the reverse-compound rule cannot split. Jeff's
    # reported cases from the week-20260716 batch.
    ("BOT", "Robot, informally", "embedded_substring"),
    ("TIL", "Until, briefly", "embedded_substring"),
    ("MEN", "Gentlemen", "embedded_substring"),
    ("MAN", "Gentleman caller", "embedded_substring"),
    ("BOT", "Robotic vacuum, casually", "embedded_substring"),
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
    # Coincidental substrings that must NOT flag (prefix is a small part of the
    # longer clue word, so the shared-prefix rule correctly leaves them alone).
    ("CARD", "Cardiac ward, informally"),
    ("PART", "Departure lounge"),
    ("RATE", "Berate harshly"),
    ("STAR", "Started the engine"),
    ("MAIN", "Remaining tasks"),
    ("OVER", "Discover the truth"),
    ("PAIN", "A painter at work"),
    # Compound containment must not fire on short non-content words that
    # embed coincidentally, nor when the leftover fragment is tiny.
    ("THRONE", "One seat fit for royalty"),
    ("TEACHER", "She guides her students"),
    ("PHONE", "One way to reach someone"),
    ("SCOFFLAW", "Habitual rule-breaker"),
    # Stopword-length answers must never crash or false-flag.
    ("AN", "Indefinite article"),
    ("IT", "Pronoun for an object"),
    # Fair abbreviation clues — no expansion word present, must NOT flag.
    ("ETA", "Flight-board figure, for short"),
    ("MCL", "Knee injury site, in brief"),
    ("CPR", "Lifesaving technique, briefly"),
    ("GPA", "Report-card stat, for short"),
    ("ATM", "Convenience-store cash dispenser, briefly"),
    ("DOB", "Form datum, for short"),
    # Embedded-substring answers clued through their own meaning — must NOT
    # flag just because the short string could embed in some other word.
    ("BOT", "Automated chat program, for short"),
    ("TIL", "Up to that point, briefly"),
    ("MEN", "Adult males"),
    ("MEN", "Restroom door label"),
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


# --- Reverse-compound rule (needs a dictionary) -----------------------------

# A tiny dictionary holding only the answers, the offending compounds, and the
# real-word remainders that justify a compound split.
_REVERSE_COMPOUND_DICT = Dictionary(
    {
        w: 60
        for w in [
            "ROBE", "BATHROBE", "BATH",
            "BERRY", "STRAWBERRY", "STRAW",
            "CAKE", "CHEESECAKE", "CHEESE",
            "FISH", "SWORDFISH", "SWORD",
            "PORT", "AIRPORT", "AIR",
            # Coincidence-control entries: the remainder is NOT a word.
            "CARD", "CARDIAC",
        ]
    },
    min_word_score=0,
    min_2letter_score=0,
)

REVERSE_COMPOUND_LEAKS = [
    ("ROBE", "Bathrobe, for example"),
    ("BERRY", "Strawberry, e.g."),
    ("CAKE", "Cheesecake type"),
    ("FISH", "Swordfish, for one"),
    ("PORT", "Airport posting"),
]


@pytest.mark.parametrize("answer,clue", REVERSE_COMPOUND_LEAKS)
def test_reverse_compound_leak_detected(answer: str, clue: str) -> None:
    finding = detect_leak(answer, clue, _REVERSE_COMPOUND_DICT)
    assert finding is not None, f"expected leak for {answer} :: {clue}"
    assert finding.kind == "reverse_compound"


def test_reverse_compound_skipped_without_dictionary() -> None:
    # The rule only runs when a dictionary is supplied; otherwise the clue is
    # clean as far as the other rules are concerned.
    assert detect_leak("ROBE", "Bathrobe, for example") is None


def test_reverse_compound_ignores_non_word_remainder() -> None:
    # CARDIAC is not CARD + a dictionary word, so it must not flag.
    assert detect_leak("CARD", "Cardiac muscle, e.g.", _REVERSE_COMPOUND_DICT) is None


def test_reverse_compound_flags_short_answer_in_real_compound() -> None:
    # 3-letter answers ARE checked now: HOUSECAT splits into HOUSE + CAT, both
    # real words, so it flags. Per Jeff's preference, over-rejecting short
    # answers in genuine compounds beats shipping the spelled-out answer; the
    # occasional coincidence (PALACE/ace) just gets regenerated.
    d = Dictionary(
        {"CAT": 60, "HOUSECAT": 60, "HOUSE": 60},
        min_word_score=0,
        min_2letter_score=0,
    )
    finding = detect_leak("CAT", "Housecat, for one", d)
    assert finding is not None
    assert finding.kind == "reverse_compound"


def test_reverse_compound_short_answer_nonword_remainder_safe() -> None:
    # A 3-letter answer whose compound remainder is NOT a real word stays clean,
    # so coincidental substrings (EAR in NUCLEAR -> "nucl") do not flag.
    d = Dictionary(
        {"EAR": 60, "NUCLEAR": 60},
        min_word_score=0,
        min_2letter_score=0,
    )
    assert detect_leak("EAR", "Nuclear option, so to speak", d) is None


# --- Shared-edge-fragment rule (needs a dictionary) -------------------------

# Holds the compound answers, their component parts, and the clue words that
# share an edge component with them.
_SHARED_FRAGMENT_DICT = Dictionary(
    {
        w: 60
        for w in [
            "MINI", "FIG", "FIGURE", "FIGURINE",
            "GRAND", "SON", "PERSON",
            # Coincidence-control: PAPER does not split into a real word + the
            # "per" that PERSON shares, so WALLPAPER must NOT flag against it.
            "PAPER", "WALL",
        ]
    },
    min_word_score=0,
    min_2letter_score=0,
)


@pytest.mark.parametrize(
    "answer,clue",
    [
        ("MINIFIG", "A small plastic figure"),
        ("MINIFIG", "Tiny collectible figure"),
        ("MINIFIG", "Little figurine in a set"),
        ("GRANDSON", "A young person in the family"),
    ],
)
def test_shared_fragment_leak_detected(answer: str, clue: str) -> None:
    finding = detect_leak(answer, clue, _SHARED_FRAGMENT_DICT)
    assert finding is not None, f"expected leak for {answer} :: {clue}"
    assert finding.kind == "shared_fragment"


def test_shared_fragment_skipped_without_dictionary() -> None:
    # The rule only runs when a dictionary is supplied.
    assert detect_leak("MINIFIG", "A small plastic figure") is None


def test_shared_fragment_requires_genuine_compound_answer() -> None:
    # WALLPAPER and "person" both touch "per", but PAPER does not split into a
    # real word + "per", so the answer is not a genuine compound at that edge
    # and the coincidental fragment must not flag.
    assert detect_leak("WALLPAPER", "A famous person", _SHARED_FRAGMENT_DICT) is None
