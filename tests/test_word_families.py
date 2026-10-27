"""Tests for same-root word families (within-grid duplicate rule)."""

from __future__ import annotations

import pytest

from crossword_generator.word_families import (
    family_keys,
    load_root_lexicon,
    same_root_pairs,
)

# Every word below is a real lexicon entry so each case exercises the
# morphology, not a missing-lemma shortcut.
_LEXICON = frozenset(
    {
        "ATE",
        "BAR",
        "BARE",
        "BARED",
        "BARRED",
        "BET",
        "BETTER",
        "BIG",
        "BIGGEST",
        "BONE",
        "BONY",
        "BUS",
        "BUSY",
        "CANDY",
        "CAR",
        "CARD",
        "COME",
        "COMEDIAN",
        "COMEDIANS",
        "COMEDIC",
        "COMEDY",
        "EAR",
        "EARLY",
        "EAT",
        "EATER",
        "ERIC",
        "ERICA",
        "FOG",
        "FOGGY",
        "HAPPILY",
        "HAPPY",
        "HIRE",
        "HIRED",
        "HIRING",
        "ICE",
        "ICED",
        "ICY",
        "MANE",
        "MANY",
        "MEET",
        "MET",
        "NOS",
        "NOSY",
        "OWE",
        "OWED",
        "PARTIES",
        "PARTY",
        "PLAY",
        "PLAYED",
        "PLAYERS",
        "RED",
        "REDO",
        "RID",
        "RIDDING",
        "RIDE",
        "RIDES",
        "SAID",
        "SAYS",
        "SAY",
        "STAR",
        "START",
        "STINK",
        "STINKS",
        "STINKY",
        "STONE",
        "STONY",
        "STOP",
        "STOPPED",
        "TEETH",
        "TOOTH",
    }
)


def _pairs(*words: str) -> list[tuple[str, str]]:
    return same_root_pairs(words, _LEXICON.__contains__)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # The four pairs Jeff hand-edited out of the 2026-09 midi batch.
        ("ATE", "EAT"),
        ("OWE", "OWED"),
        ("HIRED", "HIRING"),
        ("COMEDIANS", "COMEDIC"),
        # Earlier edits from his review history.
        ("SAID", "SAYS"),
        ("ICE", "ICED"),
        ("MEET", "MET"),
        # Regular morphology the old rule missed.
        ("PLAY", "PLAYED"),
        ("STOP", "STOPPED"),
        ("RID", "RIDDING"),
        ("BIG", "BIGGEST"),
        ("EAT", "EATER"),
        ("PARTIES", "PARTY"),
        ("HAPPILY", "HAPPY"),
        ("STONE", "STONY"),
        ("TEETH", "TOOTH"),
        # Curated -Y adjectives of short lemmas.
        ("ICE", "ICY"),
        ("BONE", "BONY"),
        ("FOG", "FOGGY"),
        # Shared lemma that is not itself in the grid.
        ("PLAYED", "PLAYERS"),
        ("STINKS", "STINKY"),
    ],
)
def test_same_root_pairs_are_caught(left: str, right: str) -> None:
    assert _pairs(left, right) == [tuple(sorted((left, right)))]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("START", "STAR"),  # START is not STAR + a suffix
        ("RID", "RIDES"),  # RIDES is RIDE + S; -ES needs a sibilant
        ("BUS", "BUSY"),  # no -Y on three-letter lemmas
        ("EAR", "EARLY"),  # no -LY on three-letter lemmas
        ("BET", "BETTER"),  # doubled -ER is a different word
        ("BAR", "BARED"),  # BARED is BARE + D; BAR + ED would be BARRED
        ("CAR", "CARD"),  # -D only after a final E
        ("MANE", "MANY"),  # MANY is not MANE + Y
        ("NOS", "NOSY"),  # NOSY is curated under NOSE only
        ("RED", "REDO"),
        ("ERIC", "ERICA"),
        ("COME", "COMEDIC"),
    ],
)
def test_coincidental_overlaps_are_not_pairs(left: str, right: str) -> None:
    assert _pairs(left, right) == []


def test_terminal_s_pairs_are_left_to_the_terminal_s_rule() -> None:
    assert _pairs("STINK", "STINKS") == []


def test_lemma_must_be_a_lexicon_word() -> None:
    # Without HIRE in the lexicon HIRED and HIRING share nothing.
    is_word = (_LEXICON - {"HIRE"}).__contains__
    assert same_root_pairs(["HIRED", "HIRING"], is_word) == []


def test_two_step_derivation() -> None:
    keys = family_keys("COMEDIANS", _LEXICON.__contains__)
    assert {"COMEDIANS", "COMEDIAN", "COMEDY"} <= keys


def test_load_root_lexicon_reads_scored_and_bare_lists(tmp_path) -> None:
    scored = tmp_path / "scored.txt"
    scored.write_text("hire;50\nOWE;60\nnot a word;50\n", encoding="utf-8")
    bare = tmp_path / "bare.txt"
    bare.write_text("Comedy\n", encoding="utf-8")

    lexicon = load_root_lexicon((str(scored), str(bare), str(tmp_path / "missing.txt")))

    assert lexicon == frozenset({"HIRE", "OWE", "COMEDY"})
