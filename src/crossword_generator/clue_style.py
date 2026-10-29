"""Deterministic house-style fixes applied to every generated clue and hint."""

from __future__ import annotations

import re

# Jeff (2026-10-27): tag abbreviation answers with "(abbr.)" or ", for short".
# Many solvers don't read "briefly" or "in brief" as an abbreviation signal,
# and he rewrote 53 of them by hand. Only the trailing tag form is rewritten,
# so a clue that means "for a short time" ("Stop briefly") is left alone.
_TRAILING_BRIEF_TAG = re.compile(
    r",\s*(?:briefly|in brief)(?=\s*$|\s*\))", re.IGNORECASE
)
_PAREN_BRIEF_TAG = re.compile(r"\(\s*(?:briefly|in brief)\s*\)", re.IGNORECASE)

# Jeff (2026-10-29): "Many people wonder WTF is 'e.g.'" Spell it out as "for
# example". Only the abbreviation itself is swapped, so the punctuation around
# it stays put: "Sedan, e.g." and '"Not guilty," e.g.' both keep their comma.
_EG = re.compile(r"\b(e)\.\s?g\.", re.IGNORECASE)

# A clue ending in "?" signals a pun or trick, and Jeff (2026-10-29) wants
# none of them. Strip trailing abbreviation tags before looking.
_TRAILING_TAG = re.compile(r"\s*(?:\([^()]*\)|,\s*for short)\s*$", re.IGNORECASE)


def normalize_clue_style(clue: str) -> str:
    """Apply every deterministic house-style fix to a clue or hint."""
    return normalize_abbreviation_tags(spell_out_examples(clue))


def normalize_abbreviation_tags(clue: str) -> str:
    """Swap "briefly" / "in brief" abbreviation tags for the house tags."""
    clue = _PAREN_BRIEF_TAG.sub("(abbr.)", clue)
    return _TRAILING_BRIEF_TAG.sub(", for short", clue)


def spell_out_examples(clue: str) -> str:
    """Replace "e.g." with "for example", keeping a leading capital."""
    return _EG.sub(
        lambda m: "For example" if m.group(1) == "E" else "for example", clue
    )


def is_question_mark_clue(clue: str) -> bool:
    """True for a pun-style clue that ends in a question mark.

    A question inside a fill-in-the-blank ("___ you okay?") or a quotation
    ('"Are we there yet?"', where the clue ends on the closing quote) is a
    literal question, not a trick, and is left alone.
    """
    text = _TRAILING_TAG.sub("", clue.strip())
    return text.endswith("?") and "___" not in text
