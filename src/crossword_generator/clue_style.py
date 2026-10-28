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


def normalize_abbreviation_tags(clue: str) -> str:
    """Swap "briefly" / "in brief" abbreviation tags for the house tags."""
    clue = _PAREN_BRIEF_TAG.sub("(abbr.)", clue)
    return _TRAILING_BRIEF_TAG.sub(", for short", clue)
