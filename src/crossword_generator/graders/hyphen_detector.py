"""Deterministic detection of wrongly-hyphenated clue text.

The clue-generation prompt tells the model not to hyphenate open compounds
("snooze button", never "snooze-button"), but a local model intermittently
ignores it, producing clues like "Lingerie-drawer" or "Paper-crane". This
module backstops the prompt with a mechanical check that flags suspect hyphens
into the repair loop, exactly like ``leak_detector`` flags answer echoes.

Policy (per editor feedback): over-flagging is acceptable — a fair clue that
gets flagged is simply rephrased by the repair LLM — but a genuine hyphenated
word (``well-known``, ``X-ray``, ``self-esteem``, ``two-time``) must NOT be
flagged. So a hyphen is suspect only when BOTH:

1. the hyphenated token is NOT a real dictionary word (``well-known`` and
   ``self-esteem`` are in the list, so they pass), AND
2. both halves are themselves real words (so the natural fix is to replace the
   hyphen with a space — "Paper-crane" -> "Paper crane").

Tokens with a leading/trailing hyphen (``"___"`` style blanks, dashes used as
punctuation) and numeric/single-letter halves (``X-ray``, ``9-to-5``) are left
alone. Multi-hyphen tokens are checked pairwise across each internal hyphen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crossword_generator.dictionary import Dictionary
from crossword_generator.models import ClueEntry

# A word-internal hyphen: a hyphen with an alphabetic character on both sides.
# This skips em/en-dashes used as punctuation and leading/trailing hyphens.
_HYPHEN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+")


@dataclass(frozen=True)
class HyphenFinding:
    """A clue containing a wrongly-hyphenated open compound."""

    number: int
    direction: str
    answer: str
    clue: str
    token: str  # the offending hyphenated token, e.g. "Paper-crane"
    suggestion: str  # the de-hyphenated form, e.g. "Paper crane"


def _is_suspect(token: str, dictionary: Dictionary) -> bool:
    """True if ``token`` is an open compound wrongly joined by a hyphen.

    Suspect = the whole hyphenated token is not a real word, but every half is.
    Genuine hyphenated words (``well-known``, ``self-esteem``) are in the
    dictionary as a unit and so are not suspect.
    """
    if dictionary.contains(token):
        return False
    parts = token.split("-")
    if len(parts) < 2:
        return False
    # Every half must be a real, multi-letter word for the space-substitution
    # fix to be the obvious correction. A single-letter or numeric half
    # (``X-ray``, ``e-book``) means a real hyphenation convention, leave it.
    return all(len(p) >= 2 and dictionary.contains(p) for p in parts)


def detect_hyphen(
    answer: str, clue: str, dictionary: Dictionary
) -> HyphenFinding | None:
    """Return the first suspect hyphenated token in ``clue``, or None.

    Answer-only fields (number/direction) are filled by ``detect_hyphens``;
    here we take answer + clue so the rule is directly unit-testable.
    """
    for match in _HYPHEN_TOKEN_RE.finditer(clue):
        token = match.group(0)
        if _is_suspect(token, dictionary):
            return HyphenFinding(
                number=0,
                direction="",
                answer=answer,
                clue=clue,
                token=token,
                suggestion=token.replace("-", " "),
            )
    return None


def detect_hyphens(
    clues: list[ClueEntry], dictionary: Dictionary
) -> list[HyphenFinding]:
    """Return one finding per clue that contains a suspect hyphenated token."""
    findings: list[HyphenFinding] = []
    for clue in clues:
        finding = detect_hyphen(clue.answer, clue.clue, dictionary)
        if finding is not None:
            findings.append(
                HyphenFinding(
                    number=clue.number,
                    direction=clue.direction,
                    answer=clue.answer,
                    clue=clue.clue,
                    token=finding.token,
                    suggestion=finding.suggestion,
                )
            )
    return findings
