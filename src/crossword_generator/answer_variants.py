"""Inflectional variant expansion for answer exclusion lists.

Cross-day answer exclusions are exact string matches, so a schedule that
blocks ART still happily fills ARTS the next day (274 singular/plural pairs
were found in the first 62 days of the daily schedule). This module expands
an excluded answer into the regular English inflections a solver would
perceive as "the same word again".

The rules mirror the daily scheduler's variant tier exactly (hey-you
``App\\Support\\CrosswordAnswerVariants``, added after the easy Mini's USERS
and the easy Midi's USER shared 2026-10-26), so a fill the generator allows
is one the scheduler will place on its day. Change both together.

- Forms: +S, +ING, +ES after S/X/Z/CH/SH/O, +ED (+D and E-dropped +ING
  after a silent E), consonant+Y -> IES/IED, and a doubled final consonant
  (STRIP -> STRIPPED, STRIPPING).
- Symmetric: TIDYING expands to TIDY just as TIDY expands to TIDYING.
- No agentive/comparative -ER, and no -ES after other letters: on the live
  schedule those mostly paired unrelated words (LINE/LINER, DORM/DORMER,
  STAT/STATES).

It is deliberately dictionary-free: it produces candidate strings (BAKEING,
USERRED), and the dictionary filter simply drops whichever of them exist.
The intra-grid terminal-S / shared-root rules in ``graders/fill_grader.py``
are separate.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable

# Minimum length for a base word and for any variant. Three-letter answers are
# the glue that keeps 9x9 (and 5x5) grids fillable; expanding them (ART ->
# ARTS, ERA -> ERAS, ...) would gut those pools for little reuse benefit.
DEFAULT_MIN_BASE_LENGTH = 4

_ANSWER = re.compile(r"[A-Z]+")
_ES_ENDINGS = ("S", "X", "Z", "CH", "SH", "O")
_CONSONANT_Y = re.compile(r"[^AEIOU]Y$")
_DOUBLING_END = re.compile(r"[^AEIOU][AEIOU]([^AEIOUWXY])$")


def expand_answer_variants(
    answers: Iterable[str],
    *,
    min_base_length: int = DEFAULT_MIN_BASE_LENGTH,
) -> set[str]:
    """Return inflectional variants of ``answers`` (excluding the answers).

    Bases themselves are never returned, and neither are variants shorter
    than ``min_base_length`` — so with the default of 4, ARTS does not
    exclude ART: the 3-letter glue pool is never touched from either
    direction (a 30-day exclusion that leaked into 3-letter words cost
    easy 5x5 fills two timeouts in eight in testing). Input is normalized
    to uppercase and stripped.
    """
    bases = {answer.strip().upper() for answer in answers}
    bases.discard("")
    variants: set[str] = set()
    for base in bases:
        variants.update(answer_variants(base, min_length=min_base_length))
    return variants - bases


@functools.lru_cache(maxsize=65536)
def answer_variants(
    answer: str, *, min_length: int = DEFAULT_MIN_BASE_LENGTH
) -> frozenset[str]:
    """Variants of one normalized (A-Z) answer, never the answer itself.

    The reverse direction checks each candidate base against the forward
    rules, so the relation is symmetric by construction.
    """
    if len(answer) < min_length or not _ANSWER.fullmatch(answer):
        return frozenset()
    found = set(_inflect(answer, min_length))
    for base in _base_candidates(answer):
        if answer in _inflect(base, min_length):
            found.add(base)
    found.discard(answer)
    return frozenset(variant for variant in found if len(variant) >= min_length)


def _inflect(base: str, min_length: int) -> list[str]:
    """Forward inflections of ``base``; over-generates on purpose."""
    if len(base) < min_length:
        return []
    forms = [base + "S", base + "ING"]
    if base.endswith(_ES_ENDINGS):
        forms.append(base + "ES")
    if base.endswith("E"):
        # BAKE -> BAKED, BAKING.
        forms += [base + "D", base[:-1] + "ING"]
    else:
        forms.append(base + "ED")
    if _CONSONANT_Y.search(base):
        # PARTY -> PARTIES, PARTIED.
        forms += [base[:-1] + "IES", base[:-1] + "IED"]
    doubling = _DOUBLING_END.search(base)
    if doubling:
        # STRIP -> STRIPPED, STRIPPING.
        forms += [base + doubling.group(1) + "ED", base + doubling.group(1) + "ING"]
    return forms


def _base_candidates(word: str) -> list[str]:
    """Every string ``_inflect`` could have produced ``word`` from."""
    return [
        word[:-1],
        word[:-2],
        word[:-3],
        word[:-4],
        word[:-3] + "E",  # BAKING -> BAKE
        word[:-3] + "Y",  # PARTIES, PARTIED -> PARTY
    ]
