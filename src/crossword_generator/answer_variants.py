"""Inflectional variant expansion for answer exclusion lists.

Cross-day answer exclusions are exact string matches, so a schedule that
blocks ART still happily fills ARTS the next day (274 singular/plural pairs
were found in the first 62 days of the daily schedule). This module expands
an excluded answer into the regular English inflections a solver would
perceive as "the same word again".

The suffix set mirrors the terminal-S / shared-root rules in
``graders/fill_grader.py`` (which catch the same variants *within* one grid)
but is deliberately generative and dictionary-free: it produces candidate
strings, and the dictionary filter simply drops whichever of them exist.
"""

from __future__ import annotations

from collections.abc import Iterable

# Regular inflectional suffixes appended to a base word.
_APPEND_SUFFIXES = ("S", "ES", "ED", "ER", "ING")

# Minimum base-word length eligible for expansion. Three-letter answers are
# the glue that keeps 9x9 (and 5x5) grids fillable; expanding them (ART ->
# ARTS, ERA -> ERAS, ...) would gut those pools for little reuse benefit.
DEFAULT_MIN_BASE_LENGTH = 4


def expand_answer_variants(
    answers: Iterable[str],
    *,
    min_base_length: int = DEFAULT_MIN_BASE_LENGTH,
) -> set[str]:
    """Return inflectional variants of ``answers`` (excluding the answers).

    For each base word of length >= ``min_base_length`` the result contains:

    * base + S / ES / ED / ER / ING (with the usual drop-trailing-E rule for
      ED / ER / ING: BAKE -> BAKED, BAKER, BAKING)
    * consonant+Y plurals and the reverse: PARTY -> PARTIES, PARTIES -> PARTY
      (plus PARTIED, PARTIER for consonant+Y bases)
    * strip-terminal-S / -ES: ARTS -> ART, BOXES -> BOX

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
        if len(base) < min_base_length or not base.isalpha():
            continue
        variants.update(_variants_of(base))
    variants -= bases
    return {variant for variant in variants if len(variant) >= min_base_length}


def _variants_of(base: str) -> set[str]:
    out: set[str] = set()

    # Appended suffixes.
    for suffix in _APPEND_SUFFIXES:
        out.add(base + suffix)
    if base.endswith("E"):
        stem = base[:-1]
        out.update({stem + "ED", stem + "ER", stem + "ING"})

    # Consonant + Y: PARTY -> PARTIES / PARTIED / PARTIER.
    if len(base) >= 2 and base.endswith("Y") and base[-2] not in "AEIOU":
        stem = base[:-1]
        out.update({stem + "IES", stem + "IED", stem + "IER"})

    # Reverse direction: PARTIES -> PARTY.
    if base.endswith("IES"):
        out.add(base[:-3] + "Y")

    # Strip terminal S / ES.
    if base.endswith("S") and not base.endswith("SS"):
        out.add(base[:-1])
        if base.endswith("ES"):
            out.add(base[:-2])

    return out
