"""Abbreviation classification for fill dictionary words.

Jeff (2026-10-27): "Try to limit the number of abbreviations used in any
one puzzle." Every abbreviation answer needs an "(abbr.)" or ", for short"
tag, and a grid full of them reads as compromised fill. The fill grader caps
abbreviation answers per grid by size (5x5: 1, 7x7: 2, 9x9: 3). Like the
proper-noun cap, the rule runs hundreds of times per puzzle, so the LLM
classification happens offline: ``classify-abbreviations`` labels every
short dictionary word once and commits the result; the grader just counts
set membership.

Classification file format: ``WORD;A`` or ``WORD;W`` per line, uppercase,
sorted. ``A`` = every fair clue would have to flag the answer as shortened
(ETA, NBA, OTS, IDS, SIS). ``W`` = the word can be clued as an ordinary
word, even if it also has an abbreviation reading (AMP, MEMO, INFO, PRO).
A word can be both a proper noun and an abbreviation (NBA); it then counts
toward both caps.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from crossword_generator.llm.base import LLMProvider
from crossword_generator.proper_nouns import (
    classify_words,
    load_classifications,
    save_classifications,
)

ABBREVIATION = "A"
WORD = "W"
LABELS = (ABBREVIATION, WORD)

# Longer entries are almost never abbreviations, so the classifier and the
# grid rule only look at answers up to this length.
MAX_CLASSIFIED_LENGTH = 6

_SYSTEM_PROMPT = (
    "You classify crossword answers as abbreviations or ordinary words.\n"
    "\n"
    "Judge each answer by how a fair crossword clue must treat it.\n"
    "\n"
    "Label a word A when EVERY fair clue for it would have to signal that "
    'the answer is shortened, with a tag like "(abbr.)" or ", for short" or '
    'an abbreviated cue word ("org.", "Dept."). This covers acronyms and '
    "initialisms (ETA, ATM, CEO, NBA, NASA, RSVP), abbreviations of words "
    "or units (ETC, ESQ, MPH, LBS, ASST, DEPT), plurals of these (OTS, IDS, "
    "GIS, CDS, MOS, CEOS), Latin short forms (ETAL, IBID), and clipped forms "
    'that are still clued with "for short" (SIS, LIB as liberal).\n'
    "\n"
    "Label a word W when at least one fair clue can treat it as an ordinary "
    "word with no shortening signal: standard nouns, verbs, adjectives, "
    "interjections, names, or phrases. Clipped forms that have become "
    "everyday words are W (MEMO, INFO, AD, PRO, GYM, LAB, VET, DOC, MATH, "
    "FLU, EXAM). Acronyms that became words are W (LASER, RADAR, SCUBA). "
    "Words that also happen to be an abbreviation are W when the ordinary "
    "reading works (AMP, MAX, ORE, CAL, SAT, MAR). When in doubt, choose W.\n"
    "\n"
    "Entries may be multi-word phrases run together; judge the whole "
    "phrase.\n"
    "\n"
    "Respond with one line per input word, in order, formatted exactly as\n"
    "WORD A\n"
    "or\n"
    "WORD W\n"
    "with no other text, numbering, or commentary."
)


def classify_abbreviations(
    provider: LLMProvider,
    words: list[str],
    *,
    batch_size: int = 100,
    max_workers: int = 4,
    model: str | None = None,
    checkpoint: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    """Classify words as ``A``/``W`` in parallel batches."""
    return classify_words(
        provider,
        words,
        batch_size=batch_size,
        max_workers=max_workers,
        model=model,
        checkpoint=checkpoint,
        system_prompt=_SYSTEM_PROMPT,
        allowed_labels=LABELS,
    )


def load_abbreviation_classifications(path: Path) -> dict[str, str]:
    """Load an existing classification file (``WORD;A|W`` per line)."""
    return load_classifications(path, allowed_labels=LABELS)


def save_abbreviation_classifications(
    path: Path, labels: dict[str, str]
) -> None:
    """Write the classification file sorted for stable diffs."""
    save_classifications(path, labels)


def load_abbreviation_set(path: Path) -> frozenset[str]:
    """Load only the abbreviation words for the fill grader."""
    if not path.exists():
        raise FileNotFoundError(
            f"Abbreviation classification file not found: {path}"
        )
    labels = load_abbreviation_classifications(path)
    return frozenset(
        word for word, label in labels.items() if label == ABBREVIATION
    )


def abbreviation_cap_for_size(
    caps_by_size: dict[int, int], grid_size: int
) -> int | None:
    """Per-grid abbreviation cap for a grid size, or None when uncapped.

    Sizes missing from the map take the cap of the largest configured size
    below them (an 11x11 midi inherits the 9x9 cap); a size smaller than
    every configured size is uncapped.
    """
    if grid_size in caps_by_size:
        return caps_by_size[grid_size]
    smaller = [size for size in caps_by_size if size < grid_size]
    if not smaller:
        return None
    return caps_by_size[max(smaller)]
