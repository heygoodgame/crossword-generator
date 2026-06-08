"""Deterministic clue-leak detection.

Catches clues that echo their own answer — the single most visible clue
defect — without an LLM call. Rules, in order of precedence:

1. ``exact``       — the answer appears as a whole word in the clue.
2. ``shared_root`` — a clue word and the answer reduce to the same root, or are
   the same root +/- one known affix at a morpheme boundary
   (teaches/teacher, trims/trim, careful/care, baker/baking). Coincidental
   substrings (card/cardiac, read/ready) are rejected.
3. ``irregular``   — an irregular form or known related word from a curated
   map (wife/wives, child/children).
4. ``abbrev_expansion`` — for abbreviation answers, the clue contains the
   spelled-out expansion, or the clue's leading letters spell the answer.
5. ``abbrev_expansion_word`` — for abbreviation answers, the clue contains ANY
   individual word the letters stand for, even a generic one (full strictness):
   ETA / "Arrival time" leaks both "arrival" (A) and "time" (T); MCL clued with
   "ligament" leaks the L. Curated ``ABBREV_EXPANSION_WORDS`` map, grows over
   time.

Scope note: morphological rules require answers of length >= 4. Three-letter
answers get only exact-match and abbreviation checking — for short strings,
substring/root matching produces too many coincidental false positives
(CAT/category, EAR/early). Collocation fill-in-the-blank leaks ("___ sauce" =>
SOY) are a SEMANTIC class this mechanical detector does not catch; see the plan.

The curated map (``LEAK_MAP``) and abbreviation map (``ABBREV_EXPANSIONS``)
start small and high-precision. Grow them from real misses observed in
batches — see docs/plans/clue-quality-and-cost-plan.md.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import snowballstemmer

from crossword_generator.models import ClueEntry

logger = logging.getLogger(__name__)

_stemmer = snowballstemmer.stemmer("english")

# Short function words that must never count as a leak even if they happen to
# match an answer (e.g. answer "AN", "IT"), and that are skipped as clue words.
_STOPWORDS = frozenset(
    """
    a an the of to in on at for and or is are was were be by with as it its
    this that these those from into onto off out up down over under
    """.split()
)

# Minimum answer length before substring / stem rules apply. Below this,
# only the exact-word rule fires (a 3-letter answer appearing verbatim).
_MIN_STEM_LEN = 4

# Irregular forms and tightly-related words the stemmer will not unify.
# Bidirectional: list every form that should leak against any other.
# Keep high-precision — only add pairs that are genuinely the "same word".
LEAK_MAP: dict[str, frozenset[str]] = {
    "wife": frozenset({"wives", "wifely"}),
    "wives": frozenset({"wife", "wifely"}),
    "child": frozenset({"children", "childs"}),
    "children": frozenset({"child"}),
    "man": frozenset({"men"}),
    "men": frozenset({"man"}),
    "woman": frozenset({"women"}),
    "women": frozenset({"woman"}),
    "foot": frozenset({"feet"}),
    "feet": frozenset({"foot"}),
    "tooth": frozenset({"teeth"}),
    "teeth": frozenset({"tooth"}),
    "goose": frozenset({"geese"}),
    "geese": frozenset({"goose"}),
    "mouse": frozenset({"mice"}),
    "mice": frozenset({"mouse"}),
    "person": frozenset({"people"}),
    "people": frozenset({"person"}),
}

# Abbreviation answers and the words that spell them out. If any expansion
# token-sequence appears in the clue, that's a leak (e.g. cluing EST with
# "Eastern Standard Time"). Lowercase, space-separated.
ABBREV_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "est": ("eastern standard time",),
    "edt": ("eastern daylight time",),
    "pst": ("pacific standard time",),
    "cst": ("central standard time",),
    "ceo": ("chief executive officer",),
    "cfo": ("chief financial officer",),
    "coo": ("chief operating officer",),
    "vp": ("vice president",),
    "vps": ("vice presidents",),
    "ops": ("operations", "operation"),
    "pol": ("politician", "political", "politics"),
    "rep": ("representative", "republican"),
    "dem": ("democrat", "democratic"),
    "gov": ("governor", "government"),
    "sen": ("senator", "senate"),
    "univ": ("university",),
    "dept": ("department",),
    "corp": ("corporation",),
    "inc": ("incorporated",),
    "assn": ("association",),
    "natl": ("national",),
    "intl": ("international",),
    "dec": ("december",),
    "feb": ("february",),
    "nasa": ("national aeronautics and space administration",),
    "epa": ("environmental protection agency",),
    "fbi": ("federal bureau of investigation",),
    "cia": ("central intelligence agency",),
}

# Abbreviation answers and the INDIVIDUAL words their letters stand for. Full
# strictness: a clue must not contain ANY expansion word (or its stem), even a
# generic one like "time" — the letter literally is that word. This catches
# partial give-aways the full-phrase ABBREV_EXPANSIONS misses, e.g. ETA clued
# "Arrival time abbreviation" (arrival = A, time = T) or MCL clued with
# "ligament" (L). Seeded from real editor finds; grows over time. Lowercase.
ABBREV_EXPANSION_WORDS: dict[str, frozenset[str]] = {
    # Time / scheduling
    "eta": frozenset({"estimated", "time", "arrival"}),
    "etd": frozenset({"estimated", "time", "departure"}),
    # Medical
    "mcl": frozenset({"medial", "collateral", "ligament"}),
    "acl": frozenset({"anterior", "cruciate", "ligament"}),
    "cpr": frozenset({"cardio", "pulmonary", "resuscitation"}),
    "icu": frozenset({"intensive", "care", "unit"}),
    "mri": frozenset({"magnetic", "resonance", "imaging"}),
    "dna": frozenset({"deoxyribonucleic", "acid"}),
    "rna": frozenset({"ribonucleic", "acid"}),
    "er": frozenset({"emergency", "room"}),
    # Common everyday initialisms
    "asap": frozenset({"soon", "possible"}),
    "fyi": frozenset({"information"}),
    "diy": frozenset({"yourself"}),
    "aka": frozenset({"known"}),
    "rsvp": frozenset({"respond", "please"}),
    "tba": frozenset({"announced"}),
    "tbd": frozenset({"determined", "decided"}),
    "faq": frozenset({"frequently", "asked", "questions", "question"}),
    "atm": frozenset({"automated", "automatic", "teller", "machine"}),
    "gps": frozenset({"global", "positioning", "system"}),
    "url": frozenset({"uniform", "resource", "locator"}),
    "pdf": frozenset({"portable", "document", "format"}),
    "ram": frozenset({"random", "access", "memory"}),
    # Finance / org
    "apr": frozenset({"annual", "percentage", "rate"}),
    "gpa": frozenset({"grade", "point", "average"}),
    "ira": frozenset({"individual", "retirement", "account"}),
    "ceo": frozenset({"chief", "executive", "officer"}),
    "cfo": frozenset({"chief", "financial", "officer"}),
    "coo": frozenset({"chief", "operating", "officer"}),
    # Personal data
    "dob": frozenset({"date", "birth"}),
    "ssn": frozenset({"social", "security", "number"}),
    # Units
    "mph": frozenset({"miles", "hour"}),
    "rpm": frozenset({"revolutions", "minute"}),
}


@dataclass(frozen=True)
class LeakFinding:
    """A clue that echoes its own answer."""

    number: int
    direction: str
    answer: str
    clue: str
    kind: str  # exact | substring | shared_root | irregular | abbrev_expansion
    detail: str  # the offending clue word or expansion, for the repair prompt


def _clue_words(clue: str) -> list[str]:
    """Lowercase alphabetic tokens in a clue, stopwords removed."""
    return [w for w in re.findall(r"[a-z]+", clue.lower()) if w not in _STOPWORDS]


def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


# Agent-noun suffixes Snowball leaves intact (teacher -> teacher) while it
# strips them from verb forms (teaches -> teach). Stripping these before a
# second stem pass unifies teacher/teaches, baker/baking, runner/running.
_AGENT_SUFFIXES = ("ers", "ors", "er", "or", "ar")


def _roots(word: str) -> set[str]:
    """Return candidate roots for a word: its stem, plus agent-noun-stripped
    variants so agent/verb pairs (teacher/teaches, baker/baking) unify.

    Stripping an agent suffix can leave a silent-``e`` truncated stem
    (baker -> bak) that the stemmer won't reunite with the verb stem
    (baking -> bake), so we add back a trailing ``e`` candidate to bridge them.
    """
    roots = {_stem(word)}
    for suf in _AGENT_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            roots.add(_stem(base))
            roots.add(_stem(base + "e"))
            # Collapse a doubled final consonant (runn -> run, stopp -> stop)
            # so agent nouns match their -ing verb stems.
            if len(base) >= 2 and base[-1] == base[-2] and base[-1].isalpha():
                roots.add(_stem(base[:-1]))
            break
    return roots


# Multi-character derivational/inflectional suffixes. A real morphological leak
# attaches one at a morpheme boundary. Short, ambiguous endings ("s", "y", "ed")
# are deliberately excluded here — they generate coincidental matches
# (read/ready, art/arty) and are instead handled by the stem comparison, which
# is grounded in real morphology.
_DERIV_SUFFIXES = (
    "ing",
    "ers",
    "ion",
    "ful",
    "less",
    "ness",
    "ment",
    "able",
    "ible",
    "ist",
    "ism",
    "er",
    "or",
)
_DERIV_PREFIXES = ("un", "re", "dis", "non", "over", "under", "house")


def _is_morphological_pair(answer: str, word: str) -> bool:
    """True if answer and word are the same root +/- one known affix at a
    morpheme boundary. Catches care/careful, baker/baking, write/writer while
    rejecting coincidental substrings (card/cardiac, read/ready)."""
    if answer == word:
        return True
    longer, shorter = (word, answer) if len(word) > len(answer) else (answer, word)
    if len(shorter) < _MIN_STEM_LEN:
        return False
    # shorter == longer with one suffix removed (care+ful, edit+or).
    for suf in _DERIV_SUFFIXES:
        if longer.endswith(suf) and longer[: -len(suf)] == shorter:
            return True
    # shorter == longer with one prefix removed (house+wife).
    for pre in _DERIV_PREFIXES:
        if longer.startswith(pre) and longer[len(pre) :] == shorter:
            return True
    # Both reduce to the same stem after stripping an agent-noun suffix —
    # unifies double-derivations the stemmer splits (baker/baking, writer/writes,
    # runner/running). Require the shared stem to be >= 4 chars to avoid noise.
    common = _roots(answer) & _roots(word)
    return any(len(r) >= _MIN_STEM_LEN for r in common)


def detect_leak(answer: str, clue: str) -> LeakFinding | None:
    """Return the first leak found for one answer/clue pair, or None.

    The (number, direction) are filled in by ``detect_leaks``; here we only
    need answer + clue, which makes this directly unit-testable.
    """
    answer_l = answer.lower().strip()
    clue_l = clue.lower()
    if not answer_l or answer_l in _STOPWORDS:
        return None

    # 1. Exact whole-word match.
    if re.search(rf"\b{re.escape(answer_l)}\b", clue_l):
        return _finding(answer, clue, "exact", answer_l)

    words = _clue_words(clue)

    # 2. Morphological relation: the answer and a clue word are the same root
    #    +/- one known affix (care/careful, child/children, trims/trim,
    #    teacher/teaches). Rejects coincidental substrings (card/cardiac).
    if len(answer_l) >= _MIN_STEM_LEN:
        answer_roots = _roots(answer_l)
        for w in words:
            if len(w) < _MIN_STEM_LEN:
                continue
            if _roots(w) & answer_roots or _is_morphological_pair(answer_l, w):
                return _finding(answer, clue, "shared_root", w)

    # 4. Irregular / curated related forms.
    related = LEAK_MAP.get(answer_l)
    if related:
        for w in words:
            if w in related:
                return _finding(answer, clue, "irregular", w)

    # 5. Abbreviation expansion + initialism.
    finding = _detect_abbrev_leak(answer, answer_l, clue_l, words)
    if finding is not None:
        return finding

    return None


def _detect_abbrev_leak(
    answer: str, answer_l: str, clue_l: str, words: list[str]
) -> LeakFinding | None:
    expansions = ABBREV_EXPANSIONS.get(answer_l)
    if expansions:
        for phrase in expansions:
            if re.search(rf"\b{re.escape(phrase)}\b", clue_l):
                return _finding(answer, answer, "abbrev_expansion", phrase)

    # Expansion-word leak: for an abbreviation answer, the clue must not contain
    # ANY individual word its letters stand for, even a generic one (full
    # strictness). E.g. ETA clued "Arrival time abbreviation" leaks both
    # "arrival" (A) and "time" (T); MCL clued with "ligament" leaks the L.
    expansion_words = ABBREV_EXPANSION_WORDS.get(answer_l)
    if expansion_words:
        expansion_stems = {_stem(w) for w in expansion_words}
        for w in words:
            if w in expansion_words or _stem(w) in expansion_stems:
                return _finding(answer, answer, "abbrev_expansion_word", w)

    # Initialism: consecutive clue words whose leading letters spell the
    # answer (e.g. answer "UCLA" clued with "University of California, L.A.").
    # Only meaningful for short all-letter answers.
    if 2 <= len(answer_l) <= 6 and answer_l.isalpha():
        leading = [w[0] for w in words]
        target = list(answer_l)
        n = len(target)
        for i in range(len(leading) - n + 1):
            if leading[i : i + n] == target:
                detail = " ".join(words[i : i + n])
                return _finding(answer, answer, "abbrev_expansion", detail)

    return None


def _finding(answer: str, clue: str, kind: str, detail: str) -> LeakFinding:
    # number/direction are placeholders; detect_leaks rebuilds with real values.
    return LeakFinding(
        number=0,
        direction="",
        answer=answer,
        clue=clue,
        kind=kind,
        detail=detail,
    )


def detect_leaks(clues: Iterable[ClueEntry]) -> list[LeakFinding]:
    """Scan a set of clues and return every leak found."""
    findings: list[LeakFinding] = []
    for entry in clues:
        if not entry.clue:
            continue
        hit = detect_leak(entry.answer, entry.clue)
        if hit is None:
            continue
        findings.append(
            LeakFinding(
                number=entry.number,
                direction=entry.direction,
                answer=entry.answer,
                clue=entry.clue,
                kind=hit.kind,
                detail=hit.detail,
            )
        )
        logger.info(
            "Leak detected [%s] %s-%s %s :: %s (offending: %s)",
            hit.kind,
            entry.number,
            entry.direction,
            entry.answer,
            entry.clue,
            hit.detail,
        )
    return findings
