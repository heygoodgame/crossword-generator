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
6. ``shared_prefix`` — AGGRESSIVE (high recall, accepts false positives): the
   answer and a clue word share a dominant leading prefix — etymological cousins
   (NAVAL/navy), spelling fragments (TRI/triangle, PRE/prefix), short-answer
   prefixes (ART/artist). Deliberately also flags coincidences (IRAN/iraq); a
   wrongly rejected fair clue is regenerated, which beats shipping the leak.
   Per editor (Jeff) feedback. See the plan, Phase 7.
7. ``compound_containment`` — a clue word (or its stem) is the leading or
   trailing component of the answer: SCOFFLAW clued with "laws", LAWSUIT clued
   with "law". Direction matters: only clue-word-inside-answer is checked here.
8. ``reverse_compound`` — the OTHER direction: a clue word is a closed compound
   that contains the answer as its leading or trailing part, with the remainder
   a real dictionary word (ROBE clued with "bathrobe", BERRY with "strawberry",
   CAT with "housecat", DOG with "bulldog"). Needs a dictionary, so it only runs
   when one is passed to ``detect_leak`` / ``detect_leaks``; otherwise skipped.
   Requiring the remainder to be a real word leaves coincidences with non-word
   remainders (CARDIAC, ELEPHANT, NUCLEAR) alone. Answers down to 3 chars are
   checked: per editor (Jeff) feedback, over-rejecting short answers in genuine
   compounds (and the occasional coincidence like PALACE/ace, which just gets
   regenerated) is preferable to shipping a clue that spells out the answer.
9. ``embedded_substring`` — a curated catch for short answers embedded in a
   longer, etymologically-related clue word the reverse-compound rule cannot
   split because the remainder is not a word (roBOT = "ro" + BOT, unTIL =
   "un" + TIL, gentleMEN = "gentle" + MEN). See ``EMBEDDED_SUBSTRING_LEAKS``.

Scope note: the morphological/stem rules (2, shared_root) require answers of
length >= 4 — for short strings, root matching produces too many coincidental
false positives (CAT/category, EAR/early). Three-letter answers still get
exact-match, abbreviation, shared-prefix, compound-containment, the curated
embedded-substring map, and reverse-compound (gated by a real-word remainder)
checking. Collocation fill-in-the-blank leaks ("___ sauce" => SOY) are a
SEMANTIC class this mechanical detector does not catch; see the plan.

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

from crossword_generator.dictionary import Dictionary
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


# Short answers (typically <= 3 letters, below the morphological floor) that
# embed as a substring of a longer, etymologically-related clue word where the
# remainder is NOT a clean dictionary word — so the reverse-compound rule cannot
# catch them (robot = "ro" + BOT; until = "un" + TIL). High-precision and
# curated, in the same spirit as LEAK_MAP / ABBREV_EXPANSION_WORDS: each entry
# maps a short answer to the longer clue words it must not be clued by. Grow
# from real editor (Jeff) misses on shipped batches. Lowercase.
EMBEDDED_SUBSTRING_LEAKS: dict[str, frozenset[str]] = {
    # Seeded from Jeff's 2026-06 review of the week-20260716 batch.
    "bot": frozenset({"robot", "robots", "robotic", "robotics"}),
    "til": frozenset({"until"}),
    "men": frozenset({"gentlemen", "gentleman"}),
    "man": frozenset({"gentleman", "gentlemen"}),
}


@dataclass(frozen=True)
class LeakFinding:
    """A clue that echoes its own answer."""

    number: int
    direction: str
    answer: str
    clue: str
    # exact | shared_root | irregular | abbrev_expansion | abbrev_expansion_word
    # | shared_prefix | compound_containment | reverse_compound
    # | embedded_substring
    kind: str
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


def detect_leak(
    answer: str, clue: str, dictionary: Dictionary | None = None
) -> LeakFinding | None:
    """Return the first leak found for one answer/clue pair, or None.

    The (number, direction) are filled in by ``detect_leaks``; here we only
    need answer + clue, which makes this directly unit-testable. ``dictionary``
    is optional: when supplied it enables the reverse-compound rule (rule 8),
    which needs a wordlist to confirm a compound split; without it that one
    rule is skipped and every other rule behaves identically.
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

    # 4b. Curated embedded-substring leak: a short answer (below the
    #     morphological floor) embedded in a longer etymologically-related clue
    #     word the reverse-compound rule cannot split (robot/BOT, until/TIL).
    embedding_words = EMBEDDED_SUBSTRING_LEAKS.get(answer_l)
    if embedding_words:
        for w in words:
            if w in embedding_words:
                return _finding(answer, clue, "embedded_substring", w)

    # 5. Abbreviation expansion + initialism.
    finding = _detect_abbrev_leak(answer, answer_l, clue_l, words)
    if finding is not None:
        return finding

    # 6. Shared-prefix / etymology / spelling-fragment leak (aggressive, high
    #    recall). Catches etymological cousins (NAVAL/navy), spelling fragments
    #    (TRI/triangle, PRE/prefix), and short-answer prefixes (ART/artist).
    #    Deliberately accepts some false positives (IRAN/iraq, CARD/cardiac-type
    #    coincidences) — a wrongly rejected fair clue is just regenerated, which
    #    is preferable to shipping these leaks. See the plan, Phase 7.
    leak_word = _shared_prefix_leak(answer_l, words)
    if leak_word is not None:
        return _finding(answer, clue, "shared_prefix", leak_word)

    # 7. Compound containment: a clue word is the leading or trailing
    #    component of the answer (SCOFFLAW / "laws", LAWSUIT / "law").
    leak_word = _compound_containment_leak(answer_l, words)
    if leak_word is not None:
        return _finding(answer, clue, "compound_containment", leak_word)

    # 8. Reverse compound containment: a clue word is a closed compound that
    #    CONTAINS the answer as its leading or trailing component (ROBE clued
    #    with "bathrobe", BERRY with "strawberry"). Needs a dictionary to
    #    confirm the remainder is a real word, so it only runs when one is
    #    supplied. See rule 8 in the module docstring.
    if dictionary is not None:
        leak_word = _reverse_compound_leak(answer_l, words, dictionary)
        if leak_word is not None:
            return _finding(answer, clue, "reverse_compound", leak_word)

    return None


# Short non-content words that embed coincidentally in many answers
# (THRONE/"one", TEACHER/"her") and must not count as compound parts.
_CONTAINMENT_SKIP = frozenset(
    "one ones her hers his him she our ours your yours".split()
)

# Minimum length for the contained clue word and for what remains of the
# answer around it. Both floors keep noise down (PHONE/"one" survives via
# the 2-char remainder even before the skip list applies).
_CONTAINMENT_MIN_PART = 3


def _compound_containment_leak(answer_l: str, words: list[str]) -> str | None:
    """Return a clue word that forms the start or end of the answer.

    Checks the raw clue word and its stem (so "laws" catches SCOFFLAW), and
    the raw answer and its stem (so "law" catches OUTLAWS). Only this
    direction is checked — see rule 7 in the module docstring.
    """
    answer_forms = {answer_l, _stem(answer_l)}
    for w in words:
        if w in _CONTAINMENT_SKIP:
            continue
        for cand in {w, _stem(w)}:
            if len(cand) < _CONTAINMENT_MIN_PART:
                continue
            for ans in answer_forms:
                if len(ans) - len(cand) < _CONTAINMENT_MIN_PART:
                    continue
                if ans.startswith(cand) or ans.endswith(cand):
                    return w
    return None


# Floors for the reverse-compound rule. The OTHER half of the compound must be
# >= 3 chars AND a real dictionary word, so we are splitting a genuine two-part
# compound, not shaving a fragment. The answer floor is 3: per Jeff's standing
# preference, over-rejecting is fine (PALACE/ace, KITTEN/ten get regenerated)
# and clearly better than shipping a clue that spells out the answer inside one
# of its words (SOMEONE/one, WEEKEND/end, WILDCAT/cat, BULLDOG/dog). The
# real-remainder requirement still leaves pure coincidences with non-word
# remainders alone (CARDIAC, ELEPHANT, ROBOT — the last caught instead by the
# curated EMBEDDED_SUBSTRING_LEAKS map).
_REVERSE_COMPOUND_MIN_ANSWER = 3
_REVERSE_COMPOUND_MIN_REMAINDER = 3


def _reverse_compound_leak(
    answer_l: str, words: list[str], dictionary: Dictionary
) -> str | None:
    """Return a clue word that is a closed compound containing the answer.

    Fires when a clue word splits into ``answer + dictionary_word`` or
    ``dictionary_word + answer`` — i.e. the clue uses a more specific compound
    of the answer to define it (ROBE / "bathrobe", BERRY / "strawberry", CAKE /
    "cheesecake"). Such clues hand the answer to the solver spelled out inside
    a single clue word, which Jeff flagged as off-putting even though it is not
    an outright echo.

    Requiring the remainder to be a real dictionary word keeps coincidental
    substrings out (CARDIAC is not CARD + a word; ELEPHANT is not ANT + a word).
    A few true compounds that are not semantic compounds still slip through
    (DISCOVER = DISC + OVER); per the detector's standing policy a wrongly
    rejected fair clue is just regenerated, which beats shipping the leak.
    """
    if len(answer_l) < _REVERSE_COMPOUND_MIN_ANSWER:
        return None
    answer_u = answer_l.upper()
    for w in words:
        cand = w.upper()
        if cand == answer_u or len(cand) <= len(answer_u):
            continue
        remainder: str | None = None
        if cand.startswith(answer_u):
            remainder = cand[len(answer_u):]
        elif cand.endswith(answer_u):
            remainder = cand[: -len(answer_u)]
        if remainder is None or len(remainder) < _REVERSE_COMPOUND_MIN_REMAINDER:
            continue
        if dictionary.contains(remainder):
            return w
    return None


# Minimum shared leading-prefix length for the shared-prefix rule, and the
# fraction of EACH word that prefix must cover to count as the "same family".
_SHARED_PREFIX_MIN = 3
_SHARED_PREFIX_COVERAGE = 0.6


def _shared_prefix_leak(answer_l: str, words: list[str]) -> str | None:
    """Return a clue word that shares a dominant leading prefix with the answer.

    Two cases count:
    - A 3-letter answer that is the literal start of a longer clue word
      (TRI/triangle, ART/artist) — flag-all for max recall on short answers.
    - Any answer and clue word sharing a prefix of >= 3 chars that covers
      >= 60% of BOTH words (NAVAL/navy, KNEE/kneel, READ/ready). The coverage
      requirement rejects coincidences where the prefix is a small part of the
      clue word (CARD/cardiac, STAR/started).
    """
    if len(answer_l) < 3:
        return None
    for w in words:
        if w == answer_l or len(w) < 3:
            continue
        common = 0
        for x, y in zip(answer_l, w):
            if x == y:
                common += 1
            else:
                break
        if len(answer_l) == 3 and len(w) > 3 and w.startswith(answer_l):
            return w
        if (
            common >= _SHARED_PREFIX_MIN
            and common >= len(answer_l) * _SHARED_PREFIX_COVERAGE
            and common >= len(w) * _SHARED_PREFIX_COVERAGE
        ):
            return w
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


def detect_leaks(
    clues: Iterable[ClueEntry], dictionary: Dictionary | None = None
) -> list[LeakFinding]:
    """Scan a set of clues and return every leak found.

    ``dictionary`` is forwarded to ``detect_leak`` to enable the
    reverse-compound rule; without it that rule is skipped (see ``detect_leak``).
    """
    findings: list[LeakFinding] = []
    for entry in clues:
        if not entry.clue:
            continue
        hit = detect_leak(entry.answer, entry.clue, dictionary)
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
