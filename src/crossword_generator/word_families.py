"""Same-root word families for the within-grid duplicate rule.

Jeff rejects grids that use two words from one family: inflections
(OWE/OWED, HIRED/HIRING, PLAY/PLAYED), irregular forms (ATE/EAT,
TEETH/TOOTH) and derivatives (STINKY/STINKS, COMEDIC/COMEDIANS). The
2026-09 midi batch shipped four such grids and his hand edits to remove
them pushed two puzzles off their target days, so the rule has to catch
them before clues are written.

``family_keys`` maps an answer to itself plus every lemma it can be derived
from; two answers in one grid that share a key are a same-root pair. Every
lemma must be a word in the supplied lexicon, which keeps coincidental
endings out (START is not STAR + a suffix). Three-letter lemmas are the
glue that keeps grids fillable and where coincidences are cheapest, so they
get stricter morphology: only inflectional suffixes, with the
consonant-doubling rule enforced (BARRED is BAR + ED; BARED is BARE + D).
Their real -Y adjectives (ICE/ICY, FOG/FOGGY) come from a curated table
instead, since the rule would also pair BUS/BUSY and MANE/MANY.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache
from itertools import chain
from pathlib import Path

IsWord = Callable[[str], bool]

# Minimum lemma length for derivational suffixes and the looser
# inflectional rules. Shorter lemmas only take _SHORT_LEMMA_SUFFIXES.
_MIN_LEMMA_LENGTH = 4

# Shortest lemma considered at all (OWE, EAT, RID).
_MIN_SHORT_LEMMA_LENGTH = 3

# Suffixes tried on lemmas of _MIN_LEMMA_LENGTH+ letters, longest first so a
# word's most specific analysis is listed first in debugging output.
_SUFFIXES = (
    "NESS",
    "LESS",
    "ABLE",
    "IANS",
    "ERS",
    "EST",
    "FUL",
    "IAN",
    "ING",
    "ES",
    "ED",
    "ER",
    "IC",
    "LY",
    "S",
    "D",
    "Y",
)

# Inflections that attach to a monosyllable (OWE -> OWED, RID -> RIDDING,
# EAT -> EATER, BOX -> BOXES). -Y, -LY and doubled -ER are left out: BUS/BUSY,
# EAR/EARLY and BET/BETTER are coincidences, not one word twice.
_SHORT_LEMMA_SUFFIXES = ("ING", "EST", "ERS", "ES", "ED", "ER", "S", "D")

# Suffixes that begin with a vowel: the lemma may have dropped a final E
# (HIRE -> HIRING) or doubled its last consonant (STOP -> STOPPED).
_VOWEL_SUFFIXES = frozenset(
    {"ABLE", "IANS", "ERS", "EST", "IAN", "ING", "ES", "ED", "ER", "IC", "Y"}
)

# Suffixes after which a lemma's final Y shows up as I (PARTY -> PARTIES,
# HAPPY -> HAPPILY, COMEDY -> COMEDIC).
_Y_TO_I_SUFFIXES = frozenset(
    {"NESS", "LESS", "IANS", "ERS", "EST", "FUL", "IAN", "ES", "ED", "ER", "LY"}
)

# Suffixes after which the lemma's final Y is dropped (COMEDY -> COMEDIC).
_Y_DROP_SUFFIXES = frozenset({"IC", "IAN", "IANS"})

_VOWELS = frozenset("AEIOU")

# Irregular inflections, lemma -> forms. Only forms of 3+ letters matter
# (2-letter answers never share a grid with these in practice).
_IRREGULAR_FAMILIES: dict[str, tuple[str, ...]] = {
    "ARISE": ("AROSE", "ARISEN"),
    "AWAKE": ("AWOKE", "AWOKEN"),
    "BE": ("ARE", "WAS", "WERE", "BEEN", "BEING"),
    "BEAR": ("BORE", "BORNE", "BORN"),
    "BEAT": ("BEATEN",),
    "BECOME": ("BECAME",),
    "BEGIN": ("BEGAN", "BEGUN"),
    "BEND": ("BENT",),
    "BITE": ("BIT", "BITTEN"),
    "BLEED": ("BLED",),
    "BLOW": ("BLEW", "BLOWN"),
    "BREAK": ("BROKE", "BROKEN"),
    "BREED": ("BRED",),
    "BRING": ("BROUGHT",),
    "BUILD": ("BUILT",),
    "BUY": ("BOUGHT",),
    "CATCH": ("CAUGHT",),
    "CHILD": ("CHILDREN",),
    "CHOOSE": ("CHOSE", "CHOSEN"),
    "CLING": ("CLUNG",),
    "COME": ("CAME",),
    "DEAL": ("DEALT",),
    "DIG": ("DUG",),
    "DIVE": ("DOVE",),
    "DO": ("DOES", "DID", "DONE", "DOING"),
    "DRAW": ("DREW", "DRAWN"),
    "DREAM": ("DREAMT",),
    "DRINK": ("DRANK", "DRUNK"),
    "DRIVE": ("DROVE", "DRIVEN"),
    "EAT": ("ATE", "EATEN"),
    "FALL": ("FELL", "FALLEN"),
    "FEED": ("FED",),
    "FEEL": ("FELT",),
    "FIGHT": ("FOUGHT",),
    "FIND": ("FOUND",),
    "FLEE": ("FLED",),
    "FLING": ("FLUNG",),
    "FLY": ("FLEW", "FLOWN", "FLIES"),
    "FOOT": ("FEET",),
    "FORGET": ("FORGOT", "FORGOTTEN"),
    "FORGIVE": ("FORGAVE", "FORGIVEN"),
    "FREEZE": ("FROZE", "FROZEN"),
    "GET": ("GOT", "GOTTEN"),
    "GIVE": ("GAVE", "GIVEN"),
    "GO": ("GOES", "WENT", "GONE", "GOING"),
    "GOOSE": ("GEESE",),
    "GRIND": ("GROUND",),
    "GROW": ("GREW", "GROWN"),
    "HANG": ("HUNG",),
    "HAVE": ("HAS", "HAD", "HAVING"),
    "HEAR": ("HEARD",),
    "HIDE": ("HID", "HIDDEN"),
    "HOLD": ("HELD",),
    "KEEP": ("KEPT",),
    "KNEEL": ("KNELT",),
    "KNOW": ("KNEW", "KNOWN"),
    "LAY": ("LAID",),
    "LEAD": ("LED",),
    "LEAP": ("LEAPT",),
    "LEAVE": ("LEFT",),
    "LEND": ("LENT",),
    "LIE": ("LAY", "LAIN", "LYING"),
    "LIGHT": ("LIT",),
    "LOSE": ("LOST",),
    "MAKE": ("MADE",),
    "MAN": ("MEN",),
    "MEAN": ("MEANT",),
    "MEET": ("MET",),
    "MOUSE": ("MICE",),
    "OX": ("OXEN",),
    "PAY": ("PAID",),
    "RIDE": ("RODE", "RIDDEN"),
    "RING": ("RANG", "RUNG"),
    "RISE": ("ROSE", "RISEN"),
    "RUN": ("RAN",),
    "SAY": ("SAID",),
    "SEE": ("SAW", "SEEN"),
    "SEEK": ("SOUGHT",),
    "SELL": ("SOLD",),
    "SEND": ("SENT",),
    "SHAKE": ("SHOOK", "SHAKEN"),
    "SHINE": ("SHONE",),
    "SHOOT": ("SHOT",),
    "SHRINK": ("SHRANK", "SHRUNK"),
    "SING": ("SANG", "SUNG"),
    "SINK": ("SANK", "SUNK"),
    "SIT": ("SAT",),
    "SLAY": ("SLEW", "SLAIN"),
    "SLEEP": ("SLEPT",),
    "SLIDE": ("SLID",),
    "SLING": ("SLUNG",),
    "SPEAK": ("SPOKE", "SPOKEN"),
    "SPEED": ("SPED",),
    "SPEND": ("SPENT",),
    "SPIN": ("SPUN",),
    "SPRING": ("SPRANG", "SPRUNG"),
    "STAND": ("STOOD",),
    "STEAL": ("STOLE", "STOLEN"),
    "STICK": ("STUCK",),
    "STING": ("STUNG",),
    "STINK": ("STANK", "STUNK"),
    "STRIDE": ("STRODE",),
    "STRIKE": ("STRUCK",),
    "STRING": ("STRUNG",),
    "STRIVE": ("STROVE", "STRIVEN"),
    "SWEAR": ("SWORE", "SWORN"),
    "SWEEP": ("SWEPT",),
    "SWIM": ("SWAM", "SWUM"),
    "SWING": ("SWUNG",),
    "TAKE": ("TOOK", "TAKEN"),
    "TEACH": ("TAUGHT",),
    "TEAR": ("TORE", "TORN"),
    "TELL": ("TOLD",),
    "THINK": ("THOUGHT",),
    "THROW": ("THREW", "THROWN"),
    "TOOTH": ("TEETH",),
    "WAKE": ("WOKE", "WOKEN"),
    "WEAR": ("WORE", "WORN"),
    "WEAVE": ("WOVE", "WOVEN"),
    "WEEP": ("WEPT",),
    "WIN": ("WON",),
    "WIND": ("WOUND",),
    "WOMAN": ("WOMEN",),
    "WRING": ("WRUNG",),
    "WRITE": ("WROTE", "WRITTEN"),
}

# -Y adjectives of lemmas too short for the -Y suffix rule (fewer than
# _MIN_LEMMA_LENGTH letters before the Y), where the rule would mostly pair
# coincidences. Curated so only real pairs count: the 2026-11-07 easy midi
# draft had ICE and ICY, and Jeff's edit removing ICE pushed it off its day.
_SHORT_Y_ADJECTIVES: dict[str, tuple[str, ...]] = {
    "AIR": ("AIRY",),
    "ART": ("ARTY",),
    "ASH": ("ASHY",),
    "BAG": ("BAGGY",),
    "BONE": ("BONY",),
    "DEW": ("DEWY",),
    "FAT": ("FATTY",),
    "FOG": ("FOGGY",),
    "FUR": ("FURRY",),
    "GORE": ("GORY",),
    "GUM": ("GUMMY",),
    "HAZE": ("HAZY",),
    "ICE": ("ICY",),
    "INK": ("INKY",),
    "LACE": ("LACY",),
    "MUD": ("MUDDY",),
    "NOSE": ("NOSY",),
    "NUT": ("NUTTY",),
    "OIL": ("OILY",),
    "OOZE": ("OOZY",),
    "ROSE": ("ROSY",),
    "SUN": ("SUNNY",),
    "WIRE": ("WIRY",),
}

_IRREGULAR_LEMMAS: dict[str, frozenset[str]] = {}
for _lemma, _forms in chain(_IRREGULAR_FAMILIES.items(), _SHORT_Y_ADJECTIVES.items()):
    for _form in _forms:
        _IRREGULAR_LEMMAS[_form] = _IRREGULAR_LEMMAS.get(_form, frozenset()) | {_lemma}


def family_keys(word: str, is_word: IsWord) -> frozenset[str]:
    """Return ``word`` plus every lemma it derives from.

    Lemmas come from irregular forms, then up to two suffix-stripping steps
    (COMEDIANS -> COMEDIAN -> COMEDY), each lemma checked against
    ``is_word``. Irregular lemmas need no lexicon check.
    """
    normalized = word.strip().upper()
    keys = {normalized}
    frontier = {normalized}
    for _ in range(2):
        found: set[str] = set()
        for form in frontier:
            found |= _IRREGULAR_LEMMAS.get(form, frozenset())
            found |= _suffix_lemmas(form, is_word)
        found -= keys
        if not found:
            break
        keys |= found
        frontier = found
    return frozenset(keys)


def same_root_pairs(words: Iterable[str], is_word: IsWord) -> list[tuple[str, str]]:
    """Return sorted pairs of distinct answers that share a family key.

    Pairs that differ only by a terminal S (ART/ARTS) are left to the
    grader's terminal-S rule so a grid is not reported twice for one pair.
    """
    unique = sorted({word.strip().upper() for word in words if word.strip()})
    keys = {word: family_keys(word, is_word) for word in unique}
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(unique):
        for right in unique[index + 1 :]:
            if right == left + "S" or left == right + "S":
                continue
            if keys[left] & keys[right]:
                pairs.append((left, right))
    return pairs


def _suffix_lemmas(word: str, is_word: IsWord) -> set[str]:
    lemmas: set[str] = set()
    for suffix in _SUFFIXES:
        if not word.endswith(suffix) or len(word) <= len(suffix):
            continue
        base = word[: -len(suffix)]
        for lemma in _lemma_candidates(base, suffix):
            if lemma == word or len(lemma) < _MIN_SHORT_LEMMA_LENGTH:
                continue
            if not _attaches(lemma, base, suffix):
                continue
            if is_word(lemma):
                lemmas.add(lemma)
    return lemmas


def _lemma_candidates(base: str, suffix: str) -> set[str]:
    candidates = {base}
    if suffix in _VOWEL_SUFFIXES:
        candidates.add(base + "E")
        if _is_doubled_consonant(base):
            candidates.add(base[:-1])
    if suffix in _Y_TO_I_SUFFIXES and base.endswith("I"):
        candidates.add(base[:-1] + "Y")
    if suffix in _Y_DROP_SUFFIXES:
        candidates.add(base + "Y")
    return candidates


def _attaches(lemma: str, base: str, suffix: str) -> bool:
    """Whether ``suffix`` really attaches to ``lemma`` to spell base+suffix."""
    if suffix == "D" and not lemma.endswith("E"):
        # BAKE + D, never BRAN + D.
        return False
    if suffix == "ES" and lemma == base:
        # BOXES, BUSES, HEROES, WISHES: -ES only after a sibilant or O.
        if not lemma.endswith(("S", "X", "Z", "CH", "SH", "O")):
            return False
    if suffix == "S" and lemma.endswith("S"):
        return False
    if suffix == "Y" and len(base) < _MIN_LEMMA_LENGTH:
        # STINKY, STONY -> STONE; but MANY is not MANE + Y, nor TINY TINE + Y.
        return False
    if len(lemma) >= _MIN_LEMMA_LENGTH:
        return True
    if suffix not in _SHORT_LEMMA_SUFFIXES:
        return False
    if suffix in _VOWEL_SUFFIXES and lemma == base and _is_cvc(lemma):
        # A stressed CVC monosyllable doubles: RIDDING, not RIDING.
        return False
    if suffix in {"ER", "ERS"} and lemma == base[:-1]:
        # Doubled -ER is usually a different word: BETTER, LETTER, BUTTER.
        return False
    return True


def _is_doubled_consonant(base: str) -> bool:
    return len(base) >= 2 and base[-1] == base[-2] and base[-1] not in _VOWELS


def _is_cvc(word: str) -> bool:
    return (
        len(word) >= 3
        and word[-1] not in _VOWELS
        and word[-1] not in "WXY"
        and word[-2] in _VOWELS
        and word[-3] not in _VOWELS
    )


@lru_cache(maxsize=8)
def load_root_lexicon(paths: tuple[str, ...]) -> frozenset[str]:
    """Load uppercase words from ``word;score`` (or bare word) list files.

    Missing files are skipped so a partial checkout still grades.
    """
    words: set[str] = set()
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                word = line.split(";", 1)[0].strip().upper()
                if word.isalpha():
                    words.add(word)
    return frozenset(words)
