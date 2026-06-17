"""Prompt template for LLM-powered clue generation."""

from __future__ import annotations

import json

from crossword_generator.exporters.numbering import NumberedEntry
from crossword_generator.models import (
    ClueEntry,
    ClueGrade,
    PuzzleDifficulty,
    PuzzleType,
    ThemeConcept,
)

_ROLE = (
    "You are an expert crossword puzzle constructor writing clues for a completed grid."
)

_GUIDELINES = (
    "GUIDELINES:\n"
    "- Write one clue per entry. Every clue must have "
    "exactly one defensible answer.\n"
    "- DO NOT use the answer word, any answer word-part, or any related "
    "morphological variant/root in the clue. This includes singular/plural "
    "forms, verb forms, compounds, abbreviation expansions, and famous-title "
    "fill-in-the-blanks that would point at a different form of the answer. "
    'For example, do not clue HOUSEWIFE with "Desperate ___wives"; do not '
    'clue WIFE with "wives" or "wifely"; do not clue TEACHER with '
    '"teaches" or "teaching"; do not clue POL with "politician" or '
    '"political"; do not clue OPS with "operations"; do not clue CEO '
    'with "executive"; do not clue EST with "Eastern Standard Time" '
    "or any words its initials stand for.\n"
    "- ABBREVIATION RULE (strict): if the answer is an abbreviation, acronym, "
    "or initialism, the clue must NOT contain ANY word that its letters stand "
    "for — not even a generic one. Each letter literally IS that word, so "
    "using it gives the answer away. For example: do not clue ETA with "
    '"arrival" OR "time" (ETA = Estimated Time of Arrival); do not clue MCL '
    'with "ligament" (Medial Collateral Ligament); do not clue CPR with '
    '"resuscitation"; do not clue GPA with "grade" or "average"; do not '
    'clue ATM with "machine". Clue the thing itself instead (e.g. ETA: '
    '"Flight-board figure, for short"; MCL: "Knee part that can tear, '
    'briefly").\n'
    "- ABBREVIATION INDICATOR RULE (strict): if the answer is an "
    "abbreviation, acronym, or initialism (DNA, CIA, USA, ETA, EST, ATM, "
    "etc.), the clue MUST signal that the answer is shortened. Add an "
    'indicator such as ", for short", ", briefly", ", in brief", or a '
    'parenthetical "(abbr.)" — or use an abbreviated cue word that makes '
    'the short form obviously expected (e.g. "org.", "agcy.", "assn."). '
    'For example, clue DNA as "Genetic blueprint material, for short" not '
    '"Genetic material"; clue CIA as "Cloak-and-dagger org." not '
    '"Spy agency"; clue USA as "Letters on some Olympic uniforms". An '
    "initialism or abbreviation answer with NO such indicator is a defect "
    "— a solver should be able to tell from the clue that the answer is a "
    "short form. This applies even when the rest of the clue is clean.\n"
    "- ETYMOLOGY / SPELLING RULE (strict): do not use a clue word that shares "
    "an obvious root, etymology, or spelling with the answer, even if it is a "
    'different word. Do not clue NAVAL with "navy" (same root); do not clue '
    'LUNAR via "moon"-Latin, SOLAR via "sun"-root words, or DENTAL via '
    '"teeth/dental"; do not clue KNEE with "kneel". Also never spell out a '
    'fragment of the answer: do not clue TRI with "Start of triangle", PRE '
    'with "Prefix...", or any answer with a clue word it literally begins, '
    "ends, or spells.\n"
    "- DO NOT use any of an entry's crossing words "
    "in that entry's clue.\n"
    "- Accuracy is more important than cleverness, freshness, or difficulty. "
    "For every clue, verify the fact, grammar, number, tense, part of speech, "
    "and exact phrase match. If you are not certain a proper noun, song, "
    "quote, sports team name, idiom, or pop-culture reference is correct, "
    "drop that angle. On Easy puzzles fall back to a plain dictionary-style "
    "clue; on Hard puzzles fall back to a different angle that is both solid "
    "and appropriately difficult, not to an instantly-solvable clue.\n"
    "- MULTI-REFERENCE RULE (strict): if a clue cues the answer through more "
    "than one reference joined by \"or\"/\"and\" (e.g. two name blanks like "
    '"X ___ or actor ___ Wilson"), EVERY reference must independently and '
    "exactly produce the answer. Verify each blank separately. A clue is wrong "
    "if even ONE reference is false, even when the other is correct — for "
    'OWEN, "Actor ___ Wilson" is right but "Architect Frank Lloyd ___" is '
    "WRONG (it is Frank Lloyd Wright, not Owen), so the whole clue is a "
    "defect. Do not pair a correct reference with a half-remembered one to "
    "look clever; if you cannot verify every component, use a single "
    "reference you are sure of, or a plain definition.\n"
    "- Fill-in-the-blank clues must fit the answer exactly, including "
    "singular/plural, tense, spacing, and contractions. Do not clue SAINT "
    'with the plural Saints, BATTING with "At ___", or GOT A SAY with '
    '"gotta say" / "my two cents" phrasing.\n'
    "- Vary clue styles within the target difficulty: definitional, "
    "fill-in-the-blank, wordplay, trivia, and lateral thinking are all "
    "available, but Easy clues should stay direct and obvious.\n"
    "- Do not repeat the exact same clue wording for the same answer across "
    "different puzzles. If prior clues are listed for an answer, write a new "
    "clue that does not exactly duplicate any of them — and when those prior "
    "clues all lean on one reference or meaning, choose a different fair "
    "angle this time rather than rewording the same one.\n"
    "- Use misdirection and cleverness only when they fit the target "
    "difficulty. For Easy clues, clarity beats cleverness.\n"
    "- Use question marks for witty/punny clues only when the target "
    'difficulty allows it (e.g., "Plant manager?" for GARDENER). For Easy '
    "clues this is FORBIDDEN: never end an Easy clue with a wordplay/tricky "
    "question mark. A literal question mark INSIDE quoted material is fine "
    '(e.g., "___ you okay?" for ARE, or a quoted question) — the ban is only '
    "on the trick-signalling ? that marks a clue as punny.\n"
    "- HYPHENATION RULE: do not hyphenate open compounds. Noun phrases like "
    '"snooze button", "lobster feast", "ice cream", "high school", and "real '
    'estate" are two separate words — never "snooze-button" or "lobster-feast". '
    "Only hyphenate where standard dictionary (Merriam-Webster) style requires "
    'it: compound modifiers before a noun ("well-known author", "two-time '
    'champ") and genuinely hyphenated words ("self-esteem", "X-ray"). When '
    "unsure, leave the words open.\n"
    "- Keep clues concise — say exactly what's needed, "
    "no filler words.\n"
    "- Avoid obscure trivia that solvers can't "
    "reason toward.\n"
    "- For pop-culture, celebrity, entertainment, sports, brand, and "
    "historical references, apply a sliding familiarity standard: the older "
    "or more niche the reference is, the more broadly iconic it must be. "
    "Avoid dated references that mostly reward one generation, fandom, or "
    "era. If an answer can be clued through an everyday meaning instead, use "
    "that angle over a stale proper-noun reference.\n"
    "- TIMELESS CLUE RULE (strict): a clue's truth must not depend on the "
    "CURRENT state of the world. Affiliations change, and puzzles are solved "
    "long after they are written — and your own knowledge of current jobs, "
    "broadcast rights, and rosters may already be out of date. Do not clue "
    "via someone's current employer, show, or role; a network's current "
    "broadcast rights; a team's current roster, coach, or city; a reigning "
    "champion or record holder; a current CEO, host, or anchor; or wording "
    'like "currently", "newest", or "latest". For example, do not clue TNT '
    'as "Channel airing NBA games" or ARI as "NPR\'s Shapiro". Prefer angles '
    "anchored in completed, permanent facts (awards won, famous works, fixed "
    "historical events) or non-temporal meanings. If a person/brand "
    "association is genuinely the best angle, phrase it so it stays true "
    'forever: "longtime", "Emmy-winning", or past tense ("Longtime \'All '
    "Things Considered' host Shapiro\").\n"
    "- REFERENCE VARIETY: when an answer has several well-known angles, do "
    "not default to the single most famous reference every time. If prior "
    "clues are listed for an answer, pick an angle or reference those clues "
    "have NOT already leaned on — e.g. if ALI has already been clued via "
    "Muhammad Ali, reach for Mahershala Ali, Ali Wong, or Ali Baba instead "
    "of rewording the boxing angle.\n"
    "- ROMAN NUMERAL RULE: if the answer reads as a Roman numeral (III, LIV, "
    'MMX), do not write a bare conversion clue like "54 in Roman numerals" '
    'or a random-context one like "Year in Claudius\'s reign". Prefer a '
    "real-world angle: a person or name (LIV → actress Liv Tyler), a famous "
    'title or event that uses the numeral (Super Bowl LIV, "Rocky III"), or '
    "a broadly known year (MMX → the Vancouver Olympics year). Use a plain "
    "numeral-conversion clue only when no familiar angle exists.\n"
    '- Avoid unpleasant clue wording. Do not use terms like "death" or '
    '"undocumented immigrant" in clues. If a clue must refer to dying, use '
    'a gentle euphemism like "passed on" rather than blunt wording.\n'
    "- Make clues culturally accessible and "
    "contemporary where possible.\n"
    '- Do not add word-count tags like "(two words)" or "(three words)". '
    "Use no word-count enumeration unless the pipeline provides explicit "
    "word-boundary metadata.\n"
    "- TITLE QUOTING RULE: put quotation marks around titles of movies, TV "
    "shows, songs, albums, books, plays, and video games — including in "
    'fill-in-the-blank clues. Write "\\"Better Call ___\\"" for SAUL, not '
    '"Better Call ___"; write "\\"Hey Jude\\" band" not "Hey Jude band". Do '
    "not quote band names, brand names, character names, or network names.\n"
    "- Put explanatory tags in parentheses, not after a comma or colon. "
    'Use "To the ___ (in the extreme)" rather than '
    '"To the ___: in the extreme"; use '
    '"Dennis ___ (pop art icon of soup cans)" rather than '
    '"Dennis ___, pop art icon of soup cans".\n'
    "- Avoid body-part, underwear, anatomy, sexuality, appearance, "
    "identity, or demographic-based jokes. If an answer is sensitive "
    "but acceptable, clue it neutrally rather than with a wink, "
    "innuendo, or objectifying angle.\n"
    "- Always write clues with a non-offensive mindset. When an answer "
    "has a violent, weapon-, drug-, or otherwise negative literal "
    "meaning, prefer a clue built on a benign alternate sense, idiom, "
    'or pop-culture usage. For example, clue BOMB as "Flop at the box '
    'office" rather than referencing an explosive device, and clue '
    'SHOOT as "Rats!" rather than referencing gunfire.\n'
    "- NO GUNS OR FIREARMS: do not reference guns, firearms, ammunition, "
    "shooting, or gun culture, even playfully. Avoid gun idioms and "
    'phrases entirely. For example, do not clue PIN as "Locked and '
    'loaded"-style wordplay or via any firearm angle; clue it with a '
    'benign sense like "Bowling-alley target" or "Sewing-kit item". '
    "Steer clear of any clue that evokes weapons or violence.\n"
    "- NO ALCOHOL, DRUNKENNESS, OR DRUGS: do not reference being drunk, "
    "drinking, hangovers, sobriety as recovery, or drug use. Clue such "
    "answers through neutral, everyday senses instead. For example, do "
    'not clue SOBER as "Newly off the bottle" or via any '
    'drinking/recovery angle; clue it as "Serious and solemn" or '
    '"Clear-headed". When an answer can read as alcohol- or '
    "drug-related, pick a different, wholesome meaning.\n"
    "- Keep the overall tone family-friendly and broadly welcoming. When "
    "in doubt about whether a clue's subject matter (weapons, alcohol, "
    "drugs, violence, or any edgy theme) could draw a complaint, choose "
    "a plainer, more neutral angle."
)

_THEME_INSTRUCTIONS = (
    "THEME CLUE INSTRUCTIONS:\n"
    "For each [THEME ENTRY], choose ONE of these three styles — "
    "and VARY the style across theme entries:\n"
    "\n"
    "1. STANDALONE: A clean clue with no reference to the theme.\n"
    "   Example (theme: CAESAR reveals Roman words): "
    'TOGA → "Garment at a frat party"\n'
    "\n"
    "2. INDIRECT ALLUSION: A standalone clue that subtly nods to\n"
    "   the theme concept without being heavy-handed.\n"
    '   Example: TOGA → "What a senator might have worn '
    'to the forum"\n'
    "\n"
    "3. POSITIONAL CROSS-REFERENCE: Reference the revealer by\n"
    "   number. Use sparingly — at most ONE theme entry should\n"
    "   use this style. The cross-reference must read naturally\n"
    "   as a standalone, solvable clue — never a bare\n"
    '   "See X-Across" with no solving context.\n'
    '   Example: TOGA → "Garb of the revealer\'s era"\n'
    "\n"
    "IMPORTANT:\n"
    "- Vary the style across theme entries. Do NOT use the same "
    "approach for all of them.\n"
    '- Do NOT write "one of [REVEALER ANSWER]" — this is '
    "grammatically unnatural for most theme types.\n"
    '- NEVER write a bare "See X-Across" or "See X-Down" '
    "with no standalone definition. Every clue must give the "
    "solver something to work with on its own.\n"
    "\n"
    "For the [REVEALER] entry:\n"
    "1. First identify the connecting element between the "
    "revealer and the theme entries. The revealer answer itself "
    "may not be the connecting word — a component of it may be "
    "(e.g., for GOLDMINES with theme entries BAR/COIN/ORE, the "
    "connecting element is GOLD, not GOLDMINES).\n"
    "2. Write a standalone definition of the revealer answer "
    "first.\n"
    "3. Then optionally add a theme hint that accurately "
    "describes the connection. Make sure your hint is factually "
    "accurate about which word or word-part connects to the "
    "theme entries.\n"
    '4. Never use the phrase "theme entries."\n'
)

# Worked good-vs-bad examples, one per leak class. This block is identical
# across every clue-generation call, so it is part of the cacheable system
# prefix — and it teaches the rules concretely, which a flat list of "do not"
# bullets does not. Each entry shows the answer, a BAD clue that leaks, why it
# leaks, and a GOOD replacement.
_LEAK_EXAMPLES = (
    "WORKED EXAMPLES — study these. Each shows an answer, a clue that LEAKS "
    "(never write these), why it leaks, and a clean replacement.\n"
    "\n"
    "1. Answer word in the clue (exact):\n"
    '   ANSWER: OCEAN  BAD: "The ocean covers most of Earth" — uses OCEAN.\n'
    '   GOOD: "Atlantic or Pacific"\n'
    "\n"
    "2. Shared root / morphological variant:\n"
    '   ANSWER: TEACHER  BAD: "One who teaches" — teaches shares TEACH-.\n'
    '   GOOD: "School-room leader"\n'
    '   ANSWER: BAKING  BAD: "What a baker does" — baker shares BAK-.\n'
    '   GOOD: "Using the oven, as cookies"\n'
    "\n"
    "3. Abbreviation expansion-word (any word the letters stand for, even a "
    "generic one):\n"
    '   ANSWER: ETA  BAD: "Arrival time, for short" — ETA = Estimated TIME of '
    "ARRIVAL, so both 'arrival' (A) and 'time' (T) leak.\n"
    '   GOOD: "Flight-board figure, for short"\n'
    '   ANSWER: MCL  BAD: "Knee ligament, briefly" — MCL = Medial Collateral '
    "LIGAMENT.\n"
    '   GOOD: "Knee part that can tear, in brief"\n'
    "\n"
    "4. Shared etymology / root (different word, same origin):\n"
    '   ANSWER: NAVAL  BAD: "Of the navy" — naval and navy share a root.\n'
    '   GOOD: "Like a fleet\'s officers"\n'
    '   ANSWER: KNEE  BAD: "Joint used to kneel" — kneel shares KNEE-.\n'
    '   GOOD: "Joint above the shin"\n'
    "\n"
    "5. Spelling fragment (answer is the literal start/letters of a clue word):\n"
    "   ANSWER: TRI  BAD: \"Start of 'triangle'\" — TRI is literally the first "
    "three letters of TRIANGLE.\n"
    '   GOOD: "Prefix for cycle or pod"\n'
    "   (Note: 'Prefix for...' is fine here because the clue word 'cycle' does "
    "not share letters with TRI; do not clue PRE with 'Prefix...'.)\n"
    "\n"
    "6. Collocation fill-in-the-blank (the blank + a partner word forms a "
    "phrase that uniquely gives the answer):\n"
    '   ANSWER: SOY  BAD: "___ sauce" — soy sauce is a fixed phrase.\n'
    '   GOOD: "Tofu base"\n'
    '   ANSWER: LIST  BAD: "Shopping ___" — shopping list is fixed.\n'
    '   GOOD: "Itemized rundown"\n'
    "\n"
    "7. Singular/plural and tense leaks:\n"
    '   ANSWER: SAINT  BAD: "New Orleans NFL player" points at SAINTS '
    "(plural).\n"
    '   GOOD: "Halo wearer"\n'
    '   ANSWER: RAN  BAD: "Run a marathon" — run shares the verb stem.\n'
    '   GOOD: "Competed in a track event, say"\n'
    "\n"
    "8. Crossing-word leaks (never use a word that crosses the entry):\n"
    "   If RAP crosses RUTH, do not put RUTH (or Ruth) anywhere in RAP's "
    "clue, and vice versa.\n"
    "\n"
    "9. More good clues that look easy but are clean:\n"
    '   ANSWER: HBO  GOOD: "\\"Game of Thrones\\" network"\n'
    '   ANSWER: VERMONT  GOOD: "New England state known for maple syrup"\n'
    '   ANSWER: EGO  GOOD: "One\'s sense of self-importance"\n'
    '   ANSWER: NASA  GOOD: "Moon-landing org."  (acronym clued by what it '
    "did, not by its expansion words, and the \"org.\" cue signals a short "
    "form)\n"
    '   ANSWER: DNA  BAD: "Genetic material" — no indicator that the answer '
    'is an initialism.  GOOD: "Genetic blueprint stuff, for short"\n'
    '   ANSWER: CIA  GOOD: "Cloak-and-dagger org."  (the "org." cue marks '
    "it as an abbreviation)\n"
    "\n"
    "10. Difficulty calibration — same answer, Easy vs Hard:\n"
    '   ANSWER: OCEAN   Easy: "Atlantic or Pacific"   '
    'Hard: "Vast expanse with a \\"floor\\" you\'ll never sweep"\n'
    '   ANSWER: EDGE    Easy: "Border of a table"     '
    'Hard: "Competitive advantage"\n'
    '   ANSWER: TIER    Easy: "Wedding-cake layer"    '
    'Hard: "Level in a ranking"\n'
    '   ANSWER: ARC     Easy: "Curved path"           '
    'Hard: "Rainbow\'s shape"\n'
    '   ANSWER: THREE   Easy: "One more than two"     '
    'Hard: "A crowd, proverbially"\n'
    "   For Easy, prefer the instantly-solvable angle. For Hard, the Easy "
    "column above is NOT acceptable — an instantly-solvable clue on a Hard "
    "puzzle is a defect. Use the fair-but-less-direct angle, while accuracy "
    "and the leak rules above ALWAYS take priority over difficulty.\n"
    "\n"
    "11. Compound / hidden-word leaks (a clue word that CONTAINS the answer):\n"
    '   ANSWER: WIFE  BAD: "Word in \\"Desperate Housewives\\"" — housewives '
    "contains the answer's form.\n"
    '   GOOD: "Spouse, traditionally"\n'
    '   ANSWER: ART  BAD: "Studio output of an artist" — artist starts with '
    "ART.\n"
    '   GOOD: "Museum display"\n'
    "\n"
    "12. Proper-noun accuracy (only use a name/title/fact if you are certain "
    "it is exactly right):\n"
    '   ANSWER: RUTH  GOOD: "Babe ___ (baseball legend)"  (verified pairing)\n'
    '   ANSWER: ANDREWS  GOOD: "Julie ___ (\\"Mary Poppins\\" actress)"\n'
    "   If unsure whether a proper noun, song, quote, team, or date is exact, "
    "use a plain dictionary-style clue instead of risking a wrong fact.\n"
    "\n"
    "13. Question-mark / pun clues (Hard only, and only when airtight):\n"
    '   ANSWER: GARDENER  Hard: "Plant manager?"  (the ? signals wordplay)\n'
    "   Never use a ? clue on Easy, and never when the pun is a stretch.\n"
    "\n"
    "14. Roman-numeral answers (clue the real-world thing, not the "
    "conversion):\n"
    '   ANSWER: LIV  BAD: "54, in old Rome" — bare numeral conversion with '
    "no real-world anchor.\n"
    '   GOOD: "Actress Tyler of \\"Armageddon\\"" or "Super Bowl the Chiefs '
    'won in 2020"\n'
    '   ANSWER: III  GOOD: "\\"Rocky ___\\" (1982 sequel)"\n'
    "\n"
    "15. Reference variety across puzzles (when prior clues are listed):\n"
    '   ANSWER: ALI  Prior clues: "Boxing legend Muhammad"; "The Greatest '
    'of the ring"\n'
    '   BAD: "Heavyweight champ Muhammad" — same angle reworded.\n'
    '   GOOD: "Mahershala of \\"Green Book\\"" or "Comedian Wong"\n'
    "\n"
    "16. Current-state / stale-fact clues (clue truth must not depend on the "
    "present moment):\n"
    '   ANSWER: TNT  BAD: "Channel airing NBA playoff games" — broadcast '
    "rights change, so this clue goes stale on the shelf.\n"
    '   GOOD: "Explosive letters" or "Classic AC/DC hit"\n'
    "   ANSWER: ARI  BAD: \"NPR's Shapiro\" — current-employer claims expire "
    "when people change jobs.\n"
    '   GOOD: "Director Aster of \\"Hereditary\\"" or "Gold of '
    '\\"Entourage\\""\n'
    "\n"
    "SELF-CHECK before finalizing EACH clue: (a) does the clue contain the "
    "answer, any part of it, a shared root, an etymological cousin, or a "
    "spelling fragment? (b) if the answer is an abbreviation, does the clue "
    "contain ANY word its letters stand for? (c) does the clue use a crossing "
    "word? (d) is it a fill-in-the-blank whose phrase gives the answer away? "
    "(e) does the clue's truth depend on a current-state fact that could "
    "change — a current role, broadcast deal, roster, or title holder? "
    "(f) if the answer is an abbreviation, acronym, or initialism, does the "
    'clue include an indicator that it is shortened (", for short", '
    '", briefly", "(abbr.)", or an "org."-style abbreviated cue)? '
    "(g) does the clue reference guns/firearms/shooting or "
    "alcohol/drunkenness/drugs, even playfully? "
    "If yes to (a)-(e) or (g), or NO to (f), rewrite the clue before "
    "returning it.\n"
)

_EXAMPLE_OUTPUT = json.dumps(
    [
        {"number": 1, "direction": "across", "clue": "Example clue text"},
        {"number": 1, "direction": "down", "clue": "Example clue text"},
    ],
    indent=2,
)

_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a JSON array with one object per entry. "
    "No other text before or after.\n"
    f"\n{_EXAMPLE_OUTPUT}\n"
    "\nNow write clues for every entry listed in the user "
    "message. Return ONLY the JSON array."
)


def _difficulty_guidance(puzzle_type: PuzzleType, difficulty: PuzzleDifficulty) -> str:
    if difficulty == PuzzleDifficulty.EASY:
        base = (
            "This is an HGG Easy crossword for a very broad casual audience. "
            "Write clues easier than an NYT Monday. Default to direct "
            "definitions, familiar everyday meanings, and totally obvious "
            "fill-in-the-blank clues. Avoid oblique definitions, tricky "
            "wordplay, niche trivia, and lateral-thinking clues. If choosing "
            "between clever and instantly solvable, choose instantly solvable. "
            "NEVER use a wordplay/tricky question-mark clue (the punny "
            '"?"-tagged kind) in an Easy puzzle. A literal "?" inside quoted '
            'text is fine (e.g., "___ you okay?" for ARE); only the trick-'
            "signalling question mark is banned."
        )
    else:
        base = (
            "This is an HGG Hard crossword. Aim for solid NYT Tuesday "
            "difficulty — Wednesday is the CEILING, not the target: still "
            "fair and broadly accessible, but every clue "
            "should make the solver pause and think for a beat. A clue the "
            "average solver answers instantly with no crossing letters is TOO "
            "EASY for this puzzle and is a defect, the same way an inaccurate "
            "clue is. Never write Easy-style giveaways like a plain primary "
            'definition, an obvious fill-in-the-blank, or arithmetic ("One '
            'more than two" for THREE is exactly what NOT to write here — use '
            'an angle like "A crowd, proverbially" instead). For each entry, '
            "DEFAULT to a harder fair angle: a common secondary meaning, a "
            "less-expected but accurate definition, a specific example instead "
            "of the category, fair trivia solvers can reason toward, mild "
            "misdirection, or an occasional question-mark clue. Reserve plain "
            "direct definitions for entries that genuinely cannot support a "
            "harder fair angle (some abbreviations, partials, and awkward "
            "glue) — they should be the exception, roughly a quarter of the "
            "clues at most, not the default. DO NOT STRAIN for difficulty: "
            "pick the most natural fair angle that is not a giveaway, not "
            "the hardest angle you can construct. If an entry's natural "
            "fair angles are only moderately tricky, use the best of those "
            "— a clean, modestly challenging clue is better than a "
            "contrived or ridiculously hard one. Difficulty should come "
            "from ONE fair twist (a secondary meaning, a specific example, "
            "or mild misdirection), never from obscurity, convoluted "
            "phrasing, or stacking multiple tricks in one clue. When in "
            "doubt between two fair angles, choose the easier one. "
            "Accuracy still constrains WHICH "
            "hard angle you pick: every clue must remain factually airtight "
            "and reasonably inferable, and if a specific clever angle is "
            "factually shaky, pick a different harder angle that is solid — "
            "do not fall back to an instantly-solvable Easy clue. Avoid "
            "Saturday-level obscurity and trivia that solvers cannot reason "
            "toward. Do not force difficulty with strained pop-culture "
            "references, ultra-current slang, or clues that only work after a "
            "long explanation. Older pop-culture references are appropriate "
            "only when they are cross-generationally iconic."
        )
    if puzzle_type == PuzzleType.MINI:
        return (
            f"{base} Because this is a MINI crossword, keep clues especially concise."
        )
    return (
        f"{base} Because this is a MIDI crossword, allow a little more surface "
        "variety while preserving the target difficulty."
    )


def build_clue_generation_messages(
    entries: list[NumberedEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
    prior_clues_by_answer: dict[str, list[str]] | None = None,
) -> tuple[str, str]:
    """Build (system, user) messages for clue generation.

    The system text bundles the role, rubric, theme-style instructions,
    and output format — content that is identical across all puzzles of
    the same (puzzle_type, themed-or-not) shape, so it caches well.
    The user text carries per-puzzle data: the entries to clue and the
    theme topic/wordplay/revealer.

    Returns:
        Tuple of (system_text, user_text).
    """
    themed = bool(theme and theme.topic)

    system_parts = [
        _ROLE,
        _difficulty_guidance(puzzle_type, difficulty),
        _GUIDELINES,
        _LEAK_EXAMPLES,
    ]
    if themed:
        system_parts.append(_THEME_INSTRUCTIONS)
    system_parts.append(_OUTPUT_SECTION)
    system_text = "\n\n".join(system_parts)

    # Identify theme entries and revealer for annotation
    revealer_info: tuple[int, str] | None = None
    seed_answers: set[str] = set()
    if themed:
        seed_answers = {s.upper() for s in theme.seed_entries}
        for entry in entries:
            if entry.answer == theme.revealer.upper():
                revealer_info = (entry.number, entry.direction)
                break

    # Build the entry list section
    entry_lines: list[str] = []
    for entry in entries:
        key = (entry.number, entry.direction)
        crossings = crossing_words.get(key, [])
        crossing_str = ", ".join(crossings) if crossings else "none"
        tag = ""
        if entry.answer in seed_answers:
            tag = " [THEME ENTRY]"
        elif revealer_info and entry.answer == theme.revealer.upper():
            tag = " [REVEALER]"
        entry_lines.append(
            f"- {entry.number}-{entry.direction.upper()}: {entry.answer}{tag} "
            f"(crossing words: {crossing_str})"
        )
    entries_block = "\n".join(entry_lines)

    prior_clues_block = ""
    if prior_clues_by_answer:
        prior_lines: list[str] = []
        for answer in sorted(prior_clues_by_answer):
            clues = prior_clues_by_answer[answer]
            if not clues:
                continue
            quoted = "; ".join(f'"{clue}"' for clue in clues)
            prior_lines.append(f"- {answer}: {quoted}")
        if prior_lines:
            prior_clues_block = (
                "\nPRIOR CLUES FOR THESE ANSWERS (already used in other "
                "puzzles):\n"
                "Do not repeat any of these exactly. Also prefer a fresh "
                "angle: when the prior clues for an answer lean on one "
                "reference or meaning, clue it through a different fair "
                "angle this time.\n"
                + "\n".join(prior_lines)
                + "\n"
            )

    theme_context_block = ""
    if themed:
        if revealer_info:
            rev_label = f"{revealer_info[0]}-{revealer_info[1].capitalize()}"
        else:
            rev_label = theme.revealer
        revealer_clue_draft = ""
        if theme.revealer_clue:
            revealer_clue_draft = (
                f'- Revealer clue draft: "{theme.revealer_clue}" '
                f"(use as inspiration — rewrite to fit the grid context)\n"
            )
        theme_context_block = (
            "\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer} ({rev_label})\n"
            f"{revealer_clue_draft}"
            "- Theme entries are marked [THEME ENTRY] above\n"
            "- For style 3 (POSITIONAL CROSS-REFERENCE), reference the "
            f"revealer as {rev_label}.\n"
        )

    user_text = (
        f"{theme_context_block}"
        f"ENTRIES TO CLUE:\n{entries_block}\n\n"
        f"{prior_clues_block}"
        f"Now write clues for all {len(entries)} entries above."
    )

    return system_text, user_text


def build_clue_generation_prompt(
    entries: list[NumberedEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
    prior_clues_by_answer: dict[str, list[str]] | None = None,
) -> str:
    """Build a single-string prompt (system+user concatenated).

    Kept for callers that don't yet pass a separate system block.
    Prefer ``build_clue_generation_messages`` to enable prompt caching.
    """
    system_text, user_text = build_clue_generation_messages(
        entries, crossing_words, puzzle_type, theme, difficulty, prior_clues_by_answer
    )
    return f"{system_text}\n\n{user_text}"


def build_clue_repair_prompt(
    entries_to_repair: list[tuple[ClueEntry, ClueGrade]],
    all_clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
) -> str:
    """Build a prompt to regenerate only clues with accuracy problems.

    Args:
        entries_to_repair: Pairs of (clue, grade) for clues needing repair.
        all_clues: All clues in the puzzle (for context / avoid duplication).
        crossing_words: Maps (number, direction) to crossing answer words.
        puzzle_type: Mini or midi — affects difficulty guidance.
        theme: Optional theme concept for midi puzzles.

    Returns:
        A prompt string ready to send to the LLM.
    """
    system_text, user_text = build_clue_repair_messages(
        entries_to_repair,
        all_clues,
        crossing_words,
        puzzle_type,
        theme,
        difficulty,
    )
    return f"{system_text}\n\n{user_text}"


_REPAIR_ROLE = (
    "You are an expert crossword puzzle constructor. "
    "The following clues had accuracy, fairness, or editorial quality problems. "
    "They may be factually wrong, too strained, too obscure, leak part of the "
    "answer, have multiple defensible answers, or have grammar/part-of-speech "
    "mismatches. Write replacement clues that are clean, factually correct, "
    "and have exactly one defensible answer."
)

_REPAIR_GUIDELINES = (
    "GUIDELINES:\n"
    "- Each replacement clue must have exactly one defensible answer.\n"
    "- DO NOT use the answer word, any answer word-part, or any related "
    "morphological variant/root in the clue. This includes singular/plural "
    "forms, verb forms, compounds, abbreviation expansions, and famous-title "
    "fill-in-the-blanks that would point at a different form of the answer. "
    'For example, do not clue HOUSEWIFE with "Desperate ___wives"; do not '
    'clue POL with "politician" or "political"; do not clue CEO with '
    '"executive".\n'
    "- ABBREVIATION RULE (strict): if the answer is an abbreviation or "
    "initialism, the clue must NOT contain ANY word its letters stand for, even "
    'a generic one (e.g. do not clue ETA with "arrival" or "time"; do not '
    'clue MCL with "ligament"). Clue the thing itself instead.\n'
    "- ABBREVIATION INDICATOR RULE (strict): if the answer is an "
    "abbreviation, acronym, or initialism (DNA, CIA, USA, EST, etc.), the "
    "replacement clue MUST signal that the answer is shortened — via an "
    'indicator like ", for short", ", briefly", "(abbr.)", or an '
    'abbreviated cue word ("org.", "agcy."). A clue with no such '
    'indicator (e.g. DNA clued plainly as "Genetic material") is a '
    "defect even when otherwise clean.\n"
    "- ETYMOLOGY / SPELLING RULE (strict): do not use a clue word that shares "
    "an obvious root, etymology, or spelling with the answer (NAVAL/navy, "
    "KNEE/kneel), and never spell out a fragment of the answer (TRI/triangle, "
    "PRE/prefix).\n"
    "- Accuracy is more important than cleverness. Verify facts, grammar, "
    "number, tense, part of speech, and exact phrase match. If the old clue "
    "used a questionable proper noun, song, quote, sports team name, idiom, "
    "or pop-culture reference, drop that angle: on Easy, replace it with a "
    "plain accurate clue; on Hard, replace it with a different solid angle "
    "that still matches the target difficulty. Apply "
    "the same sliding familiarity standard: the older or more niche the "
    "reference is, the more broadly iconic it must be.\n"
    "- MULTI-REFERENCE RULE (strict): if a replacement cues the answer through "
    "more than one reference joined by \"or\"/\"and\" (e.g. two name blanks), "
    "EVERY reference must independently and exactly produce the answer. A clue "
    "is wrong if even one reference is false, even when another is correct "
    "(e.g. for OWEN, \"Architect Frank Lloyd ___\" is wrong — it is Wright). "
    "If you cannot verify every component, use a single sure reference or a "
    "plain definition.\n"
    "- TIMELESS CLUE RULE (strict): do not write a replacement whose truth "
    "depends on the CURRENT state of the world — a current employer, show, "
    "or role; current broadcast rights; a current roster, coach, host, "
    "anchor, or CEO; or a reigning champion or record holder. Your knowledge "
    "of such facts may be out of date. Prefer completed, permanent facts "
    "(awards won, famous works, fixed historical events) or non-temporal "
    "meanings; if a current association is the only fair angle, phrase it so "
    'it stays true forever ("longtime", "Emmy-winning", or past tense).\n'
    "- If a clue was flagged as TOO EASY for a Hard puzzle, the replacement "
    "must use a harder fair angle (secondary meaning, less-expected accurate "
    "definition, fair trivia, or mild misdirection) — do not return another "
    "instantly-solvable definition.\n"
    "- If a clue was flagged as TOO HARD, strained, or contrived, the "
    "replacement should be a clean NYT Tuesday-level clue: one fair twist "
    "(a secondary meaning or mild misdirection), not obscurity, convoluted "
    "phrasing, or stacked tricks.\n"
    "- TITLE QUOTING RULE: put quotation marks around titles of movies, TV "
    "shows, songs, albums, books, plays, and video games — including in "
    'fill-in-the-blank clues ("\\"Better Call ___\\"", not '
    '"Better Call ___").\n'
    "- If a clue was flagged as a duplicate of prior clues, do not just "
    "reword the same reference — switch to a different fair angle (a "
    "different person, meaning, or style) than the prior clues used.\n"
    "- ROMAN NUMERAL RULE: for answers that read as Roman numerals (III, "
    'LIV, MMX), avoid bare conversion clues ("54 in Roman numerals") and '
    "random filler contexts; prefer a real-world angle such as a person "
    "(LIV → Liv Tyler), a famous title or event (Super Bowl LIV, "
    '"Rocky III"), or a broadly known year.\n'
    "- Fill-in-the-blank clues must fit the answer exactly, including "
    "singular/plural, tense, spacing, and contractions.\n"
    "- DO NOT use any crossing words in the clue.\n"
    "- DO NOT duplicate phrasing from the existing clues listed below.\n"
    '- Do not add word-count tags like "(two words)" or "(three words)".\n'
    "- HYPHENATION RULE: do not hyphenate open compounds. Noun phrases like "
    '"snooze button" or "lobster feast" are two separate words — never '
    '"snooze-button" or "lobster-feast". Only hyphenate where standard '
    "dictionary (Merriam-Webster) style requires it: compound modifiers before "
    'a noun ("well-known author") and genuinely hyphenated words '
    '("self-esteem", "X-ray"). When unsure, leave the words open.\n'
    "- Put explanatory tags in parentheses, not after a comma or colon.\n"
    '- Avoid unpleasant clue wording such as "death" or '
    '"undocumented immigrant"; use gentle wording like "passed on" only '
    "if dying is unavoidable.\n"
    "- NO GUNS, ALCOHOL, OR DRUGS: the replacement must not reference "
    "firearms, ammunition, shooting, or gun culture, nor drunkenness, "
    "drinking, hangovers, or drug use — even playfully. Clue such answers "
    'through a benign everyday sense instead (e.g. PIN as "Bowling-alley '
    'target", SOBER as "Serious and solemn"). Keep the tone '
    "family-friendly; when in doubt, choose a plainer neutral angle.\n"
    "- Keep clues concise and culturally accessible."
)

_REPAIR_EXAMPLE = json.dumps(
    [{"number": 1, "direction": "across", "clue": "Replacement clue text"}],
    indent=2,
)

_REPAIR_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a JSON array with one object per repaired clue. "
    "No other text before or after.\n"
    f"\n{_REPAIR_EXAMPLE}\n"
    "\nWrite replacement clues for the entries listed in the user "
    "message. Return ONLY the JSON array."
)


def build_clue_repair_messages(
    entries_to_repair: list[tuple[ClueEntry, ClueGrade]],
    all_clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
) -> tuple[str, str]:
    """Build (system, user) messages for clue repair."""
    themed = bool(theme and theme.topic)

    system_parts = [
        _REPAIR_ROLE,
        _difficulty_guidance(puzzle_type, difficulty),
        _REPAIR_GUIDELINES,
    ]
    if themed:
        system_parts.append(_THEME_REPAIR_INSTRUCTIONS)
    system_parts.append(_REPAIR_OUTPUT_SECTION)
    system_text = "\n\n".join(system_parts)

    # Identify theme entries and revealer for annotation
    revealer_answer = ""
    seed_answers: set[str] = set()
    revealer_label = ""
    if themed:
        revealer_answer = theme.revealer.upper()
        seed_answers = {s.upper() for s in theme.seed_entries}
        for clue in all_clues:
            if clue.answer.upper() == revealer_answer:
                revealer_label = f"{clue.number}-{clue.direction.capitalize()}"
                break
        if not revealer_label:
            revealer_label = theme.revealer

    # Build the repair target section
    repair_lines: list[str] = []
    for clue, grade in entries_to_repair:
        key = (clue.number, clue.direction)
        crossings = crossing_words.get(key, [])
        crossing_str = ", ".join(crossings) if crossings else "none"
        tag = ""
        if clue.answer.upper() in seed_answers:
            tag = " [THEME ENTRY]"
        elif clue.answer.upper() == revealer_answer:
            tag = " [REVEALER]"
        repair_lines.append(
            f"- {clue.number}-{clue.direction.upper()}: "
            f"Answer={clue.answer}{tag}\n"
            f'  Old clue: "{clue.clue}"\n'
            f"  Problem: {grade.feedback}\n"
            f"  (crossing words: {crossing_str})"
        )
    repair_block = "\n".join(repair_lines)

    # Build context section — other clues already in the puzzle
    repair_keys = {(c.number, c.direction) for c, _ in entries_to_repair}
    context_lines: list[str] = []
    for clue in all_clues:
        if (clue.number, clue.direction) not in repair_keys:
            context_lines.append(
                f"- {clue.number}-{clue.direction.upper()}: "
                f'{clue.answer} = "{clue.clue}"'
            )
    context_block = "\n".join(context_lines) if context_lines else "(none)"

    theme_context_block = ""
    if themed:
        theme_context_block = (
            "\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer} ({revealer_label})\n"
            "- For cross-references, refer to the revealer as "
            f"{revealer_label}.\n"
        )

    user_text = (
        f"{theme_context_block}"
        f"CLUES TO REPAIR:\n{repair_block}\n\n"
        f"EXISTING CLUES (for context — do not duplicate):\n{context_block}\n\n"
        f"Write replacement clues for the "
        f"{len(entries_to_repair)} entries above."
    )

    return system_text, user_text


_THEME_REPAIR_INSTRUCTIONS = (
    "THEME REPAIR INSTRUCTIONS:\n"
    "If any entries to repair are marked [THEME ENTRY], their replacement "
    "clues can use any of these styles:\n"
    "1. Standalone clue (no theme reference)\n"
    "2. Indirect allusion to the theme concept\n"
    "3. Cross-reference to the revealer (use sparingly — must read "
    "naturally as a standalone, solvable clue, never a bare "
    '"See X-Across" with no solving context)\n'
    "\n"
    'Do NOT write "one of [REVEALER ANSWER]" — this is grammatically '
    "unnatural. Vary the style if multiple theme entries need repair.\n"
    'NEVER write a bare "See X-Across" or "See X-Down" with no '
    "standalone definition.\n"
    "\n"
    "If the [REVEALER] entry needs repair:\n"
    "1. Identify the connecting element — the revealer answer itself may "
    "not be the connecting word; a component of it may be (e.g., GOLD in "
    "GOLDMINES). Make sure your hint is factually accurate about which "
    "word or word-part connects to the theme entries.\n"
    "2. Write a standalone definition first, then optionally add a theme "
    'hint using natural phrasing. Never use "theme entries."'
)
