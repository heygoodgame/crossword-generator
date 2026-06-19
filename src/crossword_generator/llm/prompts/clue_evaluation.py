"""Prompt template for LLM-powered clue quality evaluation."""

from __future__ import annotations

import json

from crossword_generator.models import (
    ClueEntry,
    PuzzleDifficulty,
    PuzzleType,
    ThemeConcept,
)

_ROLE = (
    "You are an expert crossword puzzle editor evaluating clue quality. "
    "Score each clue on four rubric dimensions (0-25 points each):"
)

_RUBRIC = (
    "SCORING RUBRIC (each dimension 0-25 points):\n\n"
    "1. ACCURACY (0-25):\n"
    "   20-25: Factually correct, exactly one defensible answer, "
    "grammar and part of speech match perfectly.\n"
    "   10-19: Minor ambiguity (two plausible answers) or debatable "
    "accuracy that a solver could argue.\n"
    "    0-9:  Factually wrong, clearly multiple valid answers, or "
    "grammar/part-of-speech mismatch.\n"
    "   IMPORTANT: If your feedback identifies a factual error in the "
    "clue, the accuracy score MUST be 9 or below. Do not give 20+ "
    "accuracy to a clue you describe as incorrect. Also score ACCURACY "
    "0-9 for wrong proper nouns, wrong song/quote titles, singular/plural "
    "mismatches, tense mismatches, part-of-speech mismatches, and "
    "fill-in-the-blank clues where the exact answer does not fit. Examples: "
    'SAINT clued as the plural Saints, BATTING clued by "At ___", or '
    "GOT A SAY clued as if it were GOTTA SAY.\n"
    "   CURRENCY RISK: also score ACCURACY 0-9 when the clue's truth depends "
    "on the CURRENT state of the world — a current employer, show, or role; "
    "current broadcast rights; a current roster, host, anchor, or CEO; a "
    'reigning champion or record holder; or wording like "currently", '
    '"newest", or "latest". Such clues go stale (or already are stale '
    "without your knowledge), e.g. TNT clued as an NBA broadcaster or ARI "
    'clued as "NPR\'s Shapiro". Apply this even if you believe the fact is '
    "currently true, and say in the feedback that the clue needs a timeless "
    "angle (a completed fact, famous work, or non-temporal meaning).\n\n"
    "2. FRESHNESS (0-25): Does the clue style fit the target difficulty? "
    "For Easy clues, obvious definitions and fill-in-the-blank clues can "
    "score highly when they are clean and welcoming. For Hard clues, reward "
    "fair misdirection, wordplay, or creative angles that make the clue "
    "interesting without becoming obscure. For Hard clues, do not reward "
    "cleverness unless the clue is first clearly accurate and inferable. "
    "CRITICAL — too-easy clues on Hard puzzles: if this is a Hard puzzle and "
    "the clue is an Easy-style giveaway the average solver answers instantly "
    "(a plain primary definition, an obvious fill-in-the-blank, or arithmetic "
    'like "One more than two" for THREE), score FRESHNESS 0-9 and include '
    'the phrase "too easy" in the feedback. Plain direct clues are '
    "acceptable on Hard only for entries that cannot support a harder fair "
    "angle, such as abbreviations and partials.\n\n"
    "3. CRAFT (0-25): Is the language economical and the surface reading clean? "
    "Does the clue read naturally as English? Is the difficulty appropriate "
    "for the puzzle type? Penalize missing quotation marks around titles of "
    "movies, TV shows, songs, albums, books, and plays — a fill-in-the-blank "
    'like Better Call ___ must be written "Better Call ___" (quoted), and a '
    "title used in any clue must be quoted.\n\n"
    "4. FAIRNESS (0-25): Does the clue avoid echoing the answer word, answer "
    "word-parts, roots, or related morphological variants? Penalize "
    "singular/plural leaks, verb-form leaks, compound-part leaks, and "
    'abbreviation-expansion leaks such as CEO clued with "executive," plus '
    'OPS clued with "operations" and EST clued with "Eastern Standard '
    'Time," '
    "CRITICAL — abbreviation expansion-word leaks: if the answer is an "
    "abbreviation, acronym, or initialism, heavily penalize (fairness 0-8) a "
    "clue that contains ANY word its letters stand for, even a generic one. "
    'Examples to fail: ETA clued "Arrival time, for short" (arrival = A, time '
    '= T); MCL clued with "ligament" (L); CPR clued with "resuscitation"; GPA '
    'clued with "grade" or "average"; ATM clued with "machine." '
    "CRITICAL — etymology / spelling leaks: heavily penalize (fairness 0-8) a "
    "clue that uses a word sharing an obvious root, etymology, or spelling with "
    'the answer, even a different word: NAVAL clued "Of the navy"; KNEE clued '
    'with "kneel"; LUNAR with moon-root Latin; SOLAR with sun-root words. Also '
    "fail clues that spell out a fragment of the answer: TRI clued "
    '"Start of triangle"; PRE clued "Prefix meaning before." '
    "CRITICAL — hidden-answer leaks: heavily penalize (fairness 0-8) a clue "
    "whose words CONTAIN the answer's letters in sequence anywhere — at the "
    "start, middle, or END of a longer clue word — even when that clue word "
    "is unrelated in meaning and even when the answer is short. Examples to "
    'fail: BOT clued "Robot, informally" (roBOT ends with BOT); TIL clued '
    '"Until, briefly" (unTIL ends with TIL); MEN clued "Gentlemen" '
    "(gentleMEN ends with MEN). These should be clued through the answer's "
    "own meaning instead. "
    "famous-title fill-in-the-blanks that point at a different form of the "
    'answer, e.g. HOUSEWIFE clued via "Desperate ___wives." Does it avoid '
    "using any crossing words in the clue text? Penalize short answer roots "
    'inside longer clue words too, e.g. POL clued with "politician" or '
    '"political." '
    "CRITICAL — missing abbreviation indicator: if the answer is an "
    "abbreviation, acronym, or initialism (DNA, CIA, USA, ETA, EST, etc.) "
    "and the clue gives NO signal that the answer is shortened — no "
    '", for short", ", briefly", "(abbr.)", or "org."-style abbreviated '
    "cue — penalize CRAFT (0-12) and note the clue needs an abbreviation "
    'indicator. Example to flag: DNA clued plainly as "Genetic material" '
    '(should be "Genetic material, for short" or similar). '
    "CRITICAL — collocation give-aways: heavily penalize (fairness 0-8) any "
    "fill-in-the-blank clue whose blank plus a partner word forms a common "
    "fixed phrase or compound that uniquely gives away the answer, even when "
    'the answer text never appears. Examples to fail: SOY clued "___ sauce," '
    'LIST clued "Shopping ___," MILK clued "___ shake," RAIN clued "___ '
    'forest." These are unfair because the collocation makes the answer the '
    "only fit. Prefer a plain definition instead. A blank is only fair when "
    "several different words could plausibly complete the phrase. "
    "Is the clue culturally accessible without being obscure? Penalize "
    "strained pop-culture references, ultra-current slang, and Hard clues "
    "that try to create Friday/Saturday-level difficulty. "
    "Penalize FRESHNESS (0-12) for deep-cut or obscure senses that most "
    "solvers will not know, especially when a plain everyday angle exists — "
    "this includes reaching for a barely-known meaning of a word just to "
    'avoid reusing a prior clue (e.g. COS clued "Romaine lettuce variety" '
    "instead of the companies or cosine sense). For this primarily "
    "American-audience puzzle, also penalize FAIRNESS for deep-cut British "
    'or regional slang an American solver would not know (e.g. SNIP clued '
    '"A bargain"); common Briticisms in moderation are fine, but flag clues '
    "that lean on obscure regional slang when a plain angle exists. "
    "Apply a sliding "
    "familiarity standard to pop-culture, celebrity, entertainment, sports, "
    "brand, and historical references: the older or more niche the reference "
    "is, the more broadly iconic it must be. Penalize dated references that "
    "mostly reward one generation, fandom, or era, especially when a clean "
    "everyday clue angle is available. "
    "Penalize FRESHNESS for bare Roman-numeral conversion clues: when an "
    "answer reads as a Roman numeral (III, LIV, MMX), a clue like "
    '"54 in Roman numerals" or a random context like "Year in Claudius\'s '
    'reign" should score low on FRESHNESS when a real-world angle exists '
    "(a person such as Liv Tyler, a famous title or event such as Super "
    'Bowl LIV or "Rocky III", or a broadly known year). '
    "Penalize "
    'word-count tags like "(two words)" unless explicit word-boundary '
    "metadata was provided, and reward explanatory tags only when they are "
    'formatted parenthetically, e.g. "To the ___ (in the extreme)." '
    "Penalize "
    "body-part, underwear, anatomy, sexuality, appearance, identity, "
    "or demographic-based jokes, especially when a neutral clue would "
    'be more welcoming. Penalize unpleasant wording such as "death" or '
    '"undocumented immigrant"; if dying must be referenced, gentle '
    'euphemisms like "passed on" are preferred. '
    "CRITICAL — guns, alcohol, and drugs: heavily penalize (fairness 0-8) "
    "any clue that references firearms, ammunition, shooting, or gun "
    "culture, or that references drunkenness, drinking, hangovers, or drug "
    'use — even playfully. Examples to fail: PIN clued "Locked and loaded" '
    '(gun wordplay); SOBER clued "Newly off the bottle" (drinking/recovery '
    "angle). These answers should be clued through a benign everyday sense "
    "instead. Note in the feedback that the clue needs a neutral, "
    "family-friendly angle."
)

_THEME_EVAL_RULES = (
    "THEME CLUE EVALUATION:\n"
    "- [THEME ENTRY] clues: reward well-crafted standalone clues "
    "(high CRAFT, ACCURACY). A theme entry clue does NOT need to "
    "reference the revealer to score well.\n"
    "- FRESHNESS penalty: if ALL theme entry clues use the same "
    "style (e.g., all cross-reference the revealer, or all use "
    "the identical formula), deduct FRESHNESS for lack of variety.\n"
    '- FRESHNESS penalty: formulaic phrasing like "one of '
    '[REVEALER ANSWER]" should be scored low — it is '
    "grammatically unnatural for most theme types.\n"
    "- ACCURACY penalty: if the [REVEALER] clue claims the full "
    "revealer word has a relationship to theme entries that only "
    "a *component* of the word has (e.g., saying GOLDMINES "
    "precedes BAR when only GOLD does), score ACCURACY low. The "
    "connecting element may be a sub-part of the revealer, not "
    "the full answer.\n"
    '- CRAFT penalty: a bare "See X-Across" or "See X-Down" '
    "cross-reference with no standalone definition should score "
    "low on CRAFT. Cross-references must integrate the revealer "
    "position naturally into a real, solvable clue. Awkward "
    'appendages like "per X-Across" tacked onto an otherwise '
    "complete clue should also be penalized.\n"
    "- [REVEALER] clue: reward clues that elegantly explain the "
    'theme connection using natural language (e.g., "a hint to '
    'some other answers in this puzzle"). Penalize FRESHNESS if '
    'the revealer clue uses the phrase "theme entries."'
)

_EXAMPLE_OUTPUT = json.dumps(
    [
        {
            "number": 1,
            "direction": "across",
            "accuracy": 22,
            "freshness": 18,
            "craft": 20,
            "fairness": 15,
            "feedback": "Misdirection slightly vague.",
        },
        {
            "number": 2,
            "direction": "down",
            "accuracy": 24,
            "freshness": 23,
            "craft": 24,
            "fairness": 24,
        },
    ],
    separators=(",", ":"),
)

_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a minified JSON array (no indentation, no spaces after "
    "colons/commas) with one object per clue. Return the four sub-scores "
    "(accuracy, freshness, craft, fairness — each 0-25), NOT a total score. "
    "Include a SHORT \"feedback\" string (under ~10 words) ONLY for clues with "
    "a real problem — name the issue tersely (e.g. \"leaks: navy\", \"too easy "
    "for Hard\", \"wrong: Frank Lloyd Wright\"). OMIT the \"feedback\" field "
    "entirely for clean, high-scoring clues; do not write praise. Keep the "
    "exact issue keywords the repair step looks for. "
    "No other text before or after.\n"
    f"\n{_EXAMPLE_OUTPUT}\n"
    "\n(First object: a flagged clue with terse feedback. Second: a clean clue "
    "— scores only, no feedback field.)\n"
    "Evaluate every clue listed in the user message. "
    "Return ONLY the JSON array."
)


def _difficulty_note(puzzle_type: PuzzleType, difficulty: PuzzleDifficulty) -> str:
    if difficulty == PuzzleDifficulty.EASY:
        base = (
            "This is an HGG Easy crossword that should be ridiculously "
            "beginner-friendly. Reward clues that are easier than NYT Monday: "
            "direct, obvious, familiar, and instantly solvable — plain "
            "definitions, obvious fill-in-the-blanks on familiar phrases, "
            "one-word synonyms, clear antonyms, simple categories/examples, "
            "and helpful sound-alikes. Never penalize a clue for being too "
            "plain, too direct, or for reusing a familiar angle a prior clue "
            "used — for Easy, staying easy is the goal. "
            "CRITICAL — too-hard Easy clue: if a clue uses a secondary, "
            "technical, or less-common sense of the answer when a plainer "
            "everyday sense exists (e.g. DAM clued \"Foal's mother\" instead "
            'of "River barrier"), or otherwise requires real thought, score '
            'FRESHNESS 0-9 and state "too hard for Easy" in the feedback so '
            "it is regenerated with a simpler angle. "
            "Penalize oblique definitions, "
            "tricky wordplay, niche trivia, or lateral-thinking clues as too "
            "hard for this audience. A wordplay/tricky question-mark clue (a "
            'clue ending in a "?" that signals a pun or trick) is NOT allowed '
            "in an Easy puzzle: flag it and score freshness low so it is "
            'regenerated. A literal "?" inside quoted material (e.g., '
            '"___ you okay?" for ARE, or any quoted question) is fine — only '
            "the trick-signalling question mark is disallowed."
        )
    else:
        base = (
            "This is an HGG Hard crossword. Reward solid NYT Tuesday-level "
            "clues — Wednesday is the ceiling, not the target: fair, broadly "
            "accessible, and a notch more challenging — "
            "every clue should make the solver pause and think for a beat. "
            "Accuracy and exact answer fit are more important than difficulty. "
            "Mild misdirection, common secondary meanings, wordplay, and "
            "occasional question-mark clues are appropriate only when they are "
            "factually airtight and reasonably inferable. "
            "CRITICAL — too-hard clues: if a clue is strained, contrived, "
            "stacks multiple tricks, or is unfairly hard for a Tuesday "
            "(Saturday-level obscurity, a leap solvers cannot reason toward, "
            "or wordplay that only works after a long explanation), score "
            'FRESHNESS 0-9 and include the phrase "too hard" in the '
            "feedback. Difficulty should come from ONE fair twist; reward "
            "clean, modestly challenging clues over maximal difficulty. "
            "Also penalize "
            "forced cleverness, ultra-current slang, "
            "older pop-culture references that are not cross-generationally "
            "iconic, or trivia solvers cannot reason toward. Equally, penalize "
            "clues that are too easy for this difficulty: an instantly "
            "solvable Easy-style clue (plain primary definition, obvious "
            "fill-in-the-blank, or arithmetic giveaway) must score FRESHNESS "
            '0-9, with "too easy" stated in the feedback, unless the entry is '
            "an abbreviation, partial, or similar glue that cannot support a "
            "harder fair angle."
        )
    if puzzle_type == PuzzleType.MINI:
        return f"{base} MINI clues should remain especially concise."
    return (
        f"{base} MIDI clues can have a little more surface variety, within "
        "the target difficulty."
    )


def build_clue_evaluation_messages(
    clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
) -> tuple[str, str]:
    """Build (system, user) messages for clue quality evaluation."""
    themed = bool(theme and theme.topic)

    system_parts = [_ROLE, _RUBRIC, _difficulty_note(puzzle_type, difficulty)]
    if themed:
        system_parts.append(_THEME_EVAL_RULES)
    system_parts.append(_OUTPUT_SECTION)
    system_text = "\n\n".join(system_parts)

    revealer_answer = theme.revealer.upper() if themed else ""
    seed_answers = {s.upper() for s in theme.seed_entries} if themed else set()

    clue_lines: list[str] = []
    for clue in clues:
        key = (clue.number, clue.direction)
        crossings = crossing_words.get(key, [])
        crossing_str = ", ".join(crossings) if crossings else "none"
        tag = ""
        if clue.answer.upper() in seed_answers:
            tag = " [THEME ENTRY]"
        elif clue.answer.upper() == revealer_answer:
            tag = " [REVEALER]"
        clue_lines.append(
            f"- {clue.number}-{clue.direction.upper()}: "
            f'Answer={clue.answer}{tag}, Clue="{clue.clue}" '
            f"(crossing words: {crossing_str})"
        )
    clues_block = "\n".join(clue_lines)

    theme_context_block = ""
    if themed:
        theme_context_block = (
            "\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer}\n"
        )

    user_text = (
        f"{theme_context_block}"
        f"CLUES TO EVALUATE:\n{clues_block}\n\n"
        f"Now evaluate all {len(clues)} clues above."
    )

    return system_text, user_text


def build_clue_evaluation_prompt(
    clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
) -> str:
    """Build a single-string prompt (system+user concatenated).

    Kept for backward compatibility. Prefer
    ``build_clue_evaluation_messages`` to enable prompt caching.
    """
    system_text, user_text = build_clue_evaluation_messages(
        clues, crossing_words, puzzle_type, theme, difficulty
    )
    return f"{system_text}\n\n{user_text}"
