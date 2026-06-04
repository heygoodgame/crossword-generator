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
    "You are an expert crossword puzzle constructor "
    "writing clues for a completed grid."
)

_GUIDELINES = (
    "GUIDELINES:\n"
    "- Write one clue per entry. Every clue must have "
    "exactly one defensible answer.\n"
    "- DO NOT use the answer word, any answer word-part, or any related "
    "morphological variant/root in the clue. This includes singular/plural "
    "forms, verb forms, compounds, and famous-title fill-in-the-blanks that "
    "would point at a different form of the answer. For example, do not clue "
    "HOUSEWIFE with \"Desperate ___wives\"; do not clue WIFE with \"wives\" "
    "or \"wifely\"; do not clue TEACHER with \"teaches\" or \"teaching\"; "
    "do not clue POL with \"politician\" or \"political\".\n"
    "- DO NOT use any of an entry's crossing words "
    "in that entry's clue.\n"
    "- Accuracy is more important than cleverness, freshness, or difficulty. "
    "For every clue, verify the fact, grammar, number, tense, part of speech, "
    "and exact phrase match. If you are not certain a proper noun, song, "
    "quote, sports team name, idiom, or pop-culture reference is correct, "
    "use a plain dictionary-style clue instead.\n"
    "- Fill-in-the-blank clues must fit the answer exactly, including "
    "singular/plural, tense, spacing, and contractions. Do not clue SAINT "
    "with the plural Saints, BATTING with \"At ___\", or GOT A SAY with "
    "\"gotta say\" / \"my two cents\" phrasing.\n"
    "- Vary clue styles within the target difficulty: definitional, "
    "fill-in-the-blank, wordplay, trivia, and lateral thinking are all "
    "available, but Easy clues should stay direct and obvious.\n"
    "- Do not repeat the exact same clue wording for the same answer across "
    "different puzzles. If prior clues are listed for an answer, write a new "
    "clue that does not exactly duplicate any of them.\n"
    "- Use misdirection and cleverness only when they fit the target "
    "difficulty. For Easy clues, clarity beats cleverness.\n"
    '- Use question marks for witty/punny clues only when the target '
    'difficulty allows it (e.g., "Plant manager?" for GARDENER).\n'
    "- Keep clues concise — say exactly what's needed, "
    "no filler words.\n"
    "- Avoid obscure trivia that solvers can't "
    "reason toward.\n"
    "- Avoid unpleasant clue wording. Do not use terms like \"death\" or "
    "\"undocumented immigrant\" in clues. If a clue must refer to dying, use "
    "a gentle euphemism like \"passed on\" rather than blunt wording.\n"
    "- Make clues culturally accessible and "
    "contemporary where possible.\n"
    "- Do not add word-count tags like \"(two words)\" or \"(three words)\". "
    "Use no word-count enumeration unless the pipeline provides explicit "
    "word-boundary metadata.\n"
    "- Put explanatory tags in parentheses, not after a comma or colon. "
    "Use \"To the ___ (in the extreme)\" rather than "
    "\"To the ___: in the extreme\"; use "
    "\"Dennis ___ (pop art icon of soup cans)\" rather than "
    "\"Dennis ___, pop art icon of soup cans\".\n"
    "- Avoid body-part, underwear, anatomy, sexuality, appearance, "
    "identity, or demographic-based jokes. If an answer is sensitive "
    "but acceptable, clue it neutrally rather than with a wink, "
    "innuendo, or objectifying angle.\n"
    "- Always write clues with a non-offensive mindset. When an answer "
    "has a violent, weapon-, drug-, or otherwise negative literal "
    "meaning, prefer a clue built on a benign alternate sense, idiom, "
    "or pop-culture usage. For example, clue BOMB as \"Flop at the box "
    "office\" rather than referencing an explosive device, and clue "
    "SHOOT as \"Rats!\" rather than referencing gunfire."
)

_THEME_INSTRUCTIONS = (
    "THEME CLUE INSTRUCTIONS:\n"
    "For each [THEME ENTRY], choose ONE of these three styles — "
    "and VARY the style across theme entries:\n"
    "\n"
    "1. STANDALONE: A clean clue with no reference to the theme.\n"
    "   Example (theme: CAESAR reveals Roman words): "
    "TOGA → \"Garment at a frat party\"\n"
    "\n"
    "2. INDIRECT ALLUSION: A standalone clue that subtly nods to\n"
    "   the theme concept without being heavy-handed.\n"
    "   Example: TOGA → \"What a senator might have worn "
    "to the forum\"\n"
    "\n"
    "3. POSITIONAL CROSS-REFERENCE: Reference the revealer by\n"
    "   number. Use sparingly — at most ONE theme entry should\n"
    "   use this style. The cross-reference must read naturally\n"
    "   as a standalone, solvable clue — never a bare\n"
    "   \"See X-Across\" with no solving context.\n"
    "   Example: TOGA → \"Garb of the revealer's era\"\n"
    "\n"
    "IMPORTANT:\n"
    "- Vary the style across theme entries. Do NOT use the same "
    "approach for all of them.\n"
    "- Do NOT write \"one of [REVEALER ANSWER]\" — this is "
    "grammatically unnatural for most theme types.\n"
    "- NEVER write a bare \"See X-Across\" or \"See X-Down\" "
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
    "4. Never use the phrase \"theme entries.\"\n"
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


def _difficulty_guidance(
    puzzle_type: PuzzleType, difficulty: PuzzleDifficulty
) -> str:
    if difficulty == PuzzleDifficulty.EASY:
        base = (
            "This is an HGG Easy crossword for a very broad casual audience. "
            "Write clues easier than an NYT Monday. Default to direct "
            "definitions, familiar everyday meanings, and totally obvious "
            "fill-in-the-blank clues. Avoid oblique definitions, tricky "
            "wordplay, niche trivia, and lateral-thinking clues. If choosing "
            "between clever and instantly solvable, choose instantly solvable."
        )
    else:
        base = (
            "This is an HGG Hard crossword. Aim for NYT Tuesday/Wednesday "
            "difficulty: still fair and broadly accessible, but a notch more "
            "challenging. Prefer clean, accurate clues with modest difficulty "
            "over showy wordplay. You may use less direct definitions, common "
            "secondary meanings, mild misdirection, and occasional question-mark "
            "clues, but only when the clue remains factually airtight and "
            "reasonably inferable. Avoid Saturday-level obscurity and trivia "
            "that solvers cannot reason toward. Do not force difficulty with "
            "strained pop-culture references, ultra-current slang, or clues "
            "that only work after a long explanation. If the clever angle feels "
            "debatable, use a cleaner direct clue."
        )
    if puzzle_type == PuzzleType.MINI:
        return (
            f"{base} Because this is a MINI crossword, keep clues especially "
            "concise."
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
                "\nPRIOR CLUES FOR THESE ANSWERS:\n"
                "Do not repeat any clue exactly for the same answer.\n"
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
                f"- Revealer clue draft: \"{theme.revealer_clue}\" "
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
    "forms, verb forms, compounds, and famous-title fill-in-the-blanks that "
    "would point at a different form of the answer. For example, do not clue "
    "HOUSEWIFE with \"Desperate ___wives\"; do not clue POL with "
    "\"politician\" or \"political\".\n"
    "- Accuracy is more important than cleverness. Verify facts, grammar, "
    "number, tense, part of speech, and exact phrase match. If the old clue "
    "used a questionable proper noun, song, quote, sports team name, idiom, "
    "or pop-culture reference, replace it with a plain accurate clue.\n"
    "- Fill-in-the-blank clues must fit the answer exactly, including "
    "singular/plural, tense, spacing, and contractions.\n"
    "- DO NOT use any crossing words in the clue.\n"
    "- DO NOT duplicate phrasing from the existing clues listed below.\n"
    "- Do not add word-count tags like \"(two words)\" or \"(three words)\".\n"
    "- Put explanatory tags in parentheses, not after a comma or colon.\n"
    "- Avoid unpleasant clue wording such as \"death\" or "
    "\"undocumented immigrant\"; use gentle wording like \"passed on\" only "
    "if dying is unavoidable.\n"
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
                revealer_label = (
                    f"{clue.number}-{clue.direction.capitalize()}"
                )
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
            f"  Old clue: \"{clue.clue}\"\n"
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
                f"{clue.answer} = \"{clue.clue}\""
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
    "\"See X-Across\" with no solving context)\n"
    "\n"
    "Do NOT write \"one of [REVEALER ANSWER]\" — this is grammatically "
    "unnatural. Vary the style if multiple theme entries need repair.\n"
    "NEVER write a bare \"See X-Across\" or \"See X-Down\" with no "
    "standalone definition.\n"
    "\n"
    "If the [REVEALER] entry needs repair:\n"
    "1. Identify the connecting element — the revealer answer itself may "
    "not be the connecting word; a component of it may be (e.g., GOLD in "
    "GOLDMINES). Make sure your hint is factually accurate about which "
    "word or word-part connects to the theme entries.\n"
    "2. Write a standalone definition first, then optionally add a theme "
    "hint using natural phrasing. Never use \"theme entries.\""
)
