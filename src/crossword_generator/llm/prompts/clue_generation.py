"""Prompt template for LLM-powered clue generation."""

from __future__ import annotations

import json

from crossword_generator.exporters.numbering import NumberedEntry
from crossword_generator.models import ClueEntry, ClueGrade, PuzzleType, ThemeConcept


def build_clue_generation_prompt(
    entries: list[NumberedEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
) -> str:
    """Build a prompt that asks the LLM to generate clues for all entries.

    Args:
        entries: Numbered entries with answers from the filled grid.
        crossing_words: Maps (number, direction) to the list of answer words
            that physically cross this entry in the grid.
        puzzle_type: Mini or midi — affects difficulty guidance.
        theme: Optional theme concept for midi puzzles.

    Returns:
        A prompt string ready to send to the LLM.
    """
    # Identify theme entries and revealer for annotation
    revealer_info: tuple[int, str] | None = None
    seed_answers: set[str] = set()
    if theme and theme.topic:
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

    # Difficulty guidance based on puzzle type
    if puzzle_type == PuzzleType.MINI:
        difficulty_guidance = (
            "This is a MINI crossword (like NYT Mini). Clues should be "
            "accessible and Monday-level: straightforward but not boring. "
            "Favor clean, clever clues over tricky ones."
        )
    else:
        difficulty_guidance = (
            "This is a MIDI crossword. Clues can be trickier — "
            "aim for Tuesday/Wednesday difficulty. Use more wordplay, "
            "misdirection, and varied clue styles."
        )

    # Theme section
    theme_block = ""
    if theme and theme.topic:
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
        theme_block = (
            f"\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer} ({rev_label})\n"
            f"{revealer_clue_draft}"
            f"- Theme entries are marked [THEME ENTRY] above\n"
            f"\nTHEME CLUE INSTRUCTIONS:\n"
            f"For each [THEME ENTRY], choose ONE of these three styles — "
            f"and VARY the style across theme entries:\n"
            f"\n"
            f"1. STANDALONE: A clean clue with no reference to the theme.\n"
            f"   Example (theme: CAESAR reveals Roman words): "
            f"TOGA → \"Garment at a frat party\"\n"
            f"\n"
            f"2. INDIRECT ALLUSION: A standalone clue that subtly nods to\n"
            f"   the theme concept without being heavy-handed.\n"
            f"   Example: TOGA → \"What a senator might have worn "
            f"to the forum\"\n"
            f"\n"
            f"3. POSITIONAL CROSS-REFERENCE: Reference the revealer by\n"
            f"   number. Use sparingly — at most ONE theme entry should\n"
            f"   use this style. The cross-reference must read naturally\n"
            f"   as a standalone, solvable clue — never a bare\n"
            f"   \"See X-Across\" with no solving context.\n"
            f"   Example: TOGA → \"Garb of {rev_label}'s era\"\n"
            f"\n"
            f"IMPORTANT:\n"
            f"- Vary the style across theme entries. Do NOT use the same "
            f"approach for all of them.\n"
            f"- Do NOT write \"one of [REVEALER ANSWER]\" — this is "
            f"grammatically unnatural for most theme types.\n"
            f"- NEVER write a bare \"See X-Across\" or \"See X-Down\" "
            f"with no standalone definition. Every clue must give the "
            f"solver something to work with on its own.\n"
            f"\n"
            f"For the [REVEALER] entry:\n"
            f"1. First identify the connecting element between the "
            f"revealer and the theme entries. The revealer answer itself "
            f"may not be the connecting word — a component of it may be "
            f"(e.g., for GOLDMINES with theme entries BAR/COIN/ORE, the "
            f"connecting element is GOLD, not GOLDMINES).\n"
            f"2. Write a standalone definition of the revealer answer "
            f"first.\n"
            f"3. Then optionally add a theme hint that accurately "
            f"describes the connection. Make sure your hint is factually "
            f"accurate about which word or word-part connects to the "
            f"theme entries.\n"
            f"4. Never use the phrase \"theme entries.\"\n"
        )

    # Build the JSON format example
    example_output = json.dumps(
        [
            {"number": 1, "direction": "across", "clue": "Example clue text"},
            {"number": 1, "direction": "down", "clue": "Example clue text"},
        ],
        indent=2,
    )

    role = (
        "You are an expert crossword puzzle constructor "
        "writing clues for a completed grid."
    )
    guidelines = (
        "GUIDELINES:\n"
        "- Write one clue per entry. Every clue must have "
        "exactly one defensible answer.\n"
        "- DO NOT use the answer word (or any close "
        "variant/root) in the clue.\n"
        "- DO NOT use any of an entry's crossing words "
        "in that entry's clue.\n"
        "- Vary clue styles: mix definitional, wordplay, "
        "trivia, fill-in-the-blank, and lateral thinking.\n"
        "- Prefer misdirection and cleverness over "
        "dictionary definitions.\n"
        '- Use question marks for witty/punny clues '
        '(e.g., "Plant manager?" for GARDENER).\n'
        "- Keep clues concise — say exactly what's needed, "
        "no filler words.\n"
        "- Avoid obscure trivia that solvers can't "
        "reason toward.\n"
        "- Make clues culturally accessible and "
        "contemporary where possible."
    )
    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON array with one object per entry. "
        "No other text before or after.\n"
        f"\n{example_output}\n"
        f"\nNow write clues for all {len(entries)} entries "
        "listed above. Return ONLY the JSON array."
    )

    return (
        f"{role}\n\n{difficulty_guidance}\n{theme_block}\n"
        f"ENTRIES TO CLUE:\n{entries_block}\n\n"
        f"{guidelines}\n\n{output_section}"
    )


def build_clue_repair_prompt(
    entries_to_repair: list[tuple[ClueEntry, ClueGrade]],
    all_clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
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
    # Identify theme entries and revealer for annotation
    revealer_answer = ""
    seed_answers: set[str] = set()
    revealer_label = ""
    if theme and theme.topic:
        revealer_answer = theme.revealer.upper()
        seed_answers = {s.upper() for s in theme.seed_entries}
        # Find revealer position from all_clues
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

    # Difficulty guidance
    if puzzle_type == PuzzleType.MINI:
        difficulty_guidance = (
            "This is a MINI crossword (Monday-level). "
            "Clues should be accessible and straightforward."
        )
    else:
        difficulty_guidance = (
            "This is a MIDI crossword (Tuesday/Wednesday-level). "
            "Clues can use more wordplay and misdirection."
        )

    # Theme section
    theme_block = ""
    if theme and theme.topic:
        theme_block = (
            f"\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer} ({revealer_label})\n"
            f"\nIf any entries above are marked [THEME ENTRY], their "
            f"replacement clues can use any of these styles:\n"
            f"1. Standalone clue (no theme reference)\n"
            f"2. Indirect allusion to the theme concept\n"
            f"3. Cross-reference to {revealer_label} (use sparingly — "
            f"must read naturally as a standalone, solvable clue, never "
            f"a bare \"See X-Across\" with no solving context)\n"
            f"\nDo NOT write \"one of [REVEALER ANSWER]\" — this is "
            f"grammatically unnatural. Vary the style if multiple theme "
            f"entries need repair.\n"
            f"NEVER write a bare \"See X-Across\" or \"See X-Down\" "
            f"with no standalone definition.\n"
            f"\nIf the [REVEALER] entry needs repair:\n"
            f"1. Identify the connecting element — the revealer answer "
            f"itself may not be the connecting word; a component of it "
            f"may be (e.g., GOLD in GOLDMINES). Make sure your hint "
            f"is factually accurate about which word or word-part "
            f"connects to the theme entries.\n"
            f"2. Write a standalone definition first, then optionally "
            f"add a theme hint using natural phrasing. Never use "
            f"\"theme entries.\"\n"
        )

    # JSON format example
    example_output = json.dumps(
        [
            {"number": 1, "direction": "across", "clue": "Replacement clue text"},
        ],
        indent=2,
    )

    role = (
        "You are an expert crossword puzzle constructor. "
        "The following clues had ACCURACY problems — they were factually wrong, "
        "had multiple defensible answers, or had grammar/part-of-speech mismatches. "
        "Write replacement clues that are factually correct with exactly one "
        "defensible answer."
    )

    guidelines = (
        "GUIDELINES:\n"
        "- Each replacement clue must have exactly one defensible answer.\n"
        "- DO NOT use the answer word (or any close variant/root) in the clue.\n"
        "- DO NOT use any crossing words in the clue.\n"
        "- DO NOT duplicate phrasing from the existing clues listed below.\n"
        "- Keep clues concise and culturally accessible."
    )

    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON array with one object per repaired clue. "
        "No other text before or after.\n"
        f"\n{example_output}\n"
        f"\nNow write replacement clues for the "
        f"{len(entries_to_repair)} entries above. "
        "Return ONLY the JSON array."
    )

    return (
        f"{role}\n\n{difficulty_guidance}\n{theme_block}\n"
        f"CLUES TO REPAIR:\n{repair_block}\n\n"
        f"EXISTING CLUES (for context — do not duplicate):\n{context_block}\n\n"
        f"{guidelines}\n\n{output_section}"
    )
