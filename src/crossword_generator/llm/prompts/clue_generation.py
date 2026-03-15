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
        theme_block = (
            f"\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer} ({rev_label})\n"
            f"- Theme entries are marked [THEME ENTRY] above\n"
            f"\nTHEME CLUE INSTRUCTIONS:\n"
            f"Each [THEME ENTRY] clue must tie back to the revealer. "
            f"Two good approaches:\n"
            f"1. Direct cross-reference: mention the revealer by number\n"
            f'   (e.g., "Title for many of {rev_label}\'s assassins" '
            f"for HITMAN)\n"
            f"2. Thematic angle: a standard clue that also nods to the\n"
            f"   revealer's concept (e.g., for theme \"things that are "
            f'golden,"\n'
            f'   clue GATE as "Entrance to a park, or a gilded one '
            f'in San Francisco")\n'
            f"\nMix both styles across the theme entries — "
            f"don't use the same approach for all of them.\n"
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
            f"replacement clues must tie back to the revealer — either "
            f"by referencing {revealer_label} directly or by connecting "
            f"to the theme concept.\n"
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
