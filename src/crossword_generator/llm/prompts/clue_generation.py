"""Prompt template for LLM-powered clue generation."""

from __future__ import annotations

import json

from crossword_generator.exporters.numbering import NumberedEntry
from crossword_generator.models import PuzzleType, ThemeConcept


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
    # Build the entry list section
    entry_lines: list[str] = []
    for entry in entries:
        key = (entry.number, entry.direction)
        crossings = crossing_words.get(key, [])
        crossing_str = ", ".join(crossings) if crossings else "none"
        entry_lines.append(
            f"- {entry.number}-{entry.direction.upper()}: {entry.answer} "
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
        theme_block = f"""
THEME CONTEXT:
- Topic: {theme.topic}
- Wordplay type: {theme.wordplay_type}
- Seed entries: {', '.join(theme.seed_entries)}
- Revealer: {theme.revealer}

Theme entries should have clues that subtly echo each other or build toward
the revealer. The revealer clue should work as a standalone clue AND as the
"aha" trigger for the theme mechanic.
"""

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
