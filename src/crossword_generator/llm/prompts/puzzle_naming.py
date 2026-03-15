"""Prompt template for LLM-powered puzzle title generation."""

from __future__ import annotations

import json

from crossword_generator.models import ClueEntry, PuzzleType, ThemeConcept


def build_puzzle_naming_prompt(
    puzzle_type: PuzzleType,
    grid_size: int,
    clues: list[ClueEntry],
    grid: list[list[str]],
    theme: ThemeConcept | None = None,
) -> str:
    """Build a prompt asking the LLM to generate a creative puzzle title.

    Args:
        puzzle_type: Mini or midi.
        grid_size: Grid dimension (e.g. 5, 7, 9).
        clues: All clues in the puzzle.
        grid: The filled grid.
        theme: Optional theme concept for themed puzzles.

    Returns:
        A prompt string ready to send to the LLM.
    """
    # Collect all answer words
    answers = [c.answer for c in clues]
    answers_block = ", ".join(answers)

    # Theme context
    theme_block = ""
    if theme and theme.topic:
        seed_list = ", ".join(theme.seed_entries) if theme.seed_entries else "none"
        theme_block = (
            f"\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer}\n"
            f"- Theme entries: {seed_list}\n"
            f"\nThe title should tie to the theme concept OBLIQUELY — "
            f"hint at it without stating it directly. "
            f"Think of how a crossword editor would name a themed puzzle: "
            f"evocative, indirect, clever.\n"
        )
    else:
        theme_block = (
            "\nThis is a THEMELESS puzzle. Pick up on interesting or "
            "standout fill entries for inspiration. The title should be "
            "playful and evocative.\n"
        )

    # Clue sample for context
    clue_lines = []
    for c in clues[:10]:
        clue_lines.append(
            f"- {c.number}-{c.direction.upper()}: "
            f"{c.answer} = \"{c.clue}\""
        )
    clue_block = "\n".join(clue_lines)
    if len(clues) > 10:
        clue_block += f"\n... and {len(clues) - 10} more entries"

    example_output = json.dumps({"title": "Au Naturel"}, indent=2)

    role = (
        "You are a crossword puzzle editor choosing a title for a "
        "published puzzle — like Will Shortz naming a New York Times puzzle."
    )

    guidelines = (
        "GUIDELINES:\n"
        "- The title must be 1-5 words\n"
        "- Be evocative and indirect, not literal\n"
        "- The title must NOT contain any answer word from the grid\n"
        "- For themed puzzles: hint at the theme concept without "
        "giving it away\n"
        "- For themeless puzzles: something catchy inspired by the "
        "standout fill\n"
        "\nEXAMPLES OF GOOD TITLES:\n"
        '- For a puzzle themed around "things that are golden": '
        '"Midas Touch" or "Au Naturel"\n'
        '- For a puzzle themed around "types of bridges": '
        '"Crossing Over" or "Span Class"\n'
        '- For a themeless with JAZZ, ROBOT, PIXEL: '
        '"Digital Riffs" or "Byte-Sized"\n'
    )

    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with a single key. "
        "No other text before or after.\n"
        f"\n{example_output}\n"
        "\nNow generate a title. Return ONLY the JSON object."
    )

    return (
        f"{role}\n\n"
        f"PUZZLE: {puzzle_type.value.title()} crossword ({grid_size}x{grid_size})\n"
        f"{theme_block}\n"
        f"FILL WORDS: {answers_block}\n\n"
        f"SAMPLE CLUES:\n{clue_block}\n\n"
        f"{guidelines}\n\n{output_section}"
    )
