"""Prompt template for LLM-powered puzzle title generation."""

from __future__ import annotations

import json

from crossword_generator.models import ClueEntry, PuzzleType, ThemeConcept

_ROLE = (
    "You are a crossword puzzle editor choosing a title for a "
    "published puzzle — like Will Shortz naming a New York Times puzzle."
)

_GUIDELINES = (
    "GUIDELINES:\n"
    "- The title must be 1-5 words\n"
    "- Be evocative and indirect, not literal\n"
    "- The title must NOT contain any answer word from the grid\n"
    "- Weight 1-Across heavily as the marquee entry — the title "
    "should resonate with it especially\n"
    "- For themed puzzles: hint at the theme concept without "
    "giving it away (1-Across still gets extra weight within the "
    "theme)\n"
    "- For themeless puzzles: something catchy inspired by the "
    "standout fill, anchored by 1-Across\n"
    "\nEXAMPLES OF GOOD TITLES (with reasoning):\n"
    '- Theme "things that are golden" → "Midas Touch" '
    "(everything Midas touched turned to gold — oblique nod to the "
    "theme)\n"
    '- Theme "types of bridges" → "Crossing Over" '
    "(double meaning: bridges literally cross over, and the phrase "
    "evokes transition without naming bridges)\n"
    '- Themeless with JAZZ, ROBOT, PIXEL → "Digital Riffs" '
    "(captures the modern/tech-meets-music vibe of the standout "
    "fill)"
)

_EXAMPLE_OUTPUT = json.dumps(
    {
        "why": (
            "The theme is things that are golden — 'Au' is the chemical "
            "symbol for gold, so 'Au Naturel' winks at the theme without "
            "naming it."
        ),
        "title": "Au Naturel",
    },
    indent=2,
)

_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a JSON object with two keys, in this order:\n"
    "- \"why\": one or two sentences explaining the reasoning behind "
    "the title (what it ties to, why it works for this puzzle). "
    "Think this through FIRST so the title is well-grounded.\n"
    "- \"title\": the title itself, following the guidelines above.\n"
    "No other text before or after.\n"
    f"\n{_EXAMPLE_OUTPUT}\n"
    "\nNow generate a title. Return ONLY the JSON object."
)


def build_puzzle_naming_messages(
    puzzle_type: PuzzleType,
    grid_size: int,
    clues: list[ClueEntry],
    grid: list[list[str]],
    theme: ThemeConcept | None = None,
) -> tuple[str, str]:
    """Build (system, user) messages for puzzle title generation.

    The system text holds the editor role, guidelines, and example
    output — identical across puzzles, so it caches well.
    """
    system_text = "\n\n".join([_ROLE, _GUIDELINES, _OUTPUT_SECTION])

    # Collect all answer words
    answers = [c.answer for c in clues]
    answers_block = ", ".join(answers)

    # 1-Across is traditionally the puzzle's "headliner" entry — the
    # constructor's marquee word. Surface it so the title-generating LLM
    # gives it extra weight when picking inspiration.
    one_across = next(
        (
            c
            for c in clues
            if c.number == 1 and c.direction.lower() == "across"
        ),
        None,
    )
    one_across_block = ""
    if one_across is not None:
        one_across_block = (
            "\nMARQUEE ENTRY (1-ACROSS):\n"
            f'- {one_across.answer} — "{one_across.clue}"\n'
            "1-Across is the puzzle's headliner — the entry crossword "
            "constructors traditionally pick to set the tone. Give it "
            "extra weight when choosing the title's inspiration: the "
            "title should resonate with 1-Across in particular, "
            "whether by sharing its vibe, riffing on its meaning, or "
            "picking up its imagery — without ever using the answer "
            "word itself.\n"
            f"CRITICAL: the title must NOT contain the word "
            f'"{one_across.answer}" anywhere — not as a whole word, not '
            f"as a substring, not in any form. Reject any title that "
            f'includes "{one_across.answer}" and pick a different '
            "phrasing. The same prohibition applies to every other "
            "answer in the grid.\n"
        )

    # Theme context
    if theme and theme.topic:
        seed_list = ", ".join(theme.seed_entries) if theme.seed_entries else "none"
        theme_block = (
            "\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer}\n"
            f"- Theme entries: {seed_list}\n"
            "\nThe title should tie to the theme concept OBLIQUELY — "
            "hint at it without stating it directly. "
            "Think of how a crossword editor would name a themed puzzle: "
            "evocative, indirect, clever.\n"
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

    user_text = (
        f"PUZZLE: {puzzle_type.value.title()} crossword ({grid_size}x{grid_size})\n"
        f"{theme_block}"
        f"{one_across_block}\n"
        f"FILL WORDS: {answers_block}\n\n"
        f"SAMPLE CLUES:\n{clue_block}"
    )

    return system_text, user_text


def build_puzzle_naming_prompt(
    puzzle_type: PuzzleType,
    grid_size: int,
    clues: list[ClueEntry],
    grid: list[list[str]],
    theme: ThemeConcept | None = None,
) -> str:
    """Build a single-string prompt (system+user concatenated).

    Kept for backward compatibility. Prefer
    ``build_puzzle_naming_messages`` to enable prompt caching.
    """
    system_text, user_text = build_puzzle_naming_messages(
        puzzle_type, grid_size, clues, grid, theme
    )
    return f"{system_text}\n\n{user_text}"
