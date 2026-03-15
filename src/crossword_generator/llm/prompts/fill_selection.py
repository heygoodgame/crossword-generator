"""Prompt template for LLM-powered fill board selection."""

from __future__ import annotations

import json


def _render_grid(grid: list[list[str]]) -> str:
    """Render a grid as a text block."""
    return "\n".join("  " + " ".join(cell for cell in row) for row in grid)


def _extract_words(grid: list[list[str]]) -> list[str]:
    """Extract all across and down words from a grid."""
    words: list[str] = []
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Across words
    for r in range(rows):
        word = ""
        for c in range(cols):
            ch = grid[r][c]
            if ch == ".":
                if len(word) >= 2:
                    words.append(word)
                word = ""
            else:
                word += ch
        if len(word) >= 2:
            words.append(word)

    # Down words
    for c in range(cols):
        word = ""
        for r in range(rows):
            ch = grid[r][c]
            if ch == ".":
                if len(word) >= 2:
                    words.append(word)
                word = ""
            else:
                word += ch
        if len(word) >= 2:
            words.append(word)

    return words


def build_fill_selection_prompt(
    grids: list[list[list[str]]],
) -> str:
    """Build a prompt asking the LLM to select the best fill board.

    Args:
        grids: List of filled grids (each is a 2D list of letters).

    Returns:
        A prompt string ready to send to the LLM.
    """
    boards_section_parts: list[str] = []
    for i, grid in enumerate(grids, start=1):
        rendered = _render_grid(grid)
        words = _extract_words(grid)
        word_list = ", ".join(words)
        boards_section_parts.append(
            f"BOARD {i}:\n"
            f"{rendered}\n"
            f"Words: {word_list}"
        )

    boards_block = "\n\n".join(boards_section_parts)

    example_output = json.dumps(
        {"selected_board": 1, "rationale": "Board 1 has the best vocabulary..."},
    )

    return (
        "You are an expert crossword puzzle constructor choosing the best "
        "filled grid from several candidates.\n\n"
        "Evaluate each board on these criteria:\n\n"
        "1. VOCABULARY QUALITY: Common, recognizable words that solvers "
        "will know. Penalize crosswordese (obscure short words used only "
        "in crosswords), junk abbreviations, and random letter combos.\n\n"
        "2. LIVELINESS: Contemporary, interesting vocabulary. Bonus for "
        "fun letters (J, K, X, Z, Q) used in real words.\n\n"
        "3. CLUE POTENTIAL: Entries that offer multiple interesting clue "
        "angles — wordplay, misdirection, cultural references.\n\n"
        "4. AVOIDING JUNK: Penalize standalone affixes (RE-, -ED, -ING "
        "as separate entries), Roman numerals (III, VII), partial phrases, "
        "and excessive plural/tense padding (adding S or ED to pad).\n\n"
        f"BOARDS TO EVALUATE:\n\n{boards_block}\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with these fields. "
        "No other text before or after.\n"
        f"\nExample: {example_output}\n\n"
        f"Now select the best board from the {len(grids)} candidates above. "
        "Return ONLY the JSON object."
    )
