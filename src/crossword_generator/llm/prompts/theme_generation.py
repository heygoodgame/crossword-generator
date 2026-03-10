"""Prompt template for LLM-powered theme generation."""

from __future__ import annotations

import json


def build_theme_generation_prompt(
    grid_size: int,
    available_slot_lengths: list[int],
    num_seed_entries: int = 3,
) -> str:
    """Build a prompt that asks the LLM to generate a crossword theme concept.

    Args:
        grid_size: The grid dimension (e.g., 9 for a 9x9 grid).
        available_slot_lengths: Distinct slot lengths available in the grid.
        num_seed_entries: How many themed seed entries to generate.

    Returns:
        A prompt string ready to send to the LLM.
    """
    slot_lengths_str = ", ".join(str(n) for n in sorted(available_slot_lengths))

    example_output = json.dumps(
        {
            "topic": "Things that fly",
            "wordplay_type": "literal",
            "seed_entries": ["EAGLE", "KITE", "ARROW"],
            "revealer": "AIRBORNE",
            "revealer_clue": "Up in the sky, or what the theme entries have in common",
        },
        indent=2,
    )

    role = (
        "You are an expert crossword puzzle constructor designing a theme "
        "for a themed crossword puzzle."
    )

    guidelines = (
        "GUIDELINES:\n"
        "- Choose an accessible, fun theme topic that a broad audience will enjoy.\n"
        "- Theme entries should be common words or well-known phrases "
        "(no obscure vocabulary).\n"
        "- Each seed entry must be a SINGLE WORD (no spaces, no hyphens) "
        "that fits in the grid.\n"
        "- The revealer must also be a single word.\n"
        "- Vary the lengths of seed entries — don't make them all the same length.\n"
        "- All words must be between 3 and {grid_size} letters long.\n"
        f"- Available slot lengths in the grid: {slot_lengths_str}.\n"
        "- Each entry length MUST match one of the available slot lengths.\n"
        "- Wordplay types include: literal (entries share a theme), "
        "hidden word, homophones, double meanings, category members.\n"
        "- The revealer should work as both a standalone crossword entry "
        "AND as the 'aha moment' for the theme."
    ).format(grid_size=grid_size)

    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with these fields. "
        "No other text before or after.\n"
        f"\nExample:\n{example_output}\n"
        f"\nGenerate a theme with exactly {num_seed_entries} seed entries "
        f"for a {grid_size}x{grid_size} crossword grid. "
        "Return ONLY the JSON object."
    )

    return f"{role}\n\n{guidelines}\n\n{output_section}"
