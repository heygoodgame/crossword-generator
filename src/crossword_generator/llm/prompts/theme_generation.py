"""Prompt template for LLM-powered theme generation."""

from __future__ import annotations

import json


def build_theme_generation_prompt(
    grid_size: int,
    available_slot_lengths: list[int] | None = None,
    num_seed_entries: int = 3,
    slot_counts: dict[int, int] | None = None,
    num_candidates: int | None = None,
) -> str:
    """Build a prompt that asks the LLM to generate a crossword theme concept.

    Args:
        grid_size: The grid dimension (e.g., 9 for a 9x9 grid).
        available_slot_lengths: Distinct slot lengths available in the grid.
            Used only to constrain the revealer. Seed entries are no longer
            constrained to specific slot lengths.
        num_seed_entries: How many themed seed entries to generate.
        slot_counts: Optional mapping of slot length to number of slots
            available in the grid. No longer used for constraining seeds.
        num_candidates: If set, ask for this many candidate entries instead
            of exactly num_seed_entries. Not all will be used in the grid.

    Returns:
        A prompt string ready to send to the LLM.
    """
    # When generating surplus candidates, ask for more
    effective_count = num_candidates if num_candidates else num_seed_entries

    # Revealer is still constrained to available slot lengths
    revealer_max = grid_size
    if available_slot_lengths:
        revealer_max = max(available_slot_lengths)

    # Build a dynamic example
    example_entries, example_revealer = _pick_example_entries(grid_size)
    example_output = json.dumps(
        {
            "topic": "Things that fly",
            "wordplay_type": "literal",
            "seed_entries": example_entries,
            "revealer": example_revealer,
            "revealer_clue": (
                "Up in the sky, or what the theme entries "
                "have in common"
            ),
        },
        indent=2,
    )

    role = (
        "You are an expert crossword puzzle constructor designing a theme "
        "for a themed crossword puzzle."
    )

    length_constraint = (
        f"LENGTH CONSTRAINTS:\n"
        f"- The revealer must be at most {revealer_max} letters long.\n"
        f"- Seed entries should be between 3 and {grid_size} letters long.\n"
        f"- Seed entries should generally be shorter than the revealer.\n"
        f"- Vary the lengths of seed entries — don't make them all the "
        f"same length."
    )

    guidelines = (
        "GUIDELINES:\n"
        "- Choose an accessible, fun theme topic that a broad "
        "audience will enjoy.\n"
        "- Theme entries should be common words or well-known "
        "phrases (no obscure vocabulary).\n"
        "- Each seed entry must be a SINGLE WORD (no spaces, "
        "no hyphens) that fits in the grid.\n"
        "- The revealer must also be a single word.\n"
        "- Wordplay types include: literal (entries share a "
        "theme), hidden word, homophones, double meanings, "
        "category members.\n"
        "- The revealer should work as both a standalone crossword "
        "entry AND as the 'aha moment' for the theme."
    )

    surplus_note = ""
    if num_candidates and num_candidates > num_seed_entries:
        surplus_note = (
            f" Generate {effective_count} seed entries (not all will "
            f"be used — we'll select the best {num_seed_entries} "
            f"for the grid)."
        )

    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with these fields. "
        "No other text before or after.\n"
        f"\nExample:\n{example_output}\n"
        f"\nGenerate a theme with exactly {effective_count} "
        f"seed entries for a {grid_size}x{grid_size} crossword "
        f"grid.{surplus_note} Return ONLY the JSON object, "
        f"no explanations."
    )

    return (
        f"{role}\n\n{length_constraint}\n\n"
        f"{guidelines}\n\n{output_section}"
    )


# Example words by length for building dynamic examples
_EXAMPLES_BY_LENGTH: dict[int, list[str]] = {
    3: ["OWL", "BAT", "FLY", "BEE"],
    4: ["KITE", "BIRD", "SOAR", "WING"],
    5: ["EAGLE", "CRANE", "ROBIN", "RAVEN"],
    6: ["FALCON", "FLIGHT", "PIGEON", "PARROT"],
    7: ["SPARROW", "BUZZARD", "PELICAN", "OSTRICH"],
    8: ["BLUEBIRD", "CARDINAL", "FLAMINGO"],
    9: ["ALBATROSS", "NIGHTHAWK", "BLACKBIRD"],
}


def _pick_example_entries(
    grid_size: int,
) -> tuple[list[str], str]:
    """Pick example seed entries and revealer for the prompt.

    Returns words that demonstrate length variation in the example.
    """
    # Pick seed entries with varied lengths
    entries: list[str] = []
    target_lengths = [3, 5, 4]  # varied lengths for the example
    for length in target_lengths:
        if length <= grid_size and length in _EXAMPLES_BY_LENGTH:
            words = _EXAMPLES_BY_LENGTH[length]
            word = words[len(entries) % len(words)]
            if word not in entries:
                entries.append(word)

    # Fill remaining if needed
    while len(entries) < 3:
        for length in sorted(_EXAMPLES_BY_LENGTH.keys()):
            if length <= grid_size:
                words = _EXAMPLES_BY_LENGTH[length]
                for word in words:
                    if word not in entries:
                        entries.append(word)
                        break
            if len(entries) >= 3:
                break
        else:
            break

    # Pick revealer — prefer longer word
    revealer = "SOAR"  # fallback
    for length in sorted(_EXAMPLES_BY_LENGTH.keys(), reverse=True):
        if length <= grid_size and length in _EXAMPLES_BY_LENGTH:
            for word in _EXAMPLES_BY_LENGTH[length]:
                if word not in entries:
                    revealer = word
                    break
            break

    return entries, revealer
