"""Prompt template for LLM-powered theme generation."""

from __future__ import annotations

import json


def build_theme_generation_prompt(
    grid_size: int,
    available_slot_lengths: list[int],
    num_seed_entries: int = 3,
    slot_counts: dict[int, int] | None = None,
) -> str:
    """Build a prompt that asks the LLM to generate a crossword theme concept.

    Args:
        grid_size: The grid dimension (e.g., 9 for a 9x9 grid).
        available_slot_lengths: Distinct slot lengths available in the grid.
        num_seed_entries: How many themed seed entries to generate.
        slot_counts: Optional mapping of slot length to number of slots
            available in the grid (e.g., {3: 16, 5: 4, 9: 6}).

    Returns:
        A prompt string ready to send to the LLM.
    """
    sorted_lengths = sorted(available_slot_lengths)
    slot_lengths_str = ", ".join(str(n) for n in sorted_lengths)

    # Build a dynamic example that uses the actual available lengths
    example_entries, example_revealer = _pick_example_entries(
        sorted_lengths
    )
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

    # Add letter-count verification to the example
    entry_counts = ", ".join(
        f"{w} ({len(w)} letters)" for w in example_entries
    )
    revealer_count = f"{example_revealer} ({len(example_revealer)} letters)"

    role = (
        "You are an expert crossword puzzle constructor designing a theme "
        "for a themed crossword puzzle."
    )

    slot_availability = ""
    if slot_counts:
        counts_str = ", ".join(
            f"{length}-letter: {count}"
            for length, count in sorted(slot_counts.items())
        )
        slot_availability = (
            f"\nAvailable slots by length: {counts_str}.\n"
            f"Do not use more theme words of a given length than "
            f"there are slots."
        )

    length_constraint = (
        f"CRITICAL LENGTH CONSTRAINT:\n"
        f"The ONLY allowed word lengths are: {slot_lengths_str}.\n"
        f"Every seed entry and the revealer MUST be exactly one of "
        f"these lengths. Words of any other length will be rejected. "
        f"Count the letters carefully before including a word."
        f"{slot_availability}"
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
        "- Vary the lengths of seed entries — don't make them "
        "all the same length.\n"
        f"- Allowed word lengths: {slot_lengths_str} "
        "(NO other lengths).\n"
        "- Wordplay types include: literal (entries share a "
        "theme), hidden word, homophones, double meanings, "
        "category members.\n"
        "- The revealer should work as both a standalone crossword "
        "entry AND as the 'aha moment' for the theme."
    )

    verification = (
        "VERIFICATION STEP:\n"
        "Before responding, count the letters in EVERY word. "
        "For example, the words above have these counts: "
        f"{entry_counts}, {revealer_count}. "
        f"If ANY word is not exactly {slot_lengths_str} letters, "
        "replace it with a word that is."
    )

    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with these fields. "
        "No other text before or after.\n"
        f"\nExample (note: all words are {slot_lengths_str} "
        f"letters long):\n{example_output}\n"
        f"\n{verification}\n"
        f"\nGenerate a theme with exactly {num_seed_entries} "
        f"seed entries for a {grid_size}x{grid_size} crossword "
        f"grid. Every word MUST be exactly {slot_lengths_str} "
        f"letters long. Return ONLY the JSON object, "
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
    available_lengths: list[int],
) -> tuple[list[str], str]:
    """Pick example seed entries and revealer matching available lengths.

    Returns words that demonstrate the length constraint in the example.
    """
    # Pick seed entries from different available lengths
    entries: list[str] = []
    used_lengths: set[int] = set()
    for length in available_lengths:
        if length in _EXAMPLES_BY_LENGTH and len(entries) < 3:
            words = _EXAMPLES_BY_LENGTH[length]
            word = words[len(entries) % len(words)]
            entries.append(word)
            used_lengths.add(length)

    # Fill remaining entries if needed
    while len(entries) < 3:
        for length in available_lengths:
            if length in _EXAMPLES_BY_LENGTH:
                words = _EXAMPLES_BY_LENGTH[length]
                word = words[len(entries) % len(words)]
                if word not in entries:
                    entries.append(word)
                    break
        else:
            break

    # Pick revealer from a different length than entries
    revealer = "SOAR"  # fallback
    for length in reversed(available_lengths):
        if length in _EXAMPLES_BY_LENGTH:
            for word in _EXAMPLES_BY_LENGTH[length]:
                if word not in entries:
                    revealer = word
                    break
            break

    return entries, revealer
