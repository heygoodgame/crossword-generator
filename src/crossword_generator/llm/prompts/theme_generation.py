"""Prompt template for LLM-powered theme generation."""

from __future__ import annotations

import json
import random

# Category hints for rotating diversity suggestions
_CATEGORY_HINTS: list[str] = [
    "professions or jobs",
    "activities or hobbies",
    "places or locations",
    "emotions or feelings",
    "seasons or weather",
    "food or cooking",
    "music or sounds",
    "animals or nature",
    "sports or games",
    "clothing or fashion",
    "science or technology",
    "wordplay or puns",
]


def build_theme_generation_prompt(
    grid_size: int,
    available_slot_lengths: list[int] | None = None,
    num_seed_entries: int = 3,
    slot_counts: dict[int, int] | None = None,
    num_candidates: int | None = None,
    avoid_topics: list[str] | None = None,
    max_avoid_in_prompt: int = 30,
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
        avoid_topics: Previously-generated topics to avoid (for dedup).
        max_avoid_in_prompt: Maximum number of avoid topics to include in
            the prompt. When exceeded, shows a sample with a note.

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
                "Soaring high above, or what the theme "
                "entries can do"
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
        f"- The revealer must be at most {revealer_max} letters long and should be\n"
        f"  one of the longer entries.\n"
        f"- Seed entries should be between 3 and {grid_size} letters long.\n"
        f"\n"
        f"LENGTH DISTRIBUTION (critical — count letters carefully):\n"
        f"- At least 3 entries MUST be exactly 3 letters long (e.g., OWL, BAT, FLY).\n"
        f"- At least 2 entries should be 4-5 letters long.\n"
        f"- Include 2-3 entries of 6-{grid_size} letters."
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
        "entry AND as the 'aha moment' for the theme.\n"
        "- The revealer_clue MUST start with a real, standalone clue "
        "for the revealer word (e.g., 'Completely destroyed' for "
        "SHATTERED), then optionally add a theme hint after a comma "
        "or 'or'. Never write a clue that ONLY describes the theme "
        "connection (e.g., avoid 'Quality shared by all theme entries')."
    )

    avoid_section = ""
    if avoid_topics:
        shown_topics = avoid_topics
        cap_note = ""
        total = len(avoid_topics)
        if total > max_avoid_in_prompt:
            # Show recent + random sample of older topics
            recent = avoid_topics[-15:]
            older = avoid_topics[:-15]
            sampled = random.sample(older, min(15, len(older)))
            shown_topics = sampled + recent
            cap_note = (
                f"\n(Showing {len(shown_topics)} of {total} existing "
                f"topics. There are many more — be highly original.)\n"
            )
        topic_list = "\n".join(f"- {t}" for t in shown_topics)
        avoid_section = (
            "\nAVOID THESE TOPICS (already generated — choose something "
            "fundamentally different):\n"
            f"{topic_list}\n"
            f"{cap_note}"
        )

        # Anti-pattern warning
        avoid_section += (
            "\nDo NOT use 'Things that are [adjective]' or "
            "'Things that can be [past participle]' — those patterns "
            "have been heavily overused. Be creative and specific.\n"
        )

        # Rotating category hint
        hint_index = len(avoid_topics) % len(_CATEGORY_HINTS)
        category = _CATEGORY_HINTS[hint_index]
        avoid_section += (
            f"\nSuggestion: consider a theme related to {category}.\n"
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
        f"{guidelines}\n{avoid_section}\n{output_section}"
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
    # Pick seed entries — lead with 3-letter words to model the distribution
    entries: list[str] = []
    target_lengths = [3, 3, 3, 4, 5]  # emphasize 3-letter words in example
    for length in target_lengths:
        if length <= grid_size and length in _EXAMPLES_BY_LENGTH:
            words = _EXAMPLES_BY_LENGTH[length]
            word = words[len(entries) % len(words)]
            if word not in entries:
                entries.append(word)

    # Fill remaining if needed
    while len(entries) < 5:
        for length in sorted(_EXAMPLES_BY_LENGTH.keys()):
            if length <= grid_size:
                words = _EXAMPLES_BY_LENGTH[length]
                for word in words:
                    if word not in entries:
                        entries.append(word)
                        break
            if len(entries) >= 5:
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
