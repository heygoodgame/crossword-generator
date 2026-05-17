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


_ROLE = (
    "You are an expert crossword puzzle constructor designing a theme "
    "for a themed crossword puzzle."
)

_GUIDELINES = (
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
    "or 'or'. Use natural phrasing like 'a hint to some other "
    "answers in this puzzle' — never use the phrase 'theme entries.' "
    "Never write a clue that ONLY describes the theme connection."
)


def build_theme_generation_messages(
    grid_size: int,
    available_slot_lengths: list[int] | None = None,
    num_seed_entries: int = 3,
    slot_counts: dict[int, int] | None = None,
    num_candidates: int | None = None,
    avoid_topics: list[str] | None = None,
    max_avoid_in_prompt: int = 30,
) -> tuple[str, str]:
    """Build (system, user) messages for theme generation.

    The system text holds the constructor role, length constraints, and
    guidelines — these vary only with ``grid_size``, so within a batch
    of same-size puzzles the cache hits cleanly. Per-call data
    (avoid_topics, surplus instructions) lives in the user text so
    retries can append error feedback without invalidating the cache.
    """
    effective_count = num_candidates if num_candidates else num_seed_entries

    revealer_max = grid_size
    if available_slot_lengths:
        revealer_max = max(available_slot_lengths)

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

    example_entries, example_revealer = _pick_example_entries(grid_size)
    example_output = json.dumps(
        {
            "topic": "Things that fly",
            "wordplay_type": "literal",
            "seed_entries": example_entries,
            "revealer": example_revealer,
            "revealer_clue": (
                "Soaring high above, or a hint to some "
                "other answers in this puzzle"
            ),
        },
        indent=2,
    )

    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with these fields. "
        "No other text before or after.\n"
        f"\nExample:\n{example_output}\n"
    )

    system_text = "\n\n".join(
        [_ROLE, length_constraint, _GUIDELINES, output_section]
    )

    avoid_section = ""
    if avoid_topics:
        shown_topics = avoid_topics
        cap_note = ""
        total = len(avoid_topics)
        if total > max_avoid_in_prompt:
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

        avoid_section += (
            "\nDo NOT use 'Things that are [adjective]' or "
            "'Things that can be [past participle]' — those patterns "
            "have been heavily overused. Be creative and specific.\n"
        )

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

    user_text = (
        f"{avoid_section}\n"
        f"Generate a theme with exactly {effective_count} "
        f"seed entries for a {grid_size}x{grid_size} crossword "
        f"grid.{surplus_note} Return ONLY the JSON object, "
        f"no explanations."
    )

    return system_text, user_text


def build_theme_generation_prompt(
    grid_size: int,
    available_slot_lengths: list[int] | None = None,
    num_seed_entries: int = 3,
    slot_counts: dict[int, int] | None = None,
    num_candidates: int | None = None,
    avoid_topics: list[str] | None = None,
    max_avoid_in_prompt: int = 30,
) -> str:
    """Build a single-string prompt (system+user concatenated).

    Kept for backward compatibility. Prefer
    ``build_theme_generation_messages`` to enable prompt caching.
    """
    system_text, user_text = build_theme_generation_messages(
        grid_size=grid_size,
        available_slot_lengths=available_slot_lengths,
        num_seed_entries=num_seed_entries,
        slot_counts=slot_counts,
        num_candidates=num_candidates,
        avoid_topics=avoid_topics,
        max_avoid_in_prompt=max_avoid_in_prompt,
    )
    return f"{system_text}\n\n{user_text}"


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
    entries: list[str] = []
    target_lengths = [3, 3, 3, 4, 5]
    for length in target_lengths:
        if length <= grid_size and length in _EXAMPLES_BY_LENGTH:
            words = _EXAMPLES_BY_LENGTH[length]
            word = words[len(entries) % len(words)]
            if word not in entries:
                entries.append(word)

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

    revealer = "SOAR"
    for length in sorted(_EXAMPLES_BY_LENGTH.keys(), reverse=True):
        if length <= grid_size and length in _EXAMPLES_BY_LENGTH:
            for word in _EXAMPLES_BY_LENGTH[length]:
                if word not in entries:
                    revealer = word
                    break
            break

    return entries, revealer
