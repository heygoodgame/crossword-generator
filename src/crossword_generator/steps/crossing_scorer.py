"""Score theme words for crossing friendliness against the dictionary.

Pre-builds a letter-position index from the dictionary, then scores each
theme word by how well the dictionary supports its letters at likely
crossing positions. The bottleneck (minimum support) drives the score.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from crossword_generator.dictionary import Dictionary

logger = logging.getLogger(__name__)

# Type alias: for each word length, a list of dicts mapping letter -> count
# Index i maps letter -> number of words that have that letter at position i.
LetterPositionIndex = dict[int, list[dict[str, int]]]


def build_letter_position_index(
    dictionary: Dictionary,
    grid_size: int,
) -> LetterPositionIndex:
    """Build an index of letter frequencies at each position by word length.

    For each word length from 3 to grid_size, counts how many dictionary
    words have each letter at each position.

    Args:
        dictionary: The word dictionary.
        grid_size: Maximum word length to index.

    Returns:
        A dict mapping word_length -> list of {letter: count} dicts.
    """
    index: LetterPositionIndex = {}
    for length in range(3, grid_size + 1):
        words = dictionary.words_by_length(length)
        if not words:
            continue
        position_counts: list[dict[str, int]] = [
            defaultdict(int) for _ in range(length)
        ]
        for word in words:
            for pos, letter in enumerate(word):
                position_counts[pos][letter] += 1
        # Convert defaultdicts to regular dicts
        index[length] = [dict(d) for d in position_counts]
    return index


def score_word(
    word: str,
    index: LetterPositionIndex,
    grid_size: int,
) -> float:
    """Score a single word for crossing friendliness.

    For each letter in the word, estimates how many dictionary words
    could cross at that position. The score is the log of the minimum
    support across all positions (the bottleneck).

    A higher score means the word is easier to cross in a grid.

    Args:
        word: The word to score (uppercase).
        index: Pre-built letter-position index.
        grid_size: The grid dimension.

    Returns:
        A float score >= 0. Higher is better.
    """
    word = word.upper()
    word_len = len(word)

    if word_len < 3 or word_len > grid_size:
        return 0.0

    min_support = float("inf")

    for pos, letter in enumerate(word):
        # For each position in this word, a crossing word can be of any
        # length from 3 to grid_size. Sum up support across all possible
        # crossing word lengths.
        total_support = 0
        for cross_len in range(3, grid_size + 1):
            if cross_len not in index:
                continue
            cross_positions = index[cross_len]
            # The crossing word has this letter at some position within it
            for cross_pos in range(cross_len):
                total_support += cross_positions[cross_pos].get(letter, 0)

        min_support = min(min_support, total_support)

    if min_support == float("inf") or min_support <= 0:
        return 0.0

    # Log scale to compress the range
    return math.log1p(min_support)


def rank_candidates(
    candidates: list[str],
    revealer: str,
    dictionary: Dictionary,
    grid_size: int,
) -> list[tuple[str, float]]:
    """Rank candidate theme words by crossing friendliness.

    Args:
        candidates: List of candidate words (uppercase).
        revealer: The revealer word (excluded from ranking since it's always used).
        dictionary: The word dictionary.
        grid_size: The grid dimension.

    Returns:
        List of (word, score) pairs sorted by score descending (best first).
    """
    index = build_letter_position_index(dictionary, grid_size)

    scored: list[tuple[str, float]] = []
    for word in candidates:
        word_upper = word.upper()
        if word_upper == revealer.upper():
            continue
        s = score_word(word_upper, index, grid_size)
        scored.append((word_upper, s))
        logger.debug("Crossing score for %s: %.2f", word_upper, s)

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
