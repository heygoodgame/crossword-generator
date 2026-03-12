"""Topic deduplication utilities for theme generation.

Pure functions for normalizing, comparing, and deduplicating crossword theme
topics. No dependencies on other project modules.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_STOPWORDS: frozenset[str] = frozenset({
    "things", "that", "are", "can", "be", "a", "an", "the", "or", "and",
    "to", "with", "in", "of", "is", "you", "your", "do", "have", "has",
    "been", "being",
})


def normalize_topic(topic: str) -> str:
    """Normalize a topic string for exact comparison.

    Lowercases, strips quotes and special chars (preserving apostrophes
    within words), and collapses whitespace.
    """
    text = topic.lower()
    # Remove quotes and special punctuation (keep apostrophes within words)
    text = re.sub(r"[\"'`""'']", "", text)
    # Remove non-alphanumeric except spaces and internal apostrophes
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_content_words(topic: str) -> set[str]:
    """Extract meaningful content words from a topic, removing stopwords."""
    normalized = normalize_topic(topic)
    words = set(normalized.split())
    return words - _STOPWORDS


def build_normalized_topic_set(topics: Iterable[str]) -> set[str]:
    """Build a set of normalized topic strings for O(1) duplicate lookup."""
    return {normalize_topic(t) for t in topics}


def is_topic_duplicate(
    new_topic: str, existing_normalized: set[str]
) -> bool:
    """Check if a topic is an exact duplicate after normalization."""
    return normalize_topic(new_topic) in existing_normalized


def is_topic_similar(
    new_topic: str,
    existing_topics: list[str],
    threshold: float = 0.6,
) -> tuple[bool, str | None]:
    """Check if a topic is too similar to any existing topic.

    Uses Jaccard similarity on content words.

    Returns:
        (is_similar, closest_match) — closest_match is the original topic
        string that triggered the similarity, or None if not similar.
    """
    new_words = extract_content_words(new_topic)
    if not new_words:
        return False, None

    for existing in existing_topics:
        existing_words = extract_content_words(existing)
        if not existing_words:
            continue
        intersection = new_words & existing_words
        union = new_words | existing_words
        similarity = len(intersection) / len(union)
        if similarity >= threshold:
            return True, existing

    return False, None
