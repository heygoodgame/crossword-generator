"""Tests for clue-history extraction and duplicate matching."""

from __future__ import annotations

from crossword_generator.clue_history import (
    ClueHistoryIndex,
    extract_ipuz_answer_clues,
    normalize_clue,
)
from crossword_generator.models import ClueEntry


def test_extract_ipuz_answer_clues_pairs_clues_with_numbered_answers() -> None:
    puzzle = {
        "solution": [
            ["C", "A", "T"],
            ["A", "R", "E"],
            ["B", "E", "D"],
        ],
        "clues": {
            "Across": [
                [1, "House pet"],
                [4, "Exist"],
                [5, "Sleep spot"],
            ],
            "Down": [
                [1, "Yellow taxi"],
                [2, "Second person verb"],
                [3, "Common male name"],
            ],
        },
    }

    pairs = extract_ipuz_answer_clues(puzzle)

    assert ("CAT", "House pet") in pairs
    assert ("CAB", "Yellow taxi") in pairs
    assert ("TED", "Common male name") in pairs
    assert len(pairs) == 6


def test_clue_history_detects_normalized_exact_duplicates() -> None:
    history = ClueHistoryIndex()
    history.add("MARIO", "Nintendo plumber?")

    hits = history.find_duplicates([
        ClueEntry(
            number=1,
            direction="across",
            answer="MARIO",
            clue="  Nintendo   plumber! ",
        )
    ])

    assert len(hits) == 1
    assert hits[0].existing_clue == "Nintendo plumber?"


def test_avoid_clues_for_answers_returns_only_matching_answers() -> None:
    history = ClueHistoryIndex()
    history.add("MARIO", "Nintendo plumber")
    history.add("LUIGI", "Mario's brother")

    assert history.avoid_clues_for_answers(["MARIO", "PEACH"]) == {
        "MARIO": ["Nintendo plumber"]
    }


def test_normalize_clue_collapses_punctuation_quotes_and_whitespace() -> None:
    assert normalize_clue(" \u201cNintendo\u201d   plumber?! ") == '"nintendo" plumber'
