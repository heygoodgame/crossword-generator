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


def test_extract_ipuz_answer_clues_reads_object_clues_with_hints() -> None:
    puzzle = {
        "solution": [["O", "F", "F", "S"]],
        "clues": {
            "Across": [
                {
                    "number": 1,
                    "clue": "Turns off, as lights",
                    "hints": ["Switches off"],
                }
            ],
            "Down": [],
        },
    }

    pairs = extract_ipuz_answer_clues(puzzle)

    assert pairs == [("OFFS", "Turns off, as lights")]


def test_extract_ipuz_treats_null_cells_as_blocks() -> None:
    """Daily-schedule IPUZ uses JSON null (None) for blocks, not '#'.

    Regression: blocks serialized as null ran entries straight through the
    block, e.g. OPTS + LOFT -> OPTSNONELOFT.
    """
    puzzle = {
        "solution": [
            ["O", "P", "T", "S", None, "L", "O", "F", "T"],
        ],
        "clues": {
            # In this isolated single row, OPTS is across-1 and LOFT is
            # across-2 (numbering restarts after the null block).
            "Across": [
                [1, "Makes a choice"],
                [2, "Upper-floor apartment"],
            ],
            "Down": [],
        },
    }

    pairs = extract_ipuz_answer_clues(puzzle)

    assert ("OPTS", "Makes a choice") in pairs
    assert ("LOFT", "Upper-floor apartment") in pairs
    assert not any("NONE" in answer for answer, _ in pairs)


def test_add_record_reads_nested_daily_schedule_puzzle() -> None:
    """Official daily-schedule records nest the IPUZ under data.puzzle."""
    history = ClueHistoryIndex()
    record = {
        "data": {
            "track": "main",
            "date": "2026-07-16",
            "puzzle": {
                "solution": [["C", "A", "T"]],
                "clues": {"Across": [[1, "House pet"]], "Down": []},
            },
        }
    }

    added = history.add_record(record)

    assert added == 1
    assert history.clues_for_answer("CAT") == ["House pet"]


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


def test_clue_history_is_thread_safe_under_concurrent_writes() -> None:
    """Many threads adding clues concurrently must not lose or corrupt data."""
    import threading

    history = ClueHistoryIndex()
    n_threads = 8
    per_thread = 100

    def worker(tid: int) -> None:
        for i in range(per_thread):
            history.add(f"ANS{tid}_{i}", f"Clue {tid} {i}")
            # Interleave reads with writes to exercise the lock both ways.
            history.find_duplicates(
                [ClueEntry(number=1, direction="across",
                           answer=f"ANS{tid}_{i}", clue=f"Clue {tid} {i}")]
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every distinct (answer, clue) pair should be present.
    assert history.answer_count == n_threads * per_thread
    assert history.clue_count == n_threads * per_thread
