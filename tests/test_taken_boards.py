"""Tests for the exact-duplicate solution-grid registry."""

from __future__ import annotations

import threading

from crossword_generator.taken_boards import (
    TakenBoards,
    board_key,
    ipuz_board_key,
    record_board_key,
)

FILL_GRID = [
    [".", "C", "A", "R", "T"],
    ["S", "E", "W", "E", "R"],
    ["P", "L", "A", "N", "E"],
    ["E", "L", "I", "T", "E"],
    ["D", "O", "T", "S", "."],
]


def _ipuz(block: object) -> dict[str, object]:
    return {
        "solution": [
            [block if cell == "." else cell.lower() for cell in row]
            for row in FILL_GRID
        ]
    }


def test_board_key_matches_across_block_encodings() -> None:
    """The fill grid, an IPUZ export (#), and the daily-schedule store
    (null blocks, lowercase-tolerant) all normalize to the same key."""
    expected = ".CART|SEWER|PLANE|ELITE|DOTS."
    assert board_key(FILL_GRID) == expected
    assert ipuz_board_key(_ipuz("#")) == expected
    assert ipuz_board_key(_ipuz(None)) == expected
    assert ipuz_board_key(_ipuz("")) == expected


def test_record_board_key_reads_official_and_draft_shapes() -> None:
    official = {"collection": "unlimited-pool", "data": {"puzzle": _ipuz(None)}}
    draft = {"collection": "generated-puzzles", "data": _ipuz("#")}
    assert record_board_key(official) == record_board_key(draft) == board_key(
        FILL_GRID
    )
    assert record_board_key({"data": None}) is None
    assert ipuz_board_key({}) is None


def test_add_keeps_first_holder() -> None:
    taken = TakenBoards()
    assert taken.add("AB|CD", holder="unlimited-pool unlimited:5x5:1") is True
    assert taken.add("AB|CD", holder="local batch export") is False
    assert taken.add(None, holder="ignored") is False
    assert taken.holder("AB|CD") == "unlimited-pool unlimited:5x5:1"
    assert taken.holder("ZZ|ZZ") is None
    assert len(taken) == 1


def test_reserve_refuses_a_board_someone_else_holds() -> None:
    taken = TakenBoards()
    taken.add("AB|CD", holder="prior batch easy 5x5 seed 1")
    assert taken.reserve("AB|CD", owner="easy 5x5 seed 2") is False
    assert taken.reserve("EF|GH", owner="easy 5x5 seed 2") is True
    # Idempotent for the owner, closed to everyone else.
    assert taken.reserve("EF|GH", owner="easy 5x5 seed 2") is True
    assert taken.reserve("EF|GH", owner="easy 5x5 seed 3") is False
    assert taken.holder("EF|GH") == "easy 5x5 seed 2"


def test_reserve_is_atomic_across_threads() -> None:
    """Many workers racing for one board: exactly one gets it."""
    taken = TakenBoards()
    barrier = threading.Barrier(16)
    wins: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        owner = f"easy 5x5 seed {i}"
        barrier.wait()
        if taken.reserve("AB|CD", owner=owner):
            with lock:
                wins.append(owner)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(wins) == 1
    assert taken.holder("AB|CD") == wins[0]
