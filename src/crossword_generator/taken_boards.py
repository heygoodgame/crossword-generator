"""Hard guard against exact-duplicate solution grids.

Answer-novelty weighting only makes reuse less likely; before this guard,
nothing stopped two puzzles from shipping the identical board. On 2026-09-25,
17 groups (38 Mini unlimited-pool puzzles) were exact duplicates, e.g.
``.CART|SEWER|PLANE|ELITE|DOTS.`` from two adjacent seeds of one Starter batch.
Concurrent workers and back-to-back chunks each picked from the same usage
snapshot, so the best-of-N novelty pick kept converging on the same few
"least-used" boards.

``TakenBoards`` is the registry every fill checks against. The batch runner
seeds it with every board that already exists (live official records,
uploaded drafts, earlier local batch exports, ``--prior-batch-manifest``
puzzles), and the fill step *reserves* its pick before the clue stage, so a
concurrent batch-mate that wanted the same board falls through to its next
candidate instead.
"""

from __future__ import annotations

import threading
from typing import Any

from crossword_generator.clue_history import ipuz_solution_grid


def board_key(grid: list[list[str]]) -> str | None:
    """Normalized full-solution string: rows joined by ``|``, blocks as ``.``.

    Expects the fill grid shape (letters plus ``.`` blocks); IPUZ payloads go
    through :func:`ipuz_board_key`, which normalizes their block encodings
    first. Keys of different grid sizes never collide, so one registry covers
    every size, and difficulty is deliberately not part of the key: an Easy
    and a Hard puzzle with the same board are duplicates too.
    """
    if not grid:
        return None
    return "|".join("".join(cell.upper() for cell in row) for row in grid)


def ipuz_board_key(puzzle: dict[str, Any]) -> str | None:
    """Board key of an IPUZ payload (``#``/``null``/empty blocks all match)."""
    return board_key(ipuz_solution_grid(puzzle))


def record_board_key(record: dict[str, Any]) -> str | None:
    """Board key of an admin data-store record.

    Official records (daily schedule, unlimited pool) nest the IPUZ under
    ``data.puzzle``; generated-puzzle drafts store it as ``data`` itself.
    """
    data = record.get("data")
    if not isinstance(data, dict):
        return None
    puzzle = data.get("puzzle")
    return ipuz_board_key(puzzle if isinstance(puzzle, dict) else data)


class TakenBoards:
    """Thread-safe registry of solution grids that are already in use.

    Maps each board key to a human-readable holder (a live record, a prior
    batch puzzle, or the batch item that reserved it) for log messages.
    """

    def __init__(self) -> None:
        self._holders: dict[str, str] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._holders)

    def add(self, key: str | None, *, holder: str) -> bool:
        """Mark a board as taken; returns True if it was not taken before."""
        if not key:
            return False
        with self._lock:
            if key in self._holders:
                return False
            self._holders[key] = holder
            return True

    def holder(self, key: str | None) -> str | None:
        """Who holds a board, or None if it is free."""
        if not key:
            return None
        with self._lock:
            return self._holders.get(key)

    def reserve(self, key: str | None, *, owner: str) -> bool:
        """Atomically claim a free board for ``owner``.

        Returns False when someone else already holds it. Reserving a board
        the owner already holds succeeds, so retries are idempotent.
        """
        if not key:
            return True
        with self._lock:
            current = self._holders.get(key)
            if current is not None and current != owner:
                return False
            self._holders[key] = owner
            return True
