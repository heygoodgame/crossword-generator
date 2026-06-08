"""Clue-history indexing for avoiding exact clue reuse across puzzles."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from crossword_generator.exporters.numbering import compute_numbering
from crossword_generator.models import ClueEntry

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.?!]+$")
_QUOTE_TRANS = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
})


@dataclass(frozen=True)
class DuplicateClueHit:
    """A generated clue that exactly matches a prior clue for the same answer."""

    clue: ClueEntry
    existing_clue: str


@dataclass
class ClueHistoryIndex:
    """In-memory answer -> clue index built from generated puzzle records.

    Thread-safe: a lock guards all reads and writes so the index can be shared
    across parallel batch items (which both read it for duplicate avoidance and
    mutate it as puzzles complete).
    """

    _clues_by_answer: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, answer: str, clue: str) -> None:
        answer_key = _normalize_answer(answer)
        clue_text = clue.strip()
        clue_key = normalize_clue(clue_text)
        if not answer_key or not clue_key:
            return
        with self._lock:
            self._clues_by_answer.setdefault(answer_key, {}).setdefault(
                clue_key, set()
            ).add(clue_text)

    def add_clues(self, clues: Iterable[ClueEntry]) -> None:
        for clue in clues:
            self.add(clue.answer, clue.clue)

    def add_ipuz(self, puzzle: dict[str, Any]) -> int:
        """Extract answer/clue pairs from an IPUZ-like payload and add them."""
        pairs = extract_ipuz_answer_clues(puzzle)
        for answer, clue in pairs:
            self.add(answer, clue)
        return len(pairs)

    def add_record(self, record: dict[str, Any]) -> int:
        data = record.get("data")
        if not isinstance(data, dict):
            return 0
        return self.add_ipuz(data)

    def avoid_clues_for_answers(
        self, answers: Iterable[str], *, limit_per_answer: int = 5
    ) -> dict[str, list[str]]:
        avoid: dict[str, list[str]] = {}
        with self._lock:
            for answer in answers:
                answer_key = _normalize_answer(answer)
                clue_map = self._clues_by_answer.get(answer_key, {})
                if not clue_map:
                    continue
                originals = sorted(
                    {c for values in clue_map.values() for c in values}
                )
                avoid[answer_key] = originals[:limit_per_answer]
        return avoid

    def find_duplicates(self, clues: Iterable[ClueEntry]) -> list[DuplicateClueHit]:
        hits: list[DuplicateClueHit] = []
        with self._lock:
            for clue in clues:
                answer_key = _normalize_answer(clue.answer)
                clue_key = normalize_clue(clue.clue)
                existing = self._clues_by_answer.get(answer_key, {}).get(clue_key)
                if existing:
                    hits.append(
                        DuplicateClueHit(
                            clue=clue,
                            existing_clue=sorted(existing)[0],
                        )
                    )
        return hits

    @property
    def answer_count(self) -> int:
        with self._lock:
            return len(self._clues_by_answer)

    @property
    def clue_count(self) -> int:
        with self._lock:
            return sum(
                len(clue_map) for clue_map in self._clues_by_answer.values()
            )


def normalize_clue(clue: str) -> str:
    text = clue.translate(_QUOTE_TRANS).strip().casefold()
    text = _WHITESPACE_RE.sub(" ", text)
    text = _TRAILING_PUNCT_RE.sub("", text)
    return text.strip()


def extract_ipuz_answer_clues(puzzle: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(answer, clue)`` pairs from an IPUZ crossword payload."""
    solution = puzzle.get("solution")
    clues = puzzle.get("clues")
    if not isinstance(solution, list) or not isinstance(clues, dict):
        return []

    grid = _solution_to_grid(solution)
    entries = compute_numbering(grid)
    entry_lookup = {
        (entry.number, entry.direction.capitalize()): entry.answer
        for entry in entries
    }

    pairs: list[tuple[str, str]] = []
    for section_name in ("Across", "Down"):
        section = clues.get(section_name, [])
        if not isinstance(section, list):
            continue
        for item in section:
            parsed = _parse_ipuz_clue_item(item)
            if parsed is None:
                continue
            number, clue_text = parsed
            answer = entry_lookup.get((number, section_name))
            if answer:
                pairs.append((answer, clue_text))
    return pairs


def _solution_to_grid(solution: list[Any]) -> list[list[str]]:
    grid: list[list[str]] = []
    for row in solution:
        if not isinstance(row, list):
            return []
        grid_row: list[str] = []
        for cell in row:
            text = str(cell)
            grid_row.append("." if text == "#" else text.upper())
        grid.append(grid_row)
    return grid


def _parse_ipuz_clue_item(item: object) -> tuple[int, str] | None:
    if not isinstance(item, list) or len(item) < 2:
        return None
    try:
        number = int(item[0])
    except (TypeError, ValueError):
        return None
    clue_text = str(item[1])
    return number, clue_text


def _normalize_answer(answer: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", answer.upper())
