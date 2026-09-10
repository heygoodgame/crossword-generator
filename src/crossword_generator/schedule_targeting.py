"""Target specific open daily-schedule days when generating a batch.

The default daily batch anchors its recent-answer exclusion on the server's
"first unscheduled slot" and assumes a contiguous block of open days follows
it. That model broke once the live schedule ran months ahead with scattered
single-day holes (2026-09: the schedule reached July 2027; midi easy had 15
holes spread over Oct-Nov while the anchor pointed at a midi HARD hole). None
of the answers scheduled around the real holes were excluded, so every
candidate collided on every hole and the scheduler walked them into 2027.

This module reads the live daily schedule, finds the open days for one
game/track, and builds a per-target-day exclusion set that mirrors the
scheduler's actual rules (``CrosswordPuzzleController::dailyAnswerConflict``):

- regular answers (4+ letters): +/-6 days, cross-game and cross-track;
- short glue (<= 3 letters): +/-2 days when placing a 9x9, else +/-6;
- HGG 60 answers: +/-180 days.

Each generated puzzle then carries its ``target_date`` so the reviewer (or the
scheduler's metadata fallback) schedules it onto the day it was filled for.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from crossword_generator.clue_history import extract_ipuz_answers

REGULAR_WINDOW_DAYS = 6
SHORT_WINDOW_DAYS = 2
SIXTY_WINDOW_DAYS = 180
SHORT_ANSWER_MAX_LENGTH = 3
# The relaxed short-glue window only applies when the puzzle being placed is
# a 9x9 (mirrors the server's SHORT_WINDOW_SIZE).
SHORT_WINDOW_SIZE = 9

MINI_WEEKDAY_SIZE = 5
MINI_WEEKEND_SIZE = 7

DAILY_GAME_KEYS = ("midicrossword", "minicrossword")
DAILY_TRACKS = ("easy", "hard")


@dataclass(frozen=True)
class ScheduledSlot:
    """One scheduled daily slot with its normalized answer set."""

    game_key: str
    track: str
    day_number: int
    date: _dt.date
    size: int
    answers: frozenset[str]
    title: str | None = None


def parse_date(value: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def slots_from_official_records(
    records: Iterable[dict[str, Any]],
) -> list[ScheduledSlot]:
    """Build scheduled slots from ``/admin/crossword-puzzles/official`` rows.

    Only rows in the ``daily-schedule`` collection with ``status=scheduled``
    count; retired slots no longer constrain the scheduler.
    """
    slots: list[ScheduledSlot] = []
    for record in records:
        if record.get("collection") not in (None, "daily-schedule"):
            continue
        if record.get("status") != "scheduled":
            continue
        data = record.get("data") or {}
        puzzle = data.get("puzzle")
        if not isinstance(puzzle, dict):
            continue
        track = str(data.get("track") or "").strip().lower()
        day_number = data.get("dayNumber", data.get("day_number"))
        date_value = data.get("date")
        if not track or day_number is None or not date_value:
            continue
        metadata = record.get("metadata") or {}
        size = metadata.get("size")
        if size is None:
            size = (puzzle.get("dimensions") or {}).get("width")
        answers = frozenset(
            answer.strip().upper()
            for answer in extract_ipuz_answers(puzzle)
            if answer.strip()
        )
        slots.append(
            ScheduledSlot(
                game_key=str(record.get("game_key") or ""),
                track=track,
                day_number=int(day_number),
                date=parse_date(str(date_value)[:10]),
                size=int(size or 0),
                answers=answers,
                title=metadata.get("title"),
            )
        )
    return slots


def infer_epoch(slots: Iterable[ScheduledSlot]) -> _dt.date:
    """Day-1 date implied by the slots (every slot agrees by construction)."""
    for slot in slots:
        return slot.date - _dt.timedelta(days=slot.day_number - 1)
    raise ValueError("Cannot infer the daily epoch from an empty schedule")


def day_number_for_date(date: _dt.date, epoch: _dt.date) -> int:
    return max(1, (date - epoch).days + 1)


def date_for_day_number(day_number: int, epoch: _dt.date) -> _dt.date:
    return epoch + _dt.timedelta(days=day_number - 1)


def mini_size_for_date(date: _dt.date) -> int:
    """Mini daily cadence: 7x7 on weekends, 5x5 on weekdays."""
    return MINI_WEEKEND_SIZE if date.weekday() >= 5 else MINI_WEEKDAY_SIZE


def size_for_target(game_key: str, date: _dt.date) -> int:
    return mini_size_for_date(date) if game_key == "minicrossword" else 9


def open_day_numbers(
    slots: Iterable[ScheduledSlot],
    *,
    game_key: str,
    track: str,
    start_day: int,
    end_day: int,
) -> list[int]:
    """Day numbers in ``[start_day, end_day]`` with no scheduled slot."""
    taken = {
        slot.day_number
        for slot in slots
        if slot.game_key == game_key and slot.track == track
    }
    return [day for day in range(start_day, end_day + 1) if day not in taken]


class ScheduleAnswerIndex:
    """Per-day answer exclusions derived from the live daily schedule."""

    def __init__(
        self,
        slots: Iterable[ScheduledSlot],
        *,
        sixty_answers: Iterable[str] = (),
    ) -> None:
        self._by_answer: dict[str, list[ScheduledSlot]] = {}
        self._slots = list(slots)
        for slot in self._slots:
            for answer in slot.answers:
                self._by_answer.setdefault(answer, []).append(slot)
        self._sixty = {a.strip().upper() for a in sixty_answers if a.strip()}

    @property
    def slots(self) -> list[ScheduledSlot]:
        return list(self._slots)

    def window_for(self, answer: str, *, size: int) -> int:
        if answer in self._sixty:
            return SIXTY_WINDOW_DAYS
        if size == SHORT_WINDOW_SIZE and len(answer) <= SHORT_ANSWER_MAX_LENGTH:
            return SHORT_WINDOW_DAYS
        return REGULAR_WINDOW_DAYS

    def excluded_for(self, *, day_number: int, size: int) -> set[str]:
        """Answers a puzzle placed on ``day_number`` may not contain."""
        excluded: set[str] = set()
        for answer, slots in self._by_answer.items():
            window = self.window_for(answer, size=size)
            if any(abs(slot.day_number - day_number) <= window for slot in slots):
                excluded.add(answer)
        return excluded

    def regular_window_answers(self, *, day_number: int) -> set[str]:
        """4+ letter answers scheduled within the +/-6 regular window.

        Used as the seed for inflectional-variant expansion (ART on Monday
        should also keep ARTS off Tuesday) — variants of short glue and of
        the 180-day sixty list are deliberately not expanded.
        """
        return {
            answer
            for answer, slots in self._by_answer.items()
            if len(answer) > SHORT_ANSWER_MAX_LENGTH
            and any(
                abs(slot.day_number - day_number) <= REGULAR_WINDOW_DAYS
                for slot in slots
            )
        }

    def conflicts_for(
        self,
        answers: Iterable[str],
        *,
        day_number: int,
        size: int,
    ) -> list[tuple[str, ScheduledSlot, int]]:
        """(answer, slot, window) triples that would block ``answers`` on a day."""
        hits: list[tuple[str, ScheduledSlot, int]] = []
        for answer in {a.strip().upper() for a in answers}:
            window = self.window_for(answer, size=size)
            for slot in self._by_answer.get(answer, []):
                if abs(slot.day_number - day_number) <= window:
                    hits.append((answer, slot, window))
        hits.sort(key=lambda hit: (hit[1].day_number, hit[0]))
        return hits
