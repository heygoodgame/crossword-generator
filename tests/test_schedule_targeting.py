"""Open-day targeting: schedule parsing, open days, per-day exclusions."""

from __future__ import annotations

import datetime as dt

from crossword_generator.schedule_targeting import (
    ScheduleAnswerIndex,
    ScheduledSlot,
    date_for_day_number,
    day_number_for_date,
    infer_epoch,
    mini_size_for_date,
    open_day_numbers,
    slots_from_official_records,
)

EPOCH = dt.date(2026, 6, 22)


def _slot(day: int, answers: set[str], *, game="midicrossword", track="easy", size=9):
    return ScheduledSlot(
        game_key=game,
        track=track,
        day_number=day,
        date=EPOCH + dt.timedelta(days=day - 1),
        size=size,
        answers=frozenset(answers),
    )


def _official_record(day: int, rows: list[str], *, status="scheduled", track="easy"):
    size = len(rows)
    return {
        "collection": "daily-schedule",
        "game_key": "midicrossword",
        "status": status,
        "data": {
            "track": track,
            "dayNumber": day,
            "date": (EPOCH + dt.timedelta(days=day - 1)).isoformat(),
            "puzzle": {
                "dimensions": {"width": size, "height": size},
                "solution": [list(row) for row in rows],
                "clues": {"Across": [], "Down": []},
            },
        },
        "metadata": {"size": size, "title": f"Day {day}"},
    }


def test_slots_from_official_records_reads_answers_and_skips_retired() -> None:
    records = [
        _official_record(10, ["CAT", "###", "DOG"]),
        _official_record(11, ["EEL", "###", "EMU"], status="retired"),
    ]
    slots = slots_from_official_records(records)
    assert len(slots) == 1
    slot = slots[0]
    assert slot.day_number == 10
    assert slot.date == dt.date(2026, 7, 1)
    assert slot.size == 3
    assert {"CAT", "DOG"} <= slot.answers
    assert infer_epoch(slots) == EPOCH


def test_day_number_date_roundtrip_and_mini_cadence() -> None:
    assert day_number_for_date(dt.date(2026, 10, 18), EPOCH) == 119
    assert date_for_day_number(119, EPOCH) == dt.date(2026, 10, 18)
    # 2026-10-31 is a Saturday, 2026-11-02 a Monday.
    assert mini_size_for_date(dt.date(2026, 10, 31)) == 7
    assert mini_size_for_date(dt.date(2026, 11, 2)) == 5


def test_open_day_numbers_is_per_game_and_track() -> None:
    slots = [
        _slot(100, {"AAAA"}),
        _slot(102, {"BBBB"}),
        _slot(101, {"CCCC"}, track="hard"),
        _slot(103, {"DDDD"}, game="minicrossword", size=5),
    ]
    assert open_day_numbers(
        slots, game_key="midicrossword", track="easy", start_day=100, end_day=104
    ) == [101, 103, 104]
    assert open_day_numbers(
        slots, game_key="midicrossword", track="hard", start_day=100, end_day=104
    ) == [100, 102, 103, 104]


def test_excluded_for_mirrors_scheduler_windows() -> None:
    slots = [
        _slot(100, {"SNORE", "DOG", "ZYZZYVA"}),
        # A mini on another track still counts (cross-game, cross-track).
        _slot(103, {"MARIA", "ACT"}, game="minicrossword", track="hard", size=5),
    ]
    index = ScheduleAnswerIndex(slots, sixty_answers={"ZYZZYVA"})

    # Regular answers: +/-6 either side.
    assert "SNORE" in index.excluded_for(day_number=106, size=9)
    assert "SNORE" not in index.excluded_for(day_number=107, size=9)
    assert "MARIA" in index.excluded_for(day_number=109, size=9)
    # Short glue placed on a 9x9: +/-2 only.
    assert "DOG" in index.excluded_for(day_number=102, size=9)
    assert "DOG" not in index.excluded_for(day_number=103, size=9)
    assert "ACT" in index.excluded_for(day_number=105, size=9)
    assert "ACT" not in index.excluded_for(day_number=106, size=9)
    # Short glue placed on a mini keeps the full +/-6 window.
    assert "DOG" in index.excluded_for(day_number=106, size=5)
    # Sixty answers: +/-180.
    assert "ZYZZYVA" in index.excluded_for(day_number=280, size=9)
    assert "ZYZZYVA" not in index.excluded_for(day_number=281, size=9)

    # Variant seed: 4+ letter answers in the regular window only.
    assert index.regular_window_answers(day_number=104) == {"SNORE", "ZYZZYVA", "MARIA"}
    assert index.regular_window_answers(day_number=107) == {"MARIA"}


def test_conflicts_for_lists_blocking_slots_in_day_order() -> None:
    slots = [_slot(100, {"SNORE"}), _slot(98, {"SNORE", "DOG"})]
    index = ScheduleAnswerIndex(slots)
    hits = index.conflicts_for(["snore", "dog", "FREE"], day_number=99, size=9)
    assert [(a, s.day_number, w) for a, s, w in hits] == [
        ("DOG", 98, 2),
        ("SNORE", 98, 6),
        ("SNORE", 100, 6),
    ]
