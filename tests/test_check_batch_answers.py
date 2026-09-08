"""Tests for the check-batch-answers cross-puzzle duplicate gate."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from crossword_generator.cli import check_batch_answers
from crossword_generator.clue_history import extract_ipuz_answers


def _ipuz(rows: list[str]) -> dict[str, object]:
    size = len(rows)
    return {
        "version": "http://ipuz.org/v2",
        "kind": ["http://ipuz.org/crossword#1"],
        "dimensions": {"width": size, "height": size},
        "solution": [[cell for cell in row] for row in rows],
    }


def _write_batch(
    tmp_path: Path,
    results: list[tuple[str, int, int, list[str] | None]],
    target_days: list[int] | None = None,
) -> Path:
    """Write ipuz files + manifest for (difficulty, size, seed, rows) results.

    ``rows=None`` marks a failed result with no output file. ``target_days``
    stamps each result with an absolute ``target_day_number`` (a targeted
    open-day batch).
    """
    manifest_results = []
    for index, (difficulty, size, seed, rows) in enumerate(results):
        output_path = tmp_path / f"{difficulty}-{size}-seed-{seed}.ipuz"
        if rows is not None:
            output_path.write_text(json.dumps(_ipuz(rows)))
        record = {
            "difficulty": difficulty,
            "size": size,
            "seed": seed,
            "success": rows is not None,
            "output_path": str(output_path),
        }
        if target_days is not None:
            record["target_day_number"] = target_days[index]
            record["target_date"] = f"day-{target_days[index]}"
        manifest_results.append(record)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"batch": "test-batch", "results": manifest_results})
    )
    return manifest_path


def test_extract_ipuz_answers_reads_across_and_down() -> None:
    answers = extract_ipuz_answers(_ipuz(["BAT", "AGO", "RYE"]))
    assert sorted(answers) == ["AGO", "AGY", "BAR", "BAT", "RYE", "TOE"]


def test_clean_batch_passes(tmp_path: Path) -> None:
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 5, 1, ["BAT", "AGO", "RYE"]),
            ("hard", 5, 2, ["CUP", "ONE", "DIM"]),
        ],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 0
    assert "No duplicate answers across puzzles." in result.output


def test_cross_puzzle_duplicate_fails_as_blocking(tmp_path: Path) -> None:
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 5, 1, ["BAT", "AGO", "RYE"]),
            ("hard", 9, 2, ["BAT", "AGO", "RYE"]),
        ],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 1
    assert "DUPLICATE [blocking]: BAT" in result.output
    assert "easy/5x5/seed-1" in result.output
    assert "hard/9x9/seed-2" in result.output


SPACED_NINES = [
    # Seed rank within the easy/9 bucket is the day: BAT on days 1 and 4.
    ("easy", 9, 11, ["BAT", "AGO", "RYE"]),
    ("easy", 9, 12, ["CUP", "ONE", "DIM"]),
    ("easy", 9, 13, ["FEW", "IRE", "GEL"]),
    ("easy", 9, 14, ["BAT", "ELM", "DYE"]),
]


def test_short_answer_between_nines_spaced_is_short_window(tmp_path: Path) -> None:
    manifest = _write_batch(tmp_path, SPACED_NINES)
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 1
    assert (
        "DUPLICATE [short-window (9x9-only, 3+ days apart in seed order)]: BAT"
        in result.output
    )
    assert "easy/9x9/seed-11 (day 1), easy/9x9/seed-14 (day 4)" in result.output
    assert "0 blocking, 1 short-window" in result.output


def test_short_answer_between_nines_within_window_is_blocking(
    tmp_path: Path,
) -> None:
    """Adjacent days in seed order trip the scheduler's +/-2-day window."""
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 9, 1, ["BAT", "AGO", "RYE"]),
            ("easy", 9, 2, ["CUP", "ONE", "DIM"]),
            ("easy", 9, 3, ["BAT", "ELM", "DYE"]),
        ],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 1
    assert (
        "DUPLICATE [blocking (9x9 3-letter within +/-2 days in seed order)]: BAT"
        in result.output
    )
    assert "1 blocking, 0 short-window" in result.output


def test_short_answer_across_tracks_same_day_is_blocking(tmp_path: Path) -> None:
    """Easy and hard tracks are scheduled on the same days, so the first
    puzzle of each bucket shares a day."""
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 9, 1, ["BAT", "AGO", "RYE"]),
            ("hard", 9, 2, ["BAT", "ONE", "DIM"]),
        ],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 1
    assert "blocking (9x9 3-letter within +/-2 days in seed order)]: BAT" in (
        result.output
    )
    assert "easy/9x9/seed-1 (day 1), hard/9x9/seed-2 (day 1)" in result.output


def test_allow_short_window_passes_when_only_short_dupes(tmp_path: Path) -> None:
    manifest = _write_batch(tmp_path, SPACED_NINES)
    result = CliRunner().invoke(
        check_batch_answers,
        ["--manifest", str(manifest), "--allow-short-window"],
    )
    assert result.exit_code == 0
    assert "Short-window duplicates allowed" in result.output


def test_allow_short_window_still_fails_on_blocking(tmp_path: Path) -> None:
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 9, 1, ["BAT", "AGO", "RYE"]),
            ("hard", 5, 2, ["BAT", "ONE", "DIM"]),
        ],
    )
    result = CliRunner().invoke(
        check_batch_answers,
        ["--manifest", str(manifest), "--allow-short-window"],
    )
    assert result.exit_code == 1
    assert "DUPLICATE [blocking]: BAT" in result.output


def test_failed_results_are_skipped(tmp_path: Path) -> None:
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 5, 1, ["BAT", "AGO", "RYE"]),
            ("easy", 5, 2, None),
        ],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 0
    assert "1 puzzle(s)" in result.output


def test_write_answers_file_lists_unique_answers(tmp_path: Path) -> None:
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 5, 1, ["BAT", "AGO", "RYE"]),
            ("hard", 5, 2, ["CUP", "ONE", "DIM"]),
        ],
    )
    answers_file = tmp_path / "batch-answers.txt"
    result = CliRunner().invoke(
        check_batch_answers,
        ["--manifest", str(manifest), "--write-answers-file", str(answers_file)],
    )
    assert result.exit_code == 0
    written = answers_file.read_text().splitlines()
    assert written == sorted(set(written))
    assert "BAT" in written and "COD" in written


# --- Targeted (open-day) batches: absolute days, scheduler windows ---

_NINE_A = [
    "BATSHIELD",
    "ALEUTIANS",
    "REDACTORS",
    "SEMINALLY",
    "ESSAYIST#",
    "DRAWLED##",
    "###ROTATE",
    "##REPINES",
    "#SEATTLES",
]


def _nine_with(word: str, filler: str) -> list[str]:
    """A 9x9 whose first row is ``word`` (9 letters) and rest is ``filler``."""
    return [word] + [filler * 9] * 8


def test_targeted_long_answer_outside_regular_window_is_not_blocking(
    tmp_path: Path,
) -> None:
    # AMIDSHIPS appears in two 9x9s 16 days apart: the scheduler's +/-6
    # window does not reach, so the gate must not block (the generator
    # deliberately allowed it while filling).
    manifest = _write_batch(
        tmp_path,
        [
            ("hard", 9, 1, _nine_with("AMIDSHIPS", "Q")),
            ("hard", 9, 2, _nine_with("AMIDSHIPS", "Z")),
        ],
        target_days=[167, 183],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 0, result.output
    assert "regular-window" in result.output
    assert "DUPLICATE [blocking" not in result.output
    assert "schedule each puzzle on its target_date" in result.output


def test_targeted_long_answer_inside_regular_window_is_blocking(
    tmp_path: Path,
) -> None:
    manifest = _write_batch(
        tmp_path,
        [
            ("hard", 9, 1, _nine_with("AMIDSHIPS", "Q")),
            ("hard", 9, 2, _nine_with("AMIDSHIPS", "Z")),
        ],
        target_days=[167, 173],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 1
    assert "blocking" in result.output


def test_targeted_short_answer_between_nines_uses_two_day_window(
    tmp_path: Path,
) -> None:
    # ACE in two 9x9s 4 days apart: outside +/-2, so short-window, and
    # short-window still needs --allow-short-window to pass.
    rows_a = ["ACE", "QQQ", "QQQ"]
    rows_b = ["ACE", "ZZZ", "ZZZ"]
    manifest = _write_batch(
        tmp_path,
        [
            ("hard", 9, 1, [r + "#" * 6 for r in rows_a] + ["#" * 9] * 6),
            ("hard", 9, 2, [r + "#" * 6 for r in rows_b] + ["#" * 9] * 6),
        ],
        target_days=[183, 187],
    )
    strict = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert strict.exit_code == 1
    assert "short-window" in strict.output
    lenient = CliRunner().invoke(
        check_batch_answers,
        ["--manifest", str(manifest), "--allow-short-window"],
    )
    assert lenient.exit_code == 0, lenient.output


def test_targeted_short_answer_with_mini_uses_six_day_window(
    tmp_path: Path,
) -> None:
    # ACE shared by a 5x5 mini and a 9x9 four days apart: minis use the
    # +/-6 glue window, so this is blocking.
    manifest = _write_batch(
        tmp_path,
        [
            ("easy", 5, 1, ["ACE##", "QQQQQ", "QQQQQ", "QQQQQ", "QQQQQ"]),
            ("hard", 9, 2, ["ACE" + "#" * 6] + ["#" * 9] * 8),
        ],
        target_days=[183, 187],
    )
    result = CliRunner().invoke(
        check_batch_answers, ["--manifest", str(manifest)]
    )
    assert result.exit_code == 1
    assert "blocking" in result.output
