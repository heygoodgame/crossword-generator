"""Tests for HeyGG data-store record preparation and save behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from crossword_generator import data_store
from crossword_generator.data_store import (
    DataStoreApiError,
    DataStoreError,
    bulk_save_generated_puzzles,
    delete_generated_puzzle_records,
    fetch_recent_daily_answers,
    fetch_recent_sixty_answers,
    list_generated_puzzle_records,
    make_record,
    records_from_manifest,
)


def test_make_record_uses_generated_puzzle_contract() -> None:
    record = make_record(
        {"version": "http://ipuz.org/v2", "dimensions": {"width": 5, "height": 5}},
        game_key="minicrossword",
        puzzle_type="mini",
        size=5,
        difficulty="easy",
        batch_id="phase-2b-pilot",
        seed=1,
        generator_version="0.1.0",
        generator_commit="abc123",
        fill_score=55.0,
        clue_score=80.0,
        title="Au Naturel",
        title_reasoning="Au is the chemical symbol for gold.",
    )

    assert record["namespace"] == "crosswords"
    assert record["collection"] == "generated-puzzles"
    assert record["game_key"] == "minicrossword"
    assert record["key"] == "generated:minicrossword:phase-2b-pilot:easy:5x5:seed-1"
    assert record["status"] == "draft"
    assert record["metadata"] == {
        "review_status": "unreviewed",
        "puzzle_type": "mini",
        "size": 5,
        "difficulty": "easy",
        "batch_id": "phase-2b-pilot",
        "seed": "1",
        "generator_version": "0.1.0",
        "generator_commit": "abc123",
        "fill_score": 55.0,
        "clue_score": 80.0,
        "title": "Au Naturel",
        "title_reasoning": "Au is the chemical symbol for gold.",
        "author": "crossword-generator",
        "publication_status": "draft",
    }


def test_make_record_rejects_invalid_key() -> None:
    with pytest.raises(DataStoreError, match="Invalid data-store key"):
        make_record(
            {},
            game_key="minicrossword",
            puzzle_type="mini",
            size=5,
            difficulty="easy",
            batch_id="batch",
            seed=1,
            key="bad key",
        )


def test_records_from_manifest_reads_successful_ipuz_files(tmp_path: Path) -> None:
    puzzle_path = tmp_path / "seed-001.ipuz"
    puzzle_path.write_text(json.dumps({"version": "http://ipuz.org/v2"}))
    missing_failure_path = tmp_path / "failed.ipuz"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "phase-2b-pilot",
                "results": [
                    {
                        "success": True,
                        "output_path": str(puzzle_path),
                        "difficulty": "hard",
                        "size": 9,
                        "seed": 1,
                        "fill_score": 55,
                        "clue_score": 75.5,
                        "title": "Crossing Over",
                        "title_reasoning": "Bridges literally cross over.",
                    },
                    {
                        "success": False,
                        "output_path": str(missing_failure_path),
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 2,
                    },
                ],
            }
        )
    )

    records = records_from_manifest(
        manifest_path,
        generator_version="0.1.0",
        generator_commit="abc123",
    )

    assert len(records) == 1
    assert records[0]["game_key"] == "midicrossword"
    assert records[0]["metadata"]["puzzle_type"] == "midi"
    assert records[0]["metadata"]["size"] == 9
    assert records[0]["metadata"]["difficulty"] == "hard"
    assert records[0]["metadata"]["clue_score"] == 75.5
    assert records[0]["metadata"]["title"] == "Crossing Over"
    assert (
        records[0]["metadata"]["title_reasoning"]
        == "Bridges literally cross over."
    )


def _leaky_manifest(tmp_path: Path) -> Path:
    # Real-world shape: the .ipuz has NO errors field (it's a pure puzzle
    # format); the LEAK: soft error lives in the manifest result's
    # error_message (a "; "-joined string of the envelope's errors).
    puzzle_path = tmp_path / "seed-001.ipuz"
    puzzle_path.write_text(json.dumps({"version": "http://ipuz.org/v2"}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "leak-batch",
                "results": [
                    {
                        "success": True,
                        "output_path": str(puzzle_path),
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 1,
                        "error_message": (
                            'LEAK: CAT (1-across) [exact] in clue "A pet cat" '
                            '(offending: "cat")'
                        ),
                    }
                ],
            }
        )
    )
    return manifest_path


def test_records_from_manifest_skips_leaked_puzzle(tmp_path: Path) -> None:
    """Leak in the manifest error_message (real-world location) is skipped."""
    manifest_path = _leaky_manifest(tmp_path)
    records = records_from_manifest(manifest_path)
    assert records == []


def test_records_from_manifest_skips_leak_in_envelope_errors(
    tmp_path: Path,
) -> None:
    """Fallback: a LEAK in the puzzle payload's errors list is also skipped."""
    puzzle_path = tmp_path / "seed-001.ipuz"
    puzzle_path.write_text(
        json.dumps(
            {
                "version": "http://ipuz.org/v2",
                "errors": [
                    'LEAK: CAT (1-across) [exact] in clue "A pet cat" '
                    '(offending: "cat")'
                ],
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "leak-batch",
                "results": [
                    {
                        "success": True,
                        "output_path": str(puzzle_path),
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 1,
                    }
                ],
            }
        )
    )
    records = records_from_manifest(manifest_path)
    assert records == []


def test_records_from_manifest_uploads_only_clean_puzzles(tmp_path: Path) -> None:
    """A mixed batch uploads clean puzzles and skips only the leaked one."""
    clean = tmp_path / "seed-001.ipuz"
    clean.write_text(json.dumps({"version": "http://ipuz.org/v2"}))
    leaked = tmp_path / "seed-002.ipuz"
    leaked.write_text(json.dumps({"version": "http://ipuz.org/v2"}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "mixed-batch",
                "results": [
                    {
                        "success": True,
                        "output_path": str(clean),
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 1,
                    },
                    {
                        "success": True,
                        "output_path": str(leaked),
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 2,
                        "error_message": (
                            'LEAK: TRIAD (5-across) [shared_prefix] in clue '
                            '"Trio" (offending: "trio")'
                        ),
                    },
                ],
            }
        )
    )
    records = records_from_manifest(manifest_path)
    assert len(records) == 1
    assert "seed-1" in records[0]["key"]


def test_records_from_manifest_ignores_non_leak_error_message(
    tmp_path: Path,
) -> None:
    """A non-LEAK error_message (e.g. quality threshold) must NOT block upload."""
    puzzle_path = tmp_path / "seed-001.ipuz"
    puzzle_path.write_text(json.dumps({"version": "http://ipuz.org/v2"}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "ok-batch",
                "results": [
                    {
                        "success": True,
                        "output_path": str(puzzle_path),
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 1,
                        "error_message": (
                            "Clue quality below threshold after 3 attempt(s): "
                            "best score 68.0"
                        ),
                    }
                ],
            }
        )
    )
    records = records_from_manifest(manifest_path)
    assert len(records) == 1


def test_records_from_manifest_allow_leaks_override(tmp_path: Path) -> None:
    manifest_path = _leaky_manifest(tmp_path)
    records = records_from_manifest(manifest_path, allow_leaks=True)
    assert len(records) == 1


def test_records_from_manifest_flag_issues_uploads_with_metadata(
    tmp_path: Path,
) -> None:
    """flag_issues uploads the puzzle and attaches structured clue issues."""
    manifest_path = _leaky_manifest(tmp_path)
    records = records_from_manifest(manifest_path, flag_issues=True)

    assert len(records) == 1
    meta = records[0]["metadata"]
    assert meta["review_status"] == "needs_attention"
    issues = meta["clue_issues"]
    assert len(issues) == 1
    assert issues[0]["kind"] == "LEAK"
    assert issues[0]["answer"] == "CAT"
    assert issues[0]["number"] == 1
    assert issues[0]["direction"] == "across"
    assert "A pet cat" in issues[0]["detail"]


def test_flag_issues_parses_duplicate_form() -> None:
    from crossword_generator.data_store import _parse_clue_issue

    issue = _parse_clue_issue(
        'DUPLICATE: EQUAL (6-down) clue "Sweetener brand in blue packets" '
        'already used (existing: "Sweetener brand in blue packets")'
    )
    assert issue["kind"] == "DUPLICATE"
    assert issue["answer"] == "EQUAL"
    assert issue["number"] == 6
    assert issue["direction"] == "down"


def test_allow_leaks_takes_precedence_over_flag_issues(tmp_path: Path) -> None:
    manifest_path = _leaky_manifest(tmp_path)
    records = records_from_manifest(
        manifest_path, allow_leaks=True, flag_issues=True
    )
    assert len(records) == 1
    # allow_leaks short-circuits before flagging, so no issues are attached.
    assert "clue_issues" not in records[0]["metadata"]
    assert records[0]["metadata"]["review_status"] == "unreviewed"


def test_bulk_save_skips_duplicate_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = make_record(
        {},
        game_key="minicrossword",
        puzzle_type="mini",
        size=5,
        difficulty="easy",
        batch_id="batch",
        seed=1,
    )
    calls: list[tuple[str, str]] = []

    def fake_request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        calls.append((method, path))
        if path.endswith("/bulk") or method == "POST":
            raise DataStoreApiError(422, "Duplicate key")
        return {"data": [{"id": 123, "key": record["key"]}]}

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    results = bulk_save_generated_puzzles([record], token="token", sleep_seconds=0)

    assert results[0].action == "skipped_duplicate"
    assert results[0].key == record["key"]
    assert calls[0] == ("POST", "/admin/data-store/records/bulk")
    assert calls[1] == ("POST", "/admin/data-store/records")
    assert calls[2][0] == "GET"


def test_bulk_save_patches_duplicate_records_when_replacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = make_record(
        {},
        game_key="minicrossword",
        puzzle_type="mini",
        size=5,
        difficulty="easy",
        batch_id="batch",
        seed=1,
    )

    def fake_request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        if path.endswith("/bulk") or method == "POST":
            raise DataStoreApiError(422, "Duplicate key")
        if method == "GET":
            return {"data": [{"id": 123, "key": record["key"]}]}
        if method == "PATCH":
            return {"data": {"id": 123, "key": record["key"]}}
        raise AssertionError(f"Unexpected call: {method} {path}")

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    results = bulk_save_generated_puzzles(
        [record],
        replace_existing=True,
        token="token",
        sleep_seconds=0,
    )

    assert results[0].action == "updated"
    assert results[0].response == {"id": 123, "key": record["key"]}


def test_list_generated_puzzle_records_paginates_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        calls.append((method, path))
        query = parse_qs(urlparse(path).query)
        if query.get("page") == ["1"]:
            return {
                "data": [{"id": 1, "key": "one"}],
                "meta": {"current_page": 1, "last_page": 2},
            }
        return {
            "data": [{"id": 2, "key": "two"}],
            "meta": {"current_page": 2, "last_page": 2},
        }

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    records = list_generated_puzzle_records(
        game_key="minicrossword",
        size=7,
        token="token",
    )

    assert [record["key"] for record in records] == ["one", "two"]
    assert calls[0][0] == "GET"
    assert "namespace=crosswords" in calls[0][1]
    assert "collection=generated-puzzles" in calls[0][1]
    assert "game_key=minicrossword" in calls[0][1]
    assert "filters%5Bsize%5D=7" in calls[0][1]


def test_delete_generated_puzzle_records_deletes_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        calls.append((method, path))
        return {}

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    results = delete_generated_puzzle_records(
        [{"id": 123, "key": "old"}],
        token="token",
        sleep_seconds=0,
    )

    assert results[0].action == "deleted"
    assert results[0].key == "old"
    assert calls == [("DELETE", "/admin/data-store/records/123")]


def test_records_from_manifest_skips_duplicate_puzzle(tmp_path: Path) -> None:
    """A DUPLICATE: soft error in the manifest also skips the puzzle."""
    puzzle_path = tmp_path / "seed-001.ipuz"
    puzzle_path.write_text(json.dumps({"version": "http://ipuz.org/v2"}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "dup-batch",
                "results": [
                    {
                        "success": True,
                        "output_path": str(puzzle_path),
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 1,
                        "error_message": (
                            'DUPLICATE: EYE (3-down) clue "Storm center" '
                            'already used (existing: "Storm center")'
                        ),
                    }
                ],
            }
        )
    )
    assert records_from_manifest(manifest_path) == []
    # Override includes it.
    assert len(records_from_manifest(manifest_path, allow_leaks=True)) == 1


def test_fetch_recent_sixty_answers_normalizes_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        calls.append((method, path))
        return {
            "window_days": 180,
            "answers": ["moonwalk ", "JACKPOTS"],
        }

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    answers = fetch_recent_sixty_answers(token="token")

    assert answers == ["MOONWALK", "JACKPOTS"]
    assert calls == [
        (
            "GET",
            "/admin/crossword-puzzles/daily-answers/recent-sixty"
            "?window_days=180",
        )
    ]


def test_fetch_recent_sixty_answers_rejects_bad_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_store,
        "_request_json",
        lambda *args, **kwargs: {"answers": "nope"},
    )

    with pytest.raises(DataStoreError):
        fetch_recent_sixty_answers(token="token")


def test_fetch_recent_daily_answers_normalizes_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        calls.append((method, path))
        return {
            "window_days": 7,
            "first_unscheduled_date": "2026-06-20",
            "since_date": "2026-06-13",
            "answers": ["alpha ", "BRAVO"],
        }

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    recent = fetch_recent_daily_answers(token="token")

    assert recent.answers == ["ALPHA", "BRAVO"]
    assert recent.window_days == 7
    assert recent.first_unscheduled_date == "2026-06-20"
    assert recent.since_date == "2026-06-13"
    assert calls == [
        ("GET", "/admin/crossword-puzzles/daily-answers/recent")
    ]


def test_fetch_recent_daily_answers_passes_window_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_request(
        method: str, path: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append(path)
        return {"window_days": 3, "answers": []}

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    recent = fetch_recent_daily_answers(window_days=3, token="token")

    assert recent.answers == []
    assert calls == [
        "/admin/crossword-puzzles/daily-answers/recent?window_days=3"
    ]


def test_fetch_recent_daily_answers_rejects_bad_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_store,
        "_request_json",
        lambda *args, **kwargs: {"answers": "nope"},
    )

    with pytest.raises(DataStoreError):
        fetch_recent_daily_answers(token="token")
