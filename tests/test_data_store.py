"""Tests for HeyGG data-store record preparation and save behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crossword_generator import data_store
from crossword_generator.data_store import (
    DataStoreApiError,
    DataStoreError,
    bulk_save_generated_puzzles,
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
