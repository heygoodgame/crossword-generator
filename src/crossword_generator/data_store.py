"""Helpers for saving generated puzzle candidates to the HeyGG data store."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE = os.environ.get(
    "HEYGG_API_BASE_URL", "https://id-beta.hey.gg/api"
).rstrip("/")
NAMESPACE = "crosswords"
COLLECTION = "generated-puzzles"
AUTHOR = "crossword-generator"
KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,190}$")
MAX_BULK_RECORDS = 100


class DataStoreError(RuntimeError):
    """Raised when a data-store request or record preparation fails."""


class DataStoreApiError(DataStoreError):
    """Raised when the HeyGG data-store API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HeyGG data-store API returned {status_code}: {body}")


@dataclass(frozen=True)
class SaveResult:
    """Result for one attempted data-store record save."""

    action: str
    key: str
    response: dict[str, Any]


def make_record(
    puzzle: dict[str, Any],
    *,
    game_key: str,
    puzzle_type: str,
    size: int,
    difficulty: str,
    batch_id: str,
    seed: str | int,
    generator_version: str | None = None,
    generator_commit: str | None = None,
    fill_score: float | None = None,
    clue_score: float | None = None,
    title: str | None = None,
    title_reasoning: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Build a generated-puzzle data-store record."""
    record_key = key or (
        f"generated:{game_key}:{batch_id}:{difficulty}:{size}x{size}:seed-{seed}"
    )
    record = {
        "namespace": NAMESPACE,
        "collection": COLLECTION,
        "game_key": game_key,
        "key": record_key,
        "data": puzzle,
        "metadata": {
            "review_status": "unreviewed",
            "puzzle_type": puzzle_type,
            "size": size,
            "difficulty": difficulty,
            "batch_id": batch_id,
            "seed": str(seed),
            "generator_version": generator_version,
            "generator_commit": generator_commit,
            "fill_score": fill_score,
            "clue_score": clue_score,
            "title": title,
            "title_reasoning": title_reasoning,
            "author": AUTHOR,
            "publication_status": "draft",
        },
        "status": "draft",
    }
    validate_record(record)
    return record


def records_from_manifest(
    manifest_path: Path,
    *,
    batch_id: str | None = None,
    generator_version: str | None = None,
    generator_commit: str | None = None,
    mini_game_key: str = "minicrossword",
    midi_game_key: str = "midicrossword",
) -> list[dict[str, Any]]:
    """Build data-store records from a generated batch manifest."""
    manifest = json.loads(manifest_path.read_text())
    resolved_batch_id = batch_id or str(
        manifest.get("batch") or manifest_path.parent.name
    )
    records: list[dict[str, Any]] = []

    for result in manifest.get("results", []):
        if not result.get("success"):
            continue

        output_path = Path(str(result["output_path"]))
        if not output_path.is_absolute():
            output_path = manifest_path.parent / output_path
        if not output_path.exists():
            raise DataStoreError(f"Generated puzzle file not found: {output_path}")

        puzzle = json.loads(output_path.read_text())
        size = int(result["size"])
        puzzle_type = "mini" if size in (5, 7) else "midi"
        game_key = mini_game_key if puzzle_type == "mini" else midi_game_key

        records.append(
            make_record(
                puzzle,
                game_key=game_key,
                puzzle_type=puzzle_type,
                size=size,
                difficulty=str(result["difficulty"]),
                batch_id=resolved_batch_id,
                seed=str(result["seed"]),
                generator_version=generator_version,
                generator_commit=generator_commit,
                fill_score=_optional_float(result.get("fill_score")),
                clue_score=_optional_float(result.get("clue_score")),
                title=_optional_str(result.get("title")),
                title_reasoning=_optional_str(
                    result.get("title_reasoning")
                ),
            )
        )

    return records


def bulk_save_generated_puzzles(
    records: list[dict[str, Any]],
    *,
    replace_existing: bool = False,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    sleep_seconds: float = 0.2,
) -> list[SaveResult]:
    """Create generated-puzzle records, with duplicate-safe fallback handling."""
    results: list[SaveResult] = []
    for record in records:
        validate_record(record)

    for i in range(0, len(records), MAX_BULK_RECORDS):
        chunk = records[i : i + MAX_BULK_RECORDS]
        try:
            response = _request_json(
                "POST",
                "/admin/data-store/records/bulk",
                {"records": chunk},
                api_base=api_base,
                token=token,
                timeout=timeout,
            )
        except DataStoreApiError as exc:
            if exc.status_code != 422:
                raise
            results.extend(
                save_generated_puzzle(
                    record,
                    replace_existing=replace_existing,
                    api_base=api_base,
                    token=token,
                    timeout=timeout,
                )
                for record in chunk
            )
        else:
            for record, saved in zip(chunk, response.get("data", []), strict=False):
                results.append(
                    SaveResult(
                        action="created",
                        key=str(record["key"]),
                        response=_ensure_dict(saved),
                    )
                )

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return results


def save_generated_puzzle(
    record: dict[str, Any],
    *,
    replace_existing: bool = False,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> SaveResult:
    """Create one generated-puzzle record, skipping or patching duplicates."""
    validate_record(record)
    try:
        response = _request_json(
            "POST",
            "/admin/data-store/records",
            record,
            api_base=api_base,
            token=token,
            timeout=timeout,
        )
    except DataStoreApiError as exc:
        if exc.status_code != 422:
            raise
        existing = find_existing_record(
            namespace=str(record["namespace"]),
            collection=str(record["collection"]),
            game_key=str(record["game_key"]),
            key=str(record["key"]),
            api_base=api_base,
            token=token,
            timeout=timeout,
        )
        if existing is None:
            raise
        if not replace_existing:
            return SaveResult(
                action="skipped_duplicate",
                key=str(record["key"]),
                response=existing,
            )

        record_id = existing.get("id")
        if record_id is None:
            raise DataStoreError(
                f"Cannot replace duplicate record without an id: {record['key']}"
            )
        patched = _request_json(
            "PATCH",
            f"/admin/data-store/records/{record_id}",
            record,
            api_base=api_base,
            token=token,
            timeout=timeout,
        )
        return SaveResult(
            action="updated",
            key=str(record["key"]),
            response=_ensure_dict(patched.get("data", patched)),
        )

    return SaveResult(
        action="created",
        key=str(record["key"]),
        response=_ensure_dict(response.get("data", response)),
    )


def find_existing_record(
    *,
    namespace: str,
    collection: str,
    game_key: str,
    key: str,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> dict[str, Any] | None:
    """Find an existing generated-puzzle data-store record by identity fields."""
    query = urlencode(
        {
            "namespace": namespace,
            "collection": collection,
            "game_key": game_key,
            "key": key,
        }
    )
    response = _request_json(
        "GET",
        f"/admin/data-store/records?{query}",
        api_base=api_base,
        token=token,
        timeout=timeout,
    )
    data = response.get("data", [])
    if isinstance(data, dict):
        nested = data.get("data")
        if isinstance(nested, list):
            data = nested
        else:
            data = [data]
    if not isinstance(data, list):
        raise DataStoreError(f"Unexpected list response shape: {response}")
    return _ensure_dict(data[0]) if data else None


def validate_record(record: dict[str, Any]) -> None:
    """Validate local constraints before sending a record to the API."""
    key = record.get("key")
    if not isinstance(key, str) or KEY_PATTERN.fullmatch(key) is None:
        raise DataStoreError(f"Invalid data-store key: {key!r}")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise DataStoreError("Record metadata must be a JSON object")

    data_bytes = len(json.dumps(record.get("data", {})).encode())
    metadata_bytes = len(json.dumps(metadata).encode())
    if data_bytes > 1_000_000:
        raise DataStoreError(f"Record data exceeds 1 MB: {data_bytes} bytes")
    if metadata_bytes > 64_000:
        raise DataStoreError(
            f"Record metadata exceeds 64 KB: {metadata_bytes} bytes"
        )


def _request_json(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    resolved_api_base = (api_base or API_BASE).rstrip("/")
    resolved_token = token or os.environ.get("HEYGG_ADMIN_TOKEN") or os.environ[
        "HEYGG_ADMIN_API_TOKEN"
    ]
    url = f"{resolved_api_base}{path}"
    headers = {
        "Authorization": f"Bearer {resolved_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    encoded = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=encoded, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode()
    except HTTPError as exc:
        error_body = exc.read().decode()
        raise DataStoreApiError(exc.code, error_body) from exc

    if not response_body:
        return {}
    parsed = json.loads(response_body)
    return _ensure_dict(parsed)


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ensure_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataStoreError(f"Unexpected object response shape: {value!r}")
    return value
