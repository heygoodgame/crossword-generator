"""Helpers for saving generated puzzle candidates to the HeyGG data store."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

API_BASE = os.environ.get("HEYGG_API_BASE_URL", "https://play.hey.gg/api").rstrip("/")

# Checked in order; the service-account token wins over personal admin JWTs.
ADMIN_TOKEN_ENV_VARS = (
    "HEYGG_CROSSWORD_GENERATOR_TOKEN",
    "HEYGG_ADMIN_TOKEN",
    "HEYGG_ADMIN_API_TOKEN",
)


def resolve_admin_token() -> str | None:
    """Return the first non-empty admin token from ADMIN_TOKEN_ENV_VARS."""
    for name in ADMIN_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None
NAMESPACE = "crosswords"
COLLECTION = "generated-puzzles"
UNLIMITED_COLLECTION = "unlimited-pool"
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


@dataclass(frozen=True)
class DeleteResult:
    """Result for one attempted data-store record delete."""

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
    clue_issues: list[dict[str, Any]] | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Build a generated-puzzle data-store record.

    ``clue_issues`` carries any LEAK/DUPLICATE clue problems that survived
    repair so the admin review UI can flag the specific clues for the editor
    instead of the puzzle being silently held back from upload.
    """
    record_key = key or (
        f"generated:{game_key}:{batch_id}:{difficulty}:{size}x{size}:seed-{seed}"
    )
    issues = clue_issues or []
    metadata: dict[str, Any] = {
        # A flagged puzzle still needs review, but route it so the UI can
        # surface it first: needs_attention is the editor's "look here" signal.
        "review_status": "needs_attention" if issues else "unreviewed",
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
    }
    if issues:
        metadata["clue_issues"] = issues
    record = {
        "namespace": NAMESPACE,
        "collection": COLLECTION,
        "game_key": game_key,
        "key": record_key,
        "data": puzzle,
        "metadata": metadata,
        "status": "draft",
    }
    validate_record(record)
    return record


# Soft-error prefixes that block a puzzle from uploading: an answer-leaking
# clue, a clue that exactly duplicates an existing one, or a clue the
# fact-checker still flags incorrect after repair — where repair could not fix
# it. All leave the puzzle saved but held back from upload.
_BLOCKING_ERROR_PREFIXES = ("LEAK:", "DUPLICATE:", "FACT:")


def _blocking_errors(result: dict[str, Any], puzzle: dict[str, Any]) -> list[str]:
    """Return any blocking soft errors (LEAK:/DUPLICATE:) for a generated puzzle.

    The exported ``.ipuz`` file does not carry the envelope's ``errors`` field,
    so the authoritative source is the manifest result's ``error_message``
    (a "; "-joined string of the envelope errors). We also check the puzzle
    payload's ``errors`` list in case a caller passes a full envelope.
    """

    def _is_blocking(text: str) -> bool:
        return text.startswith(_BLOCKING_ERROR_PREFIXES)

    found: list[str] = []
    message = result.get("error_message")
    if message:
        found.extend(
            part.strip()
            for part in str(message).split("; ")
            if _is_blocking(part.strip())
        )
    for e in puzzle.get("errors") or []:
        if _is_blocking(str(e)):
            found.append(str(e))
    return found


# Parse a soft-error string like:
#   LEAK: ROB (17-down) [shared_prefix] in clue "Common nickname for Robert" ...
#   DUPLICATE: EQUAL (6-down) clue "Sweetener brand..." already used (existing: ...)
#   FACT: ELENA (10-across) clue "Actress Longoria..." flagged incorrect: ...
_CLUE_ISSUE_RE = re.compile(
    r"^(?P<kind>LEAK|DUPLICATE|FACT):\s*(?P<answer>\S+)\s*"
    r"\((?P<number>\d+)-(?P<direction>across|down)\)",
    re.IGNORECASE,
)


def _parse_clue_issue(text: str) -> dict[str, Any]:
    """Turn a LEAK/DUPLICATE soft-error string into a structured issue.

    Always returns a dict; if the prefix can't be parsed past the kind, the
    full message is preserved under ``detail`` so nothing is lost in the UI.
    """
    match = _CLUE_ISSUE_RE.match(text.strip())
    if not match:
        kind = text.split(":", 1)[0].strip().upper() if ":" in text else "ISSUE"
        return {"kind": kind, "detail": text.strip()}
    return {
        "kind": match.group("kind").upper(),
        "answer": match.group("answer").upper(),
        "number": int(match.group("number")),
        "direction": match.group("direction").lower(),
        "detail": text.strip(),
    }


def records_from_manifest(
    manifest_path: Path,
    *,
    batch_id: str | None = None,
    generator_version: str | None = None,
    generator_commit: str | None = None,
    mini_game_key: str = "minicrossword",
    midi_game_key: str = "midicrossword",
    allow_leaks: bool = False,
    flag_issues: bool = False,
) -> list[dict[str, Any]]:
    """Build data-store records from a generated batch manifest.

    A puzzle carrying a blocking soft error that survived repair — a ``LEAK:``
    (a clue echoing its answer) or a ``DUPLICATE:`` (a clue matching one already
    in use) — is handled one of three ways:

    - default: held back from upload and logged (the rest of the batch proceeds);
    - ``flag_issues``: uploaded with the issues attached to
      ``metadata.clue_issues`` and ``review_status=needs_attention`` so the admin
      UI surfaces the specific clues for the editor to fix;
    - ``allow_leaks``: uploaded with no flagging (legacy override).

    ``allow_leaks`` takes precedence over ``flag_issues``.
    """
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
        clue_issues: list[dict[str, Any]] = []
        if not allow_leaks:
            blocking = _blocking_errors(result, puzzle)
            if blocking:
                if not flag_issues:
                    logger.warning(
                        "Skipping %s: clue issue(s) survived repair — %s "
                        "(pass --flag-issues to upload with the clues flagged "
                        "for review, or --allow-leaks to include silently).",
                        output_path.name,
                        "; ".join(blocking),
                    )
                    continue
                clue_issues = [_parse_clue_issue(b) for b in blocking]
                logger.warning(
                    "Flagging %s for review: %d clue issue(s) — %s",
                    output_path.name,
                    len(clue_issues),
                    "; ".join(blocking),
                )
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
                title_reasoning=_optional_str(result.get("title_reasoning")),
                clue_issues=clue_issues or None,
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


def save_data_store_record(
    record: dict[str, Any],
    *,
    replace_existing: bool = False,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> SaveResult:
    """Create one data-store record, skipping or patching duplicate keys."""
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


def request_admin_json(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Send an authenticated admin API JSON request."""
    return _request_json(
        method,
        path,
        body,
        api_base=api_base,
        token=token,
        timeout=timeout,
    )


def list_generated_puzzle_records(
    *,
    game_key: str,
    size: int | None = None,
    difficulty: str | None = None,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """List generated-puzzle records, paging through the admin API."""
    filters: dict[str, str | int] = {
        "namespace": NAMESPACE,
        "collection": COLLECTION,
        "game_key": game_key,
        "per_page": per_page,
    }
    if size is not None:
        filters["filters[size]"] = size
    if difficulty is not None:
        filters["filters[difficulty]"] = difficulty

    records: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urlencode({**filters, "page": page})
        response = _request_json(
            "GET",
            f"/admin/data-store/records?{query}",
            api_base=api_base,
            token=token,
            timeout=timeout,
        )
        data = response.get("data", [])
        if not isinstance(data, list):
            raise DataStoreError(f"Unexpected list response shape: {response}")
        records.extend(_ensure_dict(record) for record in data)

        meta = response.get("meta", {})
        if not isinstance(meta, dict):
            break
        current_page = int(meta.get("current_page", page))
        last_page = int(meta.get("last_page", current_page))
        if current_page >= last_page:
            break
        page = current_page + 1

    return records


def list_unlimited_puzzle_records(
    *,
    game_key: str,
    size: int | None = None,
    difficulty: str | None = None,
    status: str | None = "active",
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """List published unlimited-pool records, paging through the admin API."""
    records = list_official_puzzle_records(
        game_key=game_key,
        collection=UNLIMITED_COLLECTION,
        status=status,
        api_base=api_base,
        token=token,
        timeout=timeout,
        per_page=per_page,
    )

    return [
        record
        for record in records
        if _record_metadata_matches(record, size=size, difficulty=difficulty)
    ]


def list_official_puzzle_records(
    *,
    game_key: str,
    collection: str | None = None,
    status: str | None = None,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """List official (published/scheduled daily) puzzle records.

    These live in ``crosswords/daily-schedule`` and carry the full IPUZ under
    ``data.puzzle``. This is the live clue corpus solvers actually see — the
    source for cross-puzzle clue de-duplication. The draft
    ``crosswords/generated-puzzles`` store is emptied as candidates are
    promoted, so it is not a usable clue history on its own.
    """
    filters: dict[str, str | int] = {
        "game_key": game_key,
        "per_page": per_page,
    }
    if collection is not None:
        filters["collection"] = collection
    if status is not None:
        filters["status"] = status

    records: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urlencode({**filters, "page": page})
        response = _request_json(
            "GET",
            f"/admin/crossword-puzzles/official?{query}",
            api_base=api_base,
            token=token,
            timeout=timeout,
        )
        data = response.get("data", [])
        if not isinstance(data, list):
            raise DataStoreError(f"Unexpected official list shape: {response}")
        records.extend(_ensure_dict(record) for record in data)

        meta = response.get("meta")
        if not isinstance(meta, dict):
            # No pagination metadata: a single full page was returned.
            break
        current_page = int(meta.get("current_page", page))
        last_page = int(meta.get("last_page", current_page))
        if current_page >= last_page:
            break
        page = current_page + 1

    return records


def _record_metadata_matches(
    record: dict[str, Any],
    *,
    size: int | None = None,
    difficulty: str | None = None,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return size is None and difficulty is None
    if size is not None and str(metadata.get("size")) != str(size):
        return False
    if difficulty is not None and str(metadata.get("difficulty")) != difficulty:
        return False
    return True


def fetch_recent_sixty_answers(
    *,
    window_days: int = 180,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> list[str]:
    """Fetch HGG 60 answers used in scheduled dailies from the admin API.

    Returns normalized (uppercase) answers scheduled from ``window_days`` ago
    through all scheduled future days. Hard batches exclude these from the
    hgg-60 fill pool so candidates stay schedulable under the 180-day
    no-repeat window for 60-point entries.
    """
    query = urlencode({"window_days": window_days})
    response = _request_json(
        "GET",
        f"/admin/crossword-puzzles/daily-answers/recent-sixty?{query}",
        api_base=api_base,
        token=token,
        timeout=timeout,
    )
    answers = response.get("answers", [])
    if not isinstance(answers, list):
        raise DataStoreError(f"Unexpected recent-sixty response shape: {response}")
    return [str(answer).strip().upper() for answer in answers]


@dataclass(frozen=True)
class RecentDailyAnswers:
    """Recently scheduled daily answers for generation-time exclusion."""

    answers: list[str]
    window_days: int
    first_unscheduled_date: str | None
    since_date: str | None
    forward_days: int | None = None
    until_date: str | None = None
    # Per-answer count of scheduled daily slots (all games/tracks) that used
    # the answer over ``count_window_days``. Empty when the server predates
    # the counts extension; callers must degrade gracefully.
    counts: dict[str, int] = field(default_factory=dict)
    count_window_days: int | None = None


def fetch_recent_daily_answers(
    *,
    window_days: int | None = None,
    forward_days: int | None = None,
    count_window_days: int | None = None,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> RecentDailyAnswers:
    """Fetch answers used in recently scheduled dailies from the admin API.

    The server returns distinct answers scheduled within a bounded window
    around the first unscheduled daily slot: ``window_days`` (server default
    7) back through ``forward_days`` (server default 13) ahead. Batches
    exclude these from fill pools so new candidates don't collide with the
    +/-6-day no-repeat rule when scheduled.

    When the server supports it, ``count_window_days`` (server default 90)
    selects the lookback for the ``counts`` object — how many scheduled daily
    slots used each answer — which the batch runner turns into a soft
    usage penalty during fill. Older servers omit ``counts``; the returned
    dict is then empty.
    """
    path = "/admin/crossword-puzzles/daily-answers/recent"
    params: dict[str, int] = {}
    if window_days is not None:
        params["window_days"] = window_days
    if forward_days is not None:
        params["forward_days"] = forward_days
    if count_window_days is not None:
        params["count_window_days"] = count_window_days
    if params:
        path += f"?{urlencode(params)}"
    response = _request_json(
        "GET",
        path,
        api_base=api_base,
        token=token,
        timeout=timeout,
    )
    answers = response.get("answers", [])
    if not isinstance(answers, list):
        raise DataStoreError(
            f"Unexpected recent daily answers response shape: {response}"
        )
    forward_days_value = response.get("forward_days")
    raw_counts = response.get("counts")
    counts: dict[str, int] = {}
    if isinstance(raw_counts, dict):
        for answer, count in raw_counts.items():
            key = str(answer).strip().upper()
            try:
                value = int(count)
            except (TypeError, ValueError):
                continue
            if key and value > 0:
                counts[key] = value
    count_window_value = response.get("count_window_days")
    return RecentDailyAnswers(
        answers=[str(answer).strip().upper() for answer in answers],
        window_days=int(response.get("window_days", window_days or 0)),
        first_unscheduled_date=response.get("first_unscheduled_date"),
        since_date=response.get("since_date"),
        forward_days=(
            int(forward_days_value) if forward_days_value is not None else None
        ),
        until_date=response.get("until_date"),
        counts=counts,
        count_window_days=(
            int(count_window_value) if count_window_value is not None else None
        ),
    )


def delete_generated_puzzle_records(
    records: list[dict[str, Any]],
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    sleep_seconds: float = 0.1,
) -> list[DeleteResult]:
    """Delete generated-puzzle records by id through the admin API."""
    results: list[DeleteResult] = []
    for record in records:
        record_id = record.get("id")
        if record_id is None:
            raise DataStoreError(f"Cannot delete record without id: {record}")
        _request_json(
            "DELETE",
            f"/admin/data-store/records/{record_id}",
            api_base=api_base,
            token=token,
            timeout=timeout,
        )
        results.append(
            DeleteResult(
                action="deleted",
                key=str(record.get("key", record_id)),
                response=_ensure_dict(record),
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
    return save_data_store_record(
        record,
        replace_existing=replace_existing,
        api_base=api_base,
        token=token,
        timeout=timeout,
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
        raise DataStoreError(f"Record metadata exceeds 64 KB: {metadata_bytes} bytes")


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
    resolved_token = token or resolve_admin_token()
    if not resolved_token:
        raise KeyError(
            "HEYGG_CROSSWORD_GENERATOR_TOKEN (or HEYGG_ADMIN_TOKEN / "
            "HEYGG_ADMIN_API_TOKEN) must be set."
        )
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
