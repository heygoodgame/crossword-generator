"""Pull crossword_lists state from hey-you and write it back to the
committed `dictionaries/*.txt` files.

Workflow:

    Jeff edits in /admin/lists -> rows live in hey-you's MySQL.
    Operator runs `crossword-generator consolidate-list [slug]` ->
    this module fetches the current state, writes the .txt file, and
    pokes hey-you's mark-consolidated endpoint so /admin/lists shows
    whether the operator has caught up.

The on-disk .txt files remain authoritative for the generator. git diff
is the operator's review surface; this command is the only path that
mutates those files automatically.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from crossword_generator.data_store import resolve_admin_token

logger = logging.getLogger(__name__)

API_BASE = os.environ.get(
    "HEYGG_API_BASE_URL", "https://play.hey.gg/api"
).rstrip("/")


class ConsolidateError(RuntimeError):
    """Raised when consolidate-list cannot complete a request."""


@dataclass(frozen=True)
class ListSummary:
    """Result of consolidating one master list."""

    slug: str
    file_path: str
    added: int
    removed: int
    total_after: int
    wrote: bool


def list_registered_lists(
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> list[dict]:
    """Return every registered crossword list (slug, file_path, etc.)."""
    response = _request(
        "GET",
        "/admin/crossword-lists",
        api_base=api_base,
        token=token,
        timeout=timeout,
    )
    data = response.get("data", [])
    if not isinstance(data, list):
        raise ConsolidateError(f"Unexpected listsIndex response: {response}")
    return [item for item in data if isinstance(item, dict)]


def fetch_list_metadata(
    slug: str,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> dict:
    """Return the registered list row (file_path, format, scope, etc.)."""
    response = _request(
        "GET",
        f"/admin/crossword-lists/{slug}",
        api_base=api_base,
        token=token,
        timeout=timeout,
    )
    if not isinstance(response, dict) or "slug" not in response:
        raise ConsolidateError(f"Unexpected listsShow response for {slug}: {response}")
    return response


def fetch_list_contents(
    slug: str,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 300,
) -> str:
    """Stream the current .txt body for `slug` (formatted per the list's format)."""
    return _request(
        "GET",
        f"/admin/crossword-lists/{slug}/download",
        api_base=api_base,
        token=token,
        timeout=timeout,
        as_text=True,
    )


def mark_consolidated(
    slug: str,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> None:
    """Stamp last_consolidated_at on the server so /admin/lists can show staleness."""
    _request(
        "POST",
        f"/admin/crossword-lists/{slug}/mark-consolidated",
        api_base=api_base,
        token=token,
        timeout=timeout,
    )


def diff_words(old: str, new: str) -> tuple[set[str], set[str]]:
    """Return (added, removed) keyed by the WORD-before-semicolon on each line.

    Comparing by leading-word ignores per-line score / note changes — those
    aren't interesting in a consolidation diff (they're routine edits, not
    additions/removals). Operators who want the byte diff can still `git diff`.
    """
    old_words = _words_from(old)
    new_words = _words_from(new)
    return new_words - old_words, old_words - new_words


def consolidate_one(
    slug: str,
    project_root: Path,
    *,
    api_base: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    timeout: int = 300,
) -> ListSummary:
    """Sync one slug's .txt file from hey-you. Returns a summary for printing."""
    metadata = fetch_list_metadata(
        slug, api_base=api_base, token=token, timeout=timeout
    )
    file_path = str(metadata.get("file_path") or "")
    if not file_path:
        raise ConsolidateError(f"List {slug!r} has no file_path registered.")

    target = (
        project_root / file_path
        if not Path(file_path).is_absolute()
        else Path(file_path)
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    new_body, repeated = dedupe_lines(
        fetch_list_contents(slug, api_base=api_base, token=token, timeout=timeout)
    )
    if repeated:
        logger.warning(
            "%s: download repeated %d row(s) (hey-you streams the list with "
            "orderBy('word')->chunkById(), which overlaps at chunk boundaries); "
            "kept the first occurrence of each.",
            slug,
            repeated,
        )
    old_body = target.read_text() if target.exists() else ""
    added, removed = diff_words(old_body, new_body)
    total_after = len(_words_from(new_body))

    wrote = False
    if not dry_run:
        if new_body != old_body:
            target.write_text(new_body)
            wrote = True
        mark_consolidated(slug, api_base=api_base, token=token)

    return ListSummary(
        slug=slug,
        file_path=file_path,
        added=len(added),
        removed=len(removed),
        total_after=total_after,
        wrote=wrote,
    )


def dedupe_lines(body: str) -> tuple[str, int]:
    """Drop repeated lines from a downloaded list body, keeping first occurrences.

    Returns the cleaned body and how many lines were dropped. Comparison is
    on the whole line, so a word whose score or note changed is not treated
    as a repeat of its old row.
    """
    seen: set[str] = set()
    kept: list[str] = []
    repeated = 0
    for line in body.splitlines():
        if line.strip() and line in seen:
            repeated += 1
            continue
        seen.add(line)
        kept.append(line)
    cleaned = "\n".join(kept)
    if body.endswith("\n"):
        cleaned += "\n"
    return cleaned, repeated


def _words_from(body: str) -> set[str]:
    return {
        line.split(";", 1)[0].strip().upper()
        for line in body.splitlines()
        if line.strip()
    }


def _request(
    method: str,
    path: str,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    as_text: bool = False,
):
    resolved_api_base = (api_base or API_BASE).rstrip("/")
    resolved_token = token or resolve_admin_token()
    if not resolved_token:
        raise ConsolidateError(
            "HEYGG_CROSSWORD_GENERATOR_TOKEN (or HEYGG_ADMIN_TOKEN / "
            "HEYGG_ADMIN_API_TOKEN) must be set."
        )
    url = f"{resolved_api_base}{path}"
    headers = {
        "Authorization": f"Bearer {resolved_token}",
        "Accept": "application/json" if not as_text else "text/plain",
    }
    request = Request(url, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
    except HTTPError as exc:
        raise ConsolidateError(
            f"{method} {path} failed ({exc.code}): {exc.read().decode()}"
        ) from exc

    if as_text:
        return body
    if not body:
        return {}
    return json.loads(body)
