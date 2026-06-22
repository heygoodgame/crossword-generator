"""Backfill clue hints for already-uploaded crossword puzzles.

Adds an easy alternate clue ("hint") to every clue that lacks one, across both
the draft candidate store (``crosswords/generated-puzzles``) and the live
published puzzles (``/admin/crossword-puzzles/official``). Hints are written as
the spec ipuz dict form ``{number, clue, hints:[hint]}``; clues that already
have a hint are left untouched, so the script is idempotent and re-runnable.

Two LLM modes:

- default (synchronous): generate hints one puzzle at a time. Simple; good for
  small/targeted runs and for eyeballing output.
- ``--batch-api``: submit every puzzle as one Anthropic Message Batches request
  with a shared cached system prompt. Latency-tolerant (minutes-to-hours) and
  far cheaper at scale — use for the full archive.

Every generated hint is screened with the same mechanical leak detector the
clue pipeline uses; a hint that leaks its answer is dropped (the clue stays
hint-less) rather than written.

Auth: needs a prod admin token in ``HEYGG_ADMIN_API_TOKEN`` (or
``HEYGG_ADMIN_TOKEN``) and ``HEYGG_API_BASE_URL`` (default https://play.hey.gg/api).
Always run ``--dry-run`` first.

Examples::

    # Dry run over published mini puzzles, synchronous LLM
    uv run python scripts/backfill_hints.py \
        --target official --game-key minicrossword --dry-run

    # Full archive via the Batches API
    uv run python scripts/backfill_hints.py \
        --target all --game-key minicrossword --game-key midicrossword \
        --batch-api
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from crossword_generator.dictionary import Dictionary
from crossword_generator.exporters.numbering import NumberedEntry
from crossword_generator.graders.leak_detector import detect_leak
from crossword_generator.llm.prompts.hint_generation import (
    build_hint_generation_messages,
    parse_hint_response,
)

logger = logging.getLogger("backfill_hints")

DEFAULT_API_BASE = "https://play.hey.gg/api"
DEFAULT_MODEL = "claude-sonnet-4-6"

try:
    import anthropic
    from anthropic.types.message_create_params import (
        MessageCreateParamsNonStreaming,
    )
    from anthropic.types.messages.batch_create_params import Request
except ImportError:  # pragma: no cover - exercised only without the SDK
    anthropic = None


# --------------------------------------------------------------------------- #
# ipuz clue helpers — read/write both list and dict clue forms
# --------------------------------------------------------------------------- #


def _clue_number(clue: Any) -> int | None:
    if isinstance(clue, list) and clue:
        return int(clue[0])
    if isinstance(clue, dict) and "number" in clue:
        return int(clue["number"])
    return None


def _clue_text(clue: Any) -> str:
    if isinstance(clue, list) and len(clue) >= 2:
        return str(clue[1])
    if isinstance(clue, dict):
        return str(clue.get("clue", ""))
    return ""


def _clue_has_hint(clue: Any) -> bool:
    if isinstance(clue, list):
        return len(clue) >= 3 and bool(clue[2])
    if isinstance(clue, dict):
        hints = clue.get("hints")
        return isinstance(hints, list) and bool(hints) and bool(hints[0])
    return False


def _with_hint(clue: Any, hint: str) -> dict[str, Any]:
    """Return the dict-form clue carrying the hint (spec-compliant)."""
    return {"number": _clue_number(clue), "clue": _clue_text(clue), "hints": [hint]}


def _grid_dims(ipuz: dict[str, Any]) -> tuple[int, int]:
    dims = ipuz.get("dimensions") or {}
    return int(dims.get("width", 0)), int(dims.get("height", 0))


def _entries_by_key(ipuz: dict[str, Any]) -> dict[tuple[int, str], NumberedEntry]:
    """Map (number, direction) -> NumberedEntry by walking the solution grid.

    Mirrors the game's parser: numbered cells start an across run (no filled
    cell to the left) and/or a down run (none above).
    """
    width, height = _grid_dims(ipuz)
    puzzle = ipuz.get("puzzle") or []
    solution = ipuz.get("solution") or []

    def filled(grid: list, r: int, c: int) -> str | None:
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[r]):
            return None
        v = grid[r][c]
        return None if (v == "#" or v is None) else v

    number_pos: dict[int, tuple[int, int]] = {}
    for r in range(height):
        for c in range(width):
            cell = puzzle[r][c] if r < len(puzzle) and c < len(puzzle[r]) else None
            if isinstance(cell, int) and cell > 0:
                number_pos[cell] = (r, c)

    out: dict[tuple[int, str], NumberedEntry] = {}
    for num, (r, c) in number_pos.items():
        if filled(solution, r, c) and filled(solution, r, c - 1) is None:
            word = ""
            cc = c
            while filled(solution, r, cc) is not None:
                word += filled(solution, r, cc)
                cc += 1
            if len(word) >= 2:
                out[(num, "across")] = NumberedEntry(
                    number=num, direction="across", row=r, col=c,
                    length=len(word), answer=word.upper(),
                )
        if filled(solution, r, c) and filled(solution, r - 1, c) is None:
            word = ""
            rr = r
            while filled(solution, rr, c) is not None:
                word += filled(solution, rr, c)
                rr += 1
            if len(word) >= 2:
                out[(num, "down")] = NumberedEntry(
                    number=num, direction="down", row=r, col=c,
                    length=len(word), answer=word.upper(),
                )
    return out


# --------------------------------------------------------------------------- #
# Target model — one puzzle to backfill, abstracting candidate vs official
# --------------------------------------------------------------------------- #


@dataclass
class Target:
    """A single puzzle needing hints, with everything to read and write it."""

    kind: str  # "candidate" | "official"
    record_id: str
    label: str
    ipuz: dict[str, Any]
    # Path into the record's data where the ipuz lives, for candidate patches.
    ipuz_container: dict[str, Any] = field(default_factory=dict)
    ipuz_key: str = ""

    def entries_needing_hints(self) -> list[NumberedEntry]:
        by_key = _entries_by_key(self.ipuz)
        entries: list[NumberedEntry] = []
        for direction in ("Across", "Down"):
            for clue in self.ipuz.get("clues", {}).get(direction, []):
                num = _clue_number(clue)
                if num is None or _clue_has_hint(clue):
                    continue
                entry = by_key.get((num, direction.lower()))
                if entry is not None:
                    entries.append(entry)
        return entries

    def clues_by_key(self) -> dict[tuple[int, str], str]:
        out: dict[tuple[int, str], str] = {}
        for direction in ("Across", "Down"):
            for clue in self.ipuz.get("clues", {}).get(direction, []):
                num = _clue_number(clue)
                if num is not None:
                    out[(num, direction.lower())] = _clue_text(clue)
        return out

    def apply_hints(self, hints: dict[tuple[int, str], str]) -> int:
        """Write hints into this target's ipuz in place. Returns count applied."""
        applied = 0
        for direction in ("Across", "Down"):
            d = direction.lower()
            clues = self.ipuz.get("clues", {}).get(direction, [])
            for i, clue in enumerate(clues):
                num = _clue_number(clue)
                if num is None or _clue_has_hint(clue):
                    continue
                hint = hints.get((num, d))
                if hint:
                    clues[i] = _with_hint(clue, hint)
                    applied += 1
        return applied


# --------------------------------------------------------------------------- #
# Loading targets from the admin API
# --------------------------------------------------------------------------- #


def _extract_candidate_ipuz(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    """Find the editable ipuz in a generated-puzzle record's data.

    Prefers the admin overlay copy (what the game serves) so a backfill shows up
    in the editor too; falls back to the base payload. Returns
    (ipuz, container, key) so the caller can patch the right slot.
    """
    overlay = data.get("hgg_admin_edit")
    if isinstance(overlay, dict) and isinstance(overlay.get("puzzle"), dict):
        return overlay["puzzle"], overlay, "puzzle"
    if isinstance(data.get("ipuz"), dict):
        return data["ipuz"], data, "ipuz"
    if "clues" in data and "solution" in data:
        return data, data, ""  # ipuz stored at top level
    if isinstance(data.get("puzzle"), dict):
        return data["puzzle"], data, "puzzle"
    return None


def load_candidate_targets(game_key: str, api_base: str, token: str) -> list[Target]:
    from crossword_generator.data_store import list_generated_puzzle_records

    records = list_generated_puzzle_records(
        game_key=game_key, api_base=api_base, token=token
    )
    targets: list[Target] = []
    for rec in records:
        data = rec.get("data") or {}
        found = _extract_candidate_ipuz(data)
        if not found:
            continue
        ipuz, container, key = found
        targets.append(
            Target(
                kind="candidate",
                record_id=str(rec.get("id")),
                label=str(rec.get("key") or rec.get("id")),
                ipuz=ipuz,
                ipuz_container=container,
                ipuz_key=key,
            )
        )
    return targets


def load_official_targets(
    game_key: str,
    api_base: str,
    token: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    day_from: int | None = None,
    day_to: int | None = None,
) -> list[Target]:
    from crossword_generator.data_store import list_official_puzzle_records

    records = list_official_puzzle_records(
        game_key=game_key, api_base=api_base, token=token
    )
    targets: list[Target] = []
    for rec in records:
        meta = rec.get("metadata") or {}
        date = meta.get("date")
        day = meta.get("day_number")

        # Scope by day_number — the key the game actually serves by. This is
        # authoritative: the client's daily epoch and the stored metadata.date
        # can disagree, so date is NOT a reliable "which week players see"
        # filter. Prefer --day-from/--day-to for daily scoping.
        if day_from is not None or day_to is not None:
            if day is None:
                continue
            if day_from is not None and day < day_from:
                continue
            if day_to is not None and day > day_to:
                continue

        # Optional secondary scope by stored date (rarely what you want — see
        # above). Records without a date (e.g. unlimited) are excluded.
        if date_from or date_to:
            if not date:
                continue
            if date_from and date < date_from:
                continue
            if date_to and date > date_to:
                continue

        data = rec.get("data") or {}
        # Unlimited puzzles serialize ipuz at data; daily at data.puzzle.
        if isinstance(data, dict) and "clues" in data and "solution" in data:
            ipuz = data
        elif isinstance(data.get("puzzle"), dict):
            ipuz = data["puzzle"]
        else:
            continue
        label = str(rec.get("key") or rec.get("id"))
        if day is not None:
            label = f"{label} (day {day})"
        elif date:
            label = f"{label} ({date})"
        targets.append(
            Target(
                kind="official",
                record_id=str(rec.get("id")),
                label=label,
                ipuz=ipuz,
            )
        )
    return targets


# --------------------------------------------------------------------------- #
# Hint generation + leak screening
# --------------------------------------------------------------------------- #


def _screen_hints(
    raw_hints: dict[tuple[int, str], str],
    entries: list[NumberedEntry],
    dictionary: Dictionary | None,
) -> dict[tuple[int, str], str]:
    """Drop any hint that leaks its answer."""
    answers = {(e.number, e.direction): e.answer for e in entries}
    clean: dict[tuple[int, str], str] = {}
    for key, hint in raw_hints.items():
        answer = answers.get(key)
        if not answer or not hint:
            continue
        if detect_leak(answer, hint, dictionary) is not None:
            logger.warning("  dropped leaking hint for %s-%s (%s)", *key, answer)
            continue
        clean[key] = hint
    return clean


def generate_hints_sync(
    target: Target,
    client: anthropic.Anthropic,
    model: str,
    dictionary: Dictionary | None,
) -> dict[tuple[int, str], str]:
    """Generate + screen hints for one target with a single LLM call."""
    entries = target.entries_needing_hints()
    if not entries:
        return {}
    system_text, user_text = build_hint_generation_messages(
        entries, target.clues_by_key()
    )
    resp = client.messages.create(
        model=model,
        system=system_text,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_text}],
    )
    raw = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    try:
        hints = parse_hint_response(raw, entries)
    except Exception as exc:  # noqa: BLE001 - log and skip a bad puzzle
        logger.warning("  parse failed for %s: %s", target.label, exc)
        return {}
    return _screen_hints(hints, entries, dictionary)


# --------------------------------------------------------------------------- #
# Writing results back
# --------------------------------------------------------------------------- #


def write_target(target: Target, api_base: str, token: str) -> None:
    from crossword_generator.data_store import request_admin_json

    if target.kind == "official":
        request_admin_json(
            "PATCH",
            f"/admin/crossword-puzzles/official/{target.record_id}",
            {"puzzle": target.ipuz},
            api_base=api_base,
            token=token,
        )
        return

    # Candidate: PATCH the data-store record. The ipuz was mutated in place
    # inside its container, so re-send the full data payload.
    request_admin_json(
        "PATCH",
        f"/admin/data-store/records/{target.record_id}",
        {"data": _candidate_data_for(target)},
        api_base=api_base,
        token=token,
    )


def _candidate_data_for(target: Target) -> dict[str, Any]:
    # ipuz_container is the dict that holds the ipuz (or IS the ipuz when
    # stored at top level). For a top-level ipuz the container is the data dict.
    return target.ipuz_container


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@dataclass
class Stats:
    puzzles: int = 0
    puzzles_changed: int = 0
    hints_written: int = 0
    skipped_full: int = 0
    failed: int = 0


def run(args: argparse.Namespace) -> int:
    api_base = os.environ.get("HEYGG_API_BASE_URL", DEFAULT_API_BASE)
    token = os.environ.get("HEYGG_ADMIN_API_TOKEN") or os.environ.get(
        "HEYGG_ADMIN_TOKEN"
    )
    if not token:
        logger.error(
            "No admin token. Set HEYGG_ADMIN_API_TOKEN (use hgg-auth exec prod)."
        )
        return 2

    dictionary: Dictionary | None = None
    try:
        dictionary = Dictionary.load_default()
    except Exception:  # noqa: BLE001 - leak detector works without it
        logger.info("No dictionary loaded; leak detector runs without rule 8.")

    # Collect targets
    targets: list[Target] = []
    for game_key in args.game_key:
        if args.target in ("candidate", "all"):
            targets += load_candidate_targets(game_key, api_base, token)
        if args.target in ("official", "all"):
            targets += load_official_targets(
                game_key,
                api_base,
                token,
                date_from=args.date_from,
                date_to=args.date_to,
                day_from=args.day_from,
                day_to=args.day_to,
            )
    logger.info("Loaded %d target puzzle(s).", len(targets))

    if anthropic is None and not args.dry_run:
        logger.error("anthropic SDK not installed. Run: uv pip install anthropic")
        return 2
    client = anthropic.Anthropic() if anthropic is not None else None

    stats = Stats()
    if args.limit:
        targets = targets[: args.limit]

    for target in targets:
        stats.puzzles += 1
        need = target.entries_needing_hints()
        if not need:
            stats.skipped_full += 1
            continue
        logger.info("%s: %d clue(s) need hints", target.label, len(need))

        if args.dry_run:
            continue
        if args.batch_api:
            # Batch mode is handled in a separate pass below; flagged here only
            # so a mixed invocation is rejected early.
            continue

        # Isolate per-puzzle failures: a bad LLM response or a rejected PATCH
        # for one puzzle must not abort the whole backfill run.
        try:
            hints = generate_hints_sync(target, client, args.model, dictionary)
            applied = target.apply_hints(hints)
            if applied:
                write_target(target, api_base, token)
                stats.puzzles_changed += 1
                stats.hints_written += applied
                logger.info("  wrote %d hint(s)", applied)
        except Exception as exc:  # noqa: BLE001 - log and continue
            stats.failed += 1
            logger.warning("  FAILED %s: %s", target.label, exc)

    if args.batch_api and not args.dry_run:
        _run_batch_mode(targets, client, args.model, dictionary, api_base, token, stats)

    logger.info(
        "Done. puzzles=%d changed=%d hints=%d already-full=%d failed=%d",
        stats.puzzles,
        stats.puzzles_changed,
        stats.hints_written,
        stats.skipped_full,
        stats.failed,
    )
    return 1 if stats.failed else 0


def _run_batch_mode(
    targets: list[Target],
    client: anthropic.Anthropic,
    model: str,
    dictionary: Dictionary | None,
    api_base: str,
    token: str,
    stats: Stats,
) -> None:
    """Submit one Batches request per puzzle, poll, then screen + write."""
    pending = [(t, t.entries_needing_hints()) for t in targets]
    pending = [(t, e) for t, e in pending if e]
    if not pending:
        return

    requests = []
    for idx, (target, entries) in enumerate(pending):
        system_text, user_text = build_hint_generation_messages(
            entries, target.clues_by_key()
        )
        requests.append(
            Request(
                custom_id=f"hint-{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    system=[
                        {
                            "type": "text",
                            "text": system_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    max_tokens=4096,
                    messages=[{"role": "user", "content": user_text}],
                ),
            )
        )

    logger.info("Submitting %d puzzle(s) to the Batches API...", len(requests))
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch ID: %s", batch.id)

    while True:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        counts = status.request_counts
        logger.info(
            "  batch %s: %d succeeded, %d errored, %d processing",
            batch.id,
            counts.succeeded,
            counts.errored,
            counts.processing,
        )
        time.sleep(60)

    for result in client.messages.batches.results(batch.id):
        idx = int(result.custom_id.split("-", 1)[1])
        target, entries = pending[idx]
        if result.result.type != "succeeded":
            logger.warning("  %s: batch request %s", target.label, result.result.type)
            continue
        message = result.result.message
        raw = "".join(
            b.text for b in message.content if getattr(b, "type", "") == "text"
        )
        try:
            hints = parse_hint_response(raw, entries)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s: parse failed: %s", target.label, exc)
            continue
        clean = _screen_hints(hints, entries, dictionary)
        applied = target.apply_hints(clean)
        if applied:
            try:
                write_target(target, api_base, token)
                stats.puzzles_changed += 1
                stats.hints_written += applied
                logger.info("  %s: wrote %d hint(s)", target.label, applied)
            except Exception as exc:  # noqa: BLE001 - log and continue
                stats.failed += 1
                logger.warning("  FAILED %s: %s", target.label, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["candidate", "official", "all"],
        default="all",
        help="Which stores to backfill (default: all).",
    )
    parser.add_argument(
        "--game-key",
        action="append",
        default=[],
        help="Game key(s) to process; repeatable (minicrossword, midicrossword).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--batch-api",
        action="store_true",
        help="Use the Anthropic Message Batches API (cheap, latency-tolerant).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would change; make no LLM or write calls.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N puzzles (0 = all)."
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Only official daily puzzles on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Only official daily puzzles on/before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--day-from",
        type=int,
        default=None,
        help="Only daily puzzles with day_number >= this (the served key).",
    )
    parser.add_argument(
        "--day-to",
        type=int,
        default=None,
        help="Only daily puzzles with day_number <= this (the served key).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.game_key:
        args.game_key = ["minicrossword", "midicrossword"]

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
