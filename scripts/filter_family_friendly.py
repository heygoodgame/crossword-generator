#!/usr/bin/env python3
"""Filter dictionary words as SAFE or FLAGGED for family-friendliness.

Reads all words from the XWI Jeff Chen word list and classifies them
using Claude Haiku via the Anthropic Batches API. Words that contain
profanity, slurs, vulgar slang, sexual innuendo, or other content
unsuitable for all ages are flagged.

Usage:
    uv run python scripts/filter_family_friendly.py --run 1
    uv run python scripts/filter_family_friendly.py --limit 100 --run 1
    uv run python scripts/filter_family_friendly.py --dry-run
    uv run python scripts/filter_family_friendly.py --write-dictionaries
    uv run python scripts/filter_family_friendly.py --reset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
except ImportError:
    anthropic = None

DICT_PATH = Path("dictionaries/XwiJeffChenList.txt")
OUTPUT_DIR = Path("output/family_friendly_filter")

SAFE_DICT_PATH = Path("dictionaries/XwiJeffChenList-FamilyFriendly.txt")
FLAGGED_DICT_PATH = Path("dictionaries/XwiJeffChenList-NotFamilyFriendly.txt")

# Claude pricing per million tokens
CLAUDE_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """\
You are a content filter evaluating whether crossword puzzle entries are \
family-friendly — suitable for solvers of all ages including children.

IMPORTANT: Crossword entries are written WITHOUT SPACES. You must mentally \
insert spaces to recognize phrases and compound words. For example:
- ASSEMBLE → "ASSEMBLE" (safe — not profane despite containing "ass")
- COCKATOO → "COCKATOO" (safe — it's a bird)
- SHUTTLECOCK → "SHUTTLECOCK" (safe — it's a badminton term)
- HOTDOG → "HOT DOG" (safe — it's food)
- THERAPIST → "THERAPIST" (safe — it's a profession)
Always parse the FULL word/phrase before judging.

SAFE — appropriate for all ages:
- Standard English words and phrases
- Words that merely contain letter sequences resembling profanity but are \
themselves innocent (ASSEMBLE, COCKATOO, CLASSIC, THERAPIST, SHITAKE)
- Medical/anatomical terms used in everyday language (BREAST, BLADDER)
- Mild colloquialisms (BUTT, CRAP, DARN, HECK, SUCK)

FLAGGED — not appropriate for all ages:
- Profanity and vulgar expletives (the "big" swear words)
- Vulgar slang for body parts or sexual acts
- Sexual innuendo when the primary meaning is sexual
- Racial, ethnic, or gender-based slurs
- Derogatory terms targeting any group
- Crude scatological humor beyond mild terms
- Drug slang when the primary meaning is illicit drug use

When in doubt, flag it. It is better to over-flag than to let inappropriate \
content through to a family audience.

Evaluate each word independently."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["SAFE", "FLAGGED"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["word", "verdict", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Word loading
# ---------------------------------------------------------------------------


def load_all_words(dict_path: Path) -> list[tuple[str, int]]:
    """Read all words from dictionary, return sorted (word, score) list."""
    words: list[tuple[str, int]] = []
    with open(dict_path) as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            word, score_str = line.rsplit(";", 1)
            try:
                score = int(score_str)
            except ValueError:
                continue
            words.append((word.lower(), score))
    words.sort(key=lambda ws: ws[0])
    return words


def read_output_file(path: Path) -> list[tuple[str, int, str]]:
    """Read a safe/flagged output file. Returns list of (word, score, reason)."""
    if not path.exists():
        return []
    entries: list[tuple[str, int, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            parts = line.split(";", 2)
            if len(parts) >= 2:
                try:
                    reason = parts[2] if len(parts) > 2 else ""
                    entries.append((parts[0], int(parts[1]), reason))
                except ValueError:
                    continue
    return entries


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def make_batches(
    words: list[tuple[str, int]], batch_size: int
) -> list[list[tuple[str, int]]]:
    return [words[i : i + batch_size] for i in range(0, len(words), batch_size)]


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


class OutputWriter:
    def __init__(self, output_dir: Path, run: int = 1) -> None:
        self.output_dir = output_dir
        self.safe_path = output_dir / f"safe-{run}.txt"
        self.flagged_path = output_dir / f"flagged-{run}.txt"
        self.errors_path = output_dir / f"errors-{run}.txt"

    def ensure_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append_safe(self, entries: list[tuple[str, int, str]]) -> None:
        with open(self.safe_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def append_flagged(self, entries: list[tuple[str, int, str]]) -> None:
        with open(self.flagged_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def append_errors(self, entries: list[tuple[str, int, str]]) -> None:
        with open(self.errors_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def clear_all(self) -> None:
        for p in (self.safe_path, self.flagged_path, self.errors_path):
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def _load_batch_checkpoint(output_dir: Path, run: int) -> dict | None:
    path = output_dir / f"batch-checkpoint-{run}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_batch_checkpoint(output_dir: Path, run: int, data: dict) -> None:
    path = output_dir / f"batch-checkpoint-{run}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Batch API classification
# ---------------------------------------------------------------------------


def _build_batch_request(
    batch_words: list[tuple[str, int]], batch_idx: int, model: str
) -> Request:
    """Build a single Anthropic Batch API request for a word batch."""
    word_list = [w for w, _ in batch_words]
    user_msg = "Classify each word as SAFE or FLAGGED.\n\nWords:\n"
    user_msg += "\n".join(word_list)

    return Request(
        custom_id=f"batch-{batch_idx}",
        params=MessageCreateParamsNonStreaming(
            model=model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=4096,
            temperature=0.1,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": RESPONSE_SCHEMA,
                }
            },
        ),
    )


def run_batch_api_classification(
    words: list[tuple[str, int]],
    batch_size: int,
    model: str,
    output_dir: Path,
    run: int = 1,
) -> None:
    """Submit all word batches to the Anthropic Batch API, poll, and process results."""
    if anthropic is None:
        print("Error: anthropic package not installed. Run: uv pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    writer = OutputWriter(output_dir, run=run)
    writer.ensure_dir()

    batches = make_batches(words, batch_size)
    score_map = {w: s for w, s in words}

    max_retries = 5

    # Check for existing batch checkpoint
    cp = _load_batch_checkpoint(output_dir, run)
    if cp and cp.get("status") in ("in_progress", "finalizing"):
        batch_id = cp["batch_id"]
        print(f"Run {run}: resuming poll for batch {batch_id}")
    else:
        # Build and submit requests
        requests = [
            _build_batch_request(batch_words, idx, model)
            for idx, batch_words in enumerate(batches)
        ]

        print(
            f"Run {run}: submitting {len(requests)} "
            f"batches to Batch API..."
        )
        for attempt in range(max_retries):
            try:
                result = client.messages.batches.create(requests=requests)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"Submit failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Submit failed after {max_retries} attempts: {e}")
                    sys.exit(1)
        batch_id = result.id
        print(f"Batch ID: {batch_id}")

        _save_batch_checkpoint(output_dir, run, {
            "batch_id": batch_id,
            "run": run,
            "status": "in_progress",
            "total_batches": len(batches),
            "submitted_at": datetime.now(UTC).isoformat(),
        })

    # Poll until complete
    poll_interval = 60
    poll_errors = 0
    while True:
        try:
            status = client.messages.batches.retrieve(batch_id)
        except Exception as e:
            poll_errors += 1
            if poll_errors >= max_retries:
                print(f"\nPolling failed after {max_retries} "
                      f"consecutive errors: {e}")
                print(f"Re-run to resume polling batch {batch_id}")
                sys.exit(1)
            wait = 2 ** poll_errors
            sys.stderr.write(f"\nPoll error ({e}), retrying in {wait}s...\n")
            time.sleep(wait)
            continue
        poll_errors = 0
        counts = status.request_counts
        total = (counts.processing + counts.succeeded + counts.errored
                 + counts.canceled + counts.expired)
        done = (counts.succeeded + counts.errored
                + counts.canceled + counts.expired)

        sys.stderr.write(
            f"\r[Batch {batch_id}] {done}/{total} requests done "
            f"(succeeded: {counts.succeeded}, errored: {counts.errored}, "
            f"processing: {counts.processing})    "
        )
        sys.stderr.flush()

        _save_batch_checkpoint(output_dir, run, {
            "batch_id": batch_id,
            "run": run,
            "status": status.processing_status,
            "total_batches": len(batches),
        })

        if status.processing_status == "ended":
            break

        time.sleep(poll_interval)

    sys.stderr.write("\n")
    print("Batch complete. Processing results...")

    # Stream results
    safe_count = 0
    flagged_count = 0
    errored_count = 0
    results_iter = None
    for attempt in range(max_retries):
        try:
            results_iter = client.messages.batches.results(batch_id)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"Results fetch failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed to fetch results after "
                      f"{max_retries} attempts: {e}")
                print(f"Re-run to retry (batch {batch_id} is still available)")
                sys.exit(1)

    for api_result in results_iter:
        if api_result.result.type != "succeeded":
            batch_idx = int(api_result.custom_id.split("-", 1)[1])
            if batch_idx < len(batches):
                errors = [
                    (w, s, f"batch_api_{api_result.result.type}")
                    for w, s in batches[batch_idx]
                ]
                writer.append_errors(errors)
                errored_count += len(errors)
            continue

        message = api_result.result.message
        try:
            content = message.content[0].text
            parsed = json.loads(content)
            classifications = parsed.get("classifications", [])
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            batch_idx = int(api_result.custom_id.split("-", 1)[1])
            if batch_idx < len(batches):
                errors = [
                    (w, s, "parse_error") for w, s in batches[batch_idx]
                ]
                writer.append_errors(errors)
                errored_count += len(errors)
            continue

        safe: list[tuple[str, int, str]] = []
        flagged: list[tuple[str, int, str]] = []

        for item in classifications:
            word = item.get("word", "").lower().strip()
            verdict = item.get("verdict", "").upper().strip()
            reason = item.get("reason", "").strip()

            if word not in score_map:
                continue

            if verdict == "SAFE":
                safe.append((word, score_map[word], reason))
            elif verdict == "FLAGGED":
                flagged.append((word, score_map[word], reason))

        if safe:
            writer.append_safe(safe)
            safe_count += len(safe)
        if flagged:
            writer.append_flagged(flagged)
            flagged_count += len(flagged)

    _save_batch_checkpoint(output_dir, run, {
        "batch_id": batch_id,
        "run": run,
        "status": "completed",
        "safe": safe_count,
        "flagged": flagged_count,
        "errored": errored_count,
    })

    print(f"Done. SAFE: {safe_count} | FLAGGED: {flagged_count} | "
          f"ERR: {errored_count}")


# ---------------------------------------------------------------------------
# Write final dictionary files
# ---------------------------------------------------------------------------


def write_dictionaries(output_dir: Path) -> None:
    """Read run results and write family-friendly / not-family-friendly dictionaries."""
    # Collect verdicts from all runs
    verdicts: dict[str, str] = {}

    for path in sorted(output_dir.glob("safe-*.txt")):
        for word, _score, _reason in read_output_file(path):
            if word not in verdicts:
                verdicts[word] = "SAFE"

    for path in sorted(output_dir.glob("flagged-*.txt")):
        for word, _score, _reason in read_output_file(path):
            if word not in verdicts:
                verdicts[word] = "FLAGGED"

    if not verdicts:
        print("No run output files found in", output_dir)
        return

    # Read original file to preserve exact lines (case, formatting)
    safe_lines: list[str] = []
    flagged_lines: list[str] = []
    unclassified_lines: list[str] = []

    with open(DICT_PATH) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or ";" not in stripped:
                continue
            word, _ = stripped.rsplit(";", 1)
            word_lower = word.lower()
            verdict = verdicts.get(word_lower)
            if verdict == "SAFE":
                safe_lines.append(stripped)
            elif verdict == "FLAGGED":
                flagged_lines.append(stripped)
            else:
                # Unclassified — include in safe by default
                unclassified_lines.append(stripped)

    # Write safe dictionary (includes unclassified words)
    with open(SAFE_DICT_PATH, "w") as f:
        for line in safe_lines:
            f.write(line + "\n")
        for line in unclassified_lines:
            f.write(line + "\n")

    # Write flagged dictionary
    with open(FLAGGED_DICT_PATH, "w") as f:
        for line in flagged_lines:
            f.write(line + "\n")

    total_original = len(safe_lines) + len(flagged_lines) + len(unclassified_lines)
    total_output = (len(safe_lines) + len(unclassified_lines)) + len(flagged_lines)

    print(f"Original words: {total_original:,}")
    print(f"SAFE: {len(safe_lines):,}")
    print(f"FLAGGED: {len(flagged_lines):,}")
    if unclassified_lines:
        print(f"Unclassified (added to safe): {len(unclassified_lines):,}")
    print(f"Family-friendly dict: {SAFE_DICT_PATH} "
          f"({len(safe_lines) + len(unclassified_lines):,} words)")
    print(f"Not-family-friendly dict: {FLAGGED_DICT_PATH} "
          f"({len(flagged_lines):,} words)")
    print(f"Total output: {total_output:,} "
          f"(should equal original: {total_original:,})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter dictionary words for family-friendliness "
                    "using Claude Haiku via the Batch API."
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run number (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Words per API request (default: 200)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N words. For testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without running classification",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear checkpoint and output files, start over",
    )
    parser.add_argument(
        "--write-dictionaries",
        action="store_true",
        help="Read results and write final family-friendly dictionary files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Write dictionaries mode
    if args.write_dictionaries:
        write_dictionaries(args.output_dir)
        return

    run_num = args.run if args.run is not None else 1

    # Load words
    words = load_all_words(DICT_PATH)

    if args.dry_run:
        if args.limit:
            words = words[: args.limit]
        batches = make_batches(words, args.batch_size)

        print(f"Dictionary: {DICT_PATH}")
        print(f"Words found: {len(words):,}")
        if args.limit:
            print(f"Limit: {args.limit}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total batches: {len(batches):,}")
        print(f"Model: {args.model}")
        print(f"Output dir: {args.output_dir}")
        print("Mode: Batch API (async, 50% cheaper)")

        # Cost estimate
        pricing = CLAUDE_PRICING.get(args.model)
        if pricing:
            sys_prompt_tokens = 500
            avg_word_tokens = 3
            words_per_batch = args.batch_size
            input_per_batch = (sys_prompt_tokens
                              + words_per_batch * avg_word_tokens + 20)
            output_per_batch = min(words_per_batch * 25, 4096)

            total_input = input_per_batch * len(batches)
            total_output = output_per_batch * len(batches)

            input_cost = total_input / 1_000_000 * pricing["input"]
            output_cost = total_output / 1_000_000 * pricing["output"]
            total_cost = input_cost + output_cost

            print(f"\nEstimated cost ({args.model}):")
            print(f"  Input:  ~{total_input / 1_000_000:.2f}M tokens "
                  f"× ${pricing['input']:.2f}/MTok = ${input_cost:.2f}")
            print(f"  Output: ~{total_output / 1_000_000:.2f}M tokens "
                  f"× ${pricing['output']:.2f}/MTok = ${output_cost:.2f}")
            print(f"  Total:  ~${total_cost:.2f}")
            print(f"  With Batch API (50% off): ~${total_cost / 2:.2f}")

        cp = _load_batch_checkpoint(args.output_dir, run_num)
        if cp:
            print(f"\nCheckpoint found: batch {cp.get('batch_id', '?')}, "
                  f"status: {cp.get('status', '?')}")
        return

    if args.reset:
        output_dir = args.output_dir
        cleared = 0
        if args.run is not None:
            # Clear only the specified run
            for pattern in (f"safe-{run_num}.txt", f"flagged-{run_num}.txt",
                            f"errors-{run_num}.txt",
                            f"batch-checkpoint-{run_num}.json"):
                path = output_dir / pattern
                if path.exists():
                    path.unlink()
                    cleared += 1
            print(f"Run {run_num} files cleared ({cleared} files removed).")
        else:
            for pattern in ("safe-*.txt", "flagged-*.txt", "errors-*.txt",
                            "batch-checkpoint-*.json"):
                for path in output_dir.glob(pattern):
                    path.unlink(missing_ok=True)
                    cleared += 1
            print(f"All run files cleared ({cleared} files removed).")
        return

    # Apply limit
    if args.limit:
        words = words[: args.limit]

    run_batch_api_classification(
        words,
        args.batch_size,
        args.model,
        args.output_dir,
        run=run_num,
    )


if __name__ == "__main__":
    main()
