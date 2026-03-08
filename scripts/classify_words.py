#!/usr/bin/env python3
"""Classify dictionary words as KEEP or REJECT using an LLM.

Loads words scored 50-60 from the Jeff Chen word list and sends batches
to Ollama for crossword fill quality classification. Results are written
incrementally to kept.txt and rejected.txt with checkpoint/resume support.

Usage:
    uv run python scripts/classify_words.py                    # Start fresh or resume
    uv run python scripts/classify_words.py --batch-size 100   # Larger batches
    uv run python scripts/classify_words.py --model llama3.1   # Different model
    uv run python scripts/classify_words.py --dry-run
    uv run python scripts/classify_words.py --reset
    uv run python scripts/classify_words.py --retry-errors
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import ollama

DICT_PATH = Path("dictionaries/XwiJeffChenList.txt")

SYSTEM_PROMPT = """\
You are a crossword puzzle construction expert evaluating word quality for \
crossword grid fill.

KEEP — suitable for crossword fill:
- Common everyday English words solvers recognize
- Lively, contemporary vocabulary
- Well-known proper nouns (see proper noun rules below)
- Multi-word phrases people actually say (entries may have no spaces)
- Words with interesting letter patterns

REJECT — unsuitable for crossword fill:
- Obscure crosswordese (exists only for convenient letter patterns)
- Abbreviations not used in everyday speech
- Incomplete partial phrases
- Roman numerals used as filler
- Archaic or variant spellings
- Foreign words that aren't common English loanwords
- Three-letter glue/junk words
- Contrived or dictionary-only entries
- Standalone prefixes or suffixes
- Obscure proper nouns (see proper noun rules below)

PROPER NOUN RULES:
Test: Would most American adults recognize this name without context?
If you need a qualifier to explain who they are, REJECT.

Categories that are usually KEEP:
- US presidents, world leaders, major historical figures
- World capitals, US states, major world cities
- Oscar/Grammy-level actors, musicians, directors
- Major sports teams, Hall of Fame athletes
- Global brands, Fortune 500 companies

Categories that are usually REJECT:
- Single-season reality TV contestants
- Minor biblical or mythological figures
- Local or regional politicians
- Niche athletes outside mainstream sports
- One-hit wonders remembered only by crossword constructors

Cross-generational check: Would both a 25-year-old and a 65-year-old \
recognize it? If only one generation knows it, lean REJECT.

Examples:
- ELVIS → KEEP (universally recognized musician)
- OPRAH → KEEP (universally recognized TV host)
- AALIYAH → KEEP (widely known singer)
- TESLA → KEEP (major brand and historical figure)
- OTERI → REJECT (needs qualifier: "SNL cast member")
- ESAI → REJECT (crosswordese actor name)
- NENE → REJECT (reality TV, not broadly known)
- ELIHU → REJECT (obscure biblical figure)

For each word, provide a brief reason (under 10 words)."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["KEEP", "REJECT"]},
                    "reason": {"type": "string"},
                },
                "required": ["word", "verdict", "reason"],
            },
        },
    },
    "required": ["classifications"],
}


# ---------------------------------------------------------------------------
# Word loading
# ---------------------------------------------------------------------------


def load_target_words(
    dict_path: Path, min_score: int, max_score: int
) -> list[tuple[str, int]]:
    """Read dictionary, filter to score range, return sorted (word, score) list."""
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
            if min_score <= score <= max_score:
                words.append((word.lower(), score))
    words.sort(key=lambda ws: ws[0])
    return words


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    last_completed_batch: int = -1
    total_batches: int = 0
    batch_size: int = 50
    model: str = "llama3"
    started_at: str = ""
    words_processed: int = 0
    words_kept: int = 0
    words_rejected: int = 0
    words_errored: int = 0


class CheckpointManager:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "checkpoint.json"

    def load(self) -> Checkpoint | None:
        if not self.path.exists():
            return None
        with open(self.path) as f:
            data = json.load(f)
        return Checkpoint(**data)

    def save(self, cp: Checkpoint) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(vars(cp), f, indent=2)
        os.replace(tmp, self.path)

    def reset(self) -> None:
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


class OutputWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.kept_path = output_dir / "kept.txt"
        self.rejected_path = output_dir / "rejected.txt"
        self.errors_path = output_dir / "errors.txt"

    def ensure_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append_kept(self, entries: list[tuple[str, int, str]]) -> None:
        with open(self.kept_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def append_rejected(self, entries: list[tuple[str, int, str]]) -> None:
        with open(self.rejected_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def append_errors(self, entries: list[tuple[str, int, str]]) -> None:
        with open(self.errors_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def clear_all(self) -> None:
        for p in (self.kept_path, self.rejected_path, self.errors_path):
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


@dataclass
class ProgressTracker:
    total_batches: int
    total_words: int
    batch_size: int
    kept: int = 0
    rejected: int = 0
    errored: int = 0
    batches_done: int = 0
    start_time: float = field(default_factory=time.monotonic)

    def update(self, n_kept: int, n_rejected: int, n_errored: int) -> None:
        self.batches_done += 1
        self.kept += n_kept
        self.rejected += n_rejected
        self.errored += n_errored

    def display(self) -> None:
        processed = self.kept + self.rejected + self.errored
        pct = processed / self.total_words * 100 if self.total_words else 0
        elapsed = time.monotonic() - self.start_time
        if self.batches_done > 0:
            secs_per_batch = elapsed / self.batches_done
            remaining = (self.total_batches - self.batches_done) * secs_per_batch
            eta_h = remaining / 3600
            eta_str = f"{eta_h:.1f}h"
        else:
            eta_str = "??h"
        line = (
            f"\r[{self.batches_done}/{self.total_batches}] "
            f"{processed}/{self.total_words} ({pct:.1f}%) | "
            f"KEEP: {self.kept} REJECT: {self.rejected} ERR: {self.errored} | "
            f"ETA: {eta_str}"
        )
        sys.stderr.write(line)
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Word classifier
# ---------------------------------------------------------------------------


class WordClassifier:
    def __init__(self, model: str, base_url: str, max_retries: int) -> None:
        self.model = model
        self.client = ollama.Client(host=base_url)
        self.max_retries = max_retries

    def classify_batch(
        self, words_with_scores: list[tuple[str, int]]
    ) -> tuple[
        list[tuple[str, int, str]],  # kept
        list[tuple[str, int, str]],  # rejected
        list[tuple[str, int, str]],  # errors
    ]:
        """Classify a batch of words. Returns (kept, rejected, errors)."""
        score_map = {w: s for w, s in words_with_scores}
        word_list = [w for w, _ in words_with_scores]

        results = self._call_llm(word_list)
        if results is None:
            return (
                [],
                [],
                [(w, s, "llm_call_failed") for w, s in words_with_scores],
            )

        kept, rejected, classified = self._process_results(results, score_map)

        # Find missing words and retry once
        missing = [w for w in word_list if w not in classified]
        if missing:
            retry_results = self._call_llm(missing)
            if retry_results:
                k2, r2, c2 = self._process_results(retry_results, score_map)
                kept.extend(k2)
                rejected.extend(r2)
                classified.update(c2)

        # Anything still missing goes to errors
        still_missing = [w for w in word_list if w not in classified]
        errors = [(w, score_map[w], "missing_from_response") for w in still_missing]

        return kept, rejected, errors

    def _call_llm(self, words: list[str]) -> list[dict] | None:
        """Call Ollama and return parsed classifications list, or None on failure."""
        user_msg = "Classify each word as KEEP or REJECT.\n\nWords:\n"
        user_msg += "\n".join(words)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    format=RESPONSE_SCHEMA,
                    options={"temperature": 0.1},
                    keep_alive="24h",
                )
                content = response["message"]["content"]
                parsed = json.loads(content)
                return parsed.get("classifications", [])
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                continue
            except Exception:
                if attempt < self.max_retries - 1:
                    backoff = 2 ** (2 * attempt + 1)  # 2s, 8s, 32s
                    time.sleep(backoff)
                continue
        return None

    def _process_results(
        self, results: list[dict], score_map: dict[str, int]
    ) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]], set[str]]:
        """Parse LLM results into kept/rejected lists."""
        kept: list[tuple[str, int, str]] = []
        rejected: list[tuple[str, int, str]] = []
        classified: set[str] = set()

        for item in results:
            word = item.get("word", "").lower().strip()
            verdict = item.get("verdict", "").upper().strip()
            reason = item.get("reason", "").strip()

            # Skip hallucinated words
            if word not in score_map:
                continue

            if verdict == "KEEP":
                kept.append((word, score_map[word], reason))
                classified.add(word)
            elif verdict == "REJECT":
                rejected.append((word, score_map[word], reason))
                classified.add(word)
            # Invalid verdict — skip, will be caught as missing

        return kept, rejected, classified


# ---------------------------------------------------------------------------
# Error retry
# ---------------------------------------------------------------------------


def load_error_words(errors_path: Path) -> list[tuple[str, int]]:
    """Load words from errors.txt for retry."""
    if not errors_path.exists():
        return []
    words: list[tuple[str, int]] = []
    with open(errors_path) as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            parts = line.split(";", 2)
            if len(parts) >= 2:
                try:
                    words.append((parts[0], int(parts[1])))
                except ValueError:
                    continue
    return words


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_batches(
    words: list[tuple[str, int]], batch_size: int
) -> list[list[tuple[str, int]]]:
    return [words[i : i + batch_size] for i in range(0, len(words), batch_size)]


def run_classification(
    words: list[tuple[str, int]],
    batch_size: int,
    model: str,
    base_url: str,
    output_dir: Path,
    max_retries: int,
    start_batch: int = 0,
    checkpoint: Checkpoint | None = None,
) -> None:
    batches = make_batches(words, batch_size)
    total_batches = len(batches)

    writer = OutputWriter(output_dir)
    writer.ensure_dir()
    cp_mgr = CheckpointManager(output_dir)
    classifier = WordClassifier(model, base_url, max_retries)

    if checkpoint is None:
        checkpoint = Checkpoint(
            total_batches=total_batches,
            batch_size=batch_size,
            model=model,
            started_at=datetime.now(UTC).isoformat(),
        )

    progress = ProgressTracker(
        total_batches=total_batches,
        total_words=len(words),
        batch_size=batch_size,
        kept=checkpoint.words_kept,
        rejected=checkpoint.words_rejected,
        errored=checkpoint.words_errored,
        batches_done=start_batch,
    )

    # Handle graceful shutdown
    interrupted = False

    def handle_interrupt(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    prev_handler = signal.signal(signal.SIGINT, handle_interrupt)

    try:
        for batch_idx in range(start_batch, total_batches):
            if interrupted:
                break

            batch = batches[batch_idx]
            kept, rejected, errors = classifier.classify_batch(batch)

            if kept:
                writer.append_kept(kept)
            if rejected:
                writer.append_rejected(rejected)
            if errors:
                writer.append_errors(errors)

            checkpoint.last_completed_batch = batch_idx
            checkpoint.words_processed += len(batch)
            checkpoint.words_kept += len(kept)
            checkpoint.words_rejected += len(rejected)
            checkpoint.words_errored += len(errors)
            cp_mgr.save(checkpoint)

            progress.update(len(kept), len(rejected), len(errors))
            progress.display()
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    sys.stderr.write("\n")
    if interrupted:
        print(
            f"\nInterrupted at batch {checkpoint.last_completed_batch + 1}"
            f"/{total_batches}. Resume by re-running the script."
        )
    else:
        print(f"\nDone. KEEP: {checkpoint.words_kept} | "
              f"REJECT: {checkpoint.words_rejected} | "
              f"ERR: {checkpoint.words_errored}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify dictionary words as KEEP or REJECT using an LLM."
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Words per LLM call (default: 50)"
    )
    parser.add_argument(
        "--model", default="llama3", help="Ollama model name (default: llama3)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/word_classification"),
        help="Output directory (default: output/word_classification)",
    )
    parser.add_argument(
        "--min-score", type=int, default=50, help="Minimum word score (default: 50)"
    )
    parser.add_argument(
        "--max-score", type=int, default=60, help="Maximum word score (default: 60)"
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
        "--retry-errors",
        action="store_true",
        help="Re-process words from errors.txt",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per LLM call (default: 3)",
    )
    args = parser.parse_args()

    # Load words
    words = load_target_words(DICT_PATH, args.min_score, args.max_score)
    batches = make_batches(words, args.batch_size)

    if args.dry_run:
        print(f"Dictionary: {DICT_PATH}")
        print(f"Score range: {args.min_score}-{args.max_score}")
        print(f"Words found: {len(words):,}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total batches: {len(batches):,}")
        print(f"Model: {args.model}")
        print(f"Output dir: {args.output_dir}")

        cp_mgr = CheckpointManager(args.output_dir)
        cp = cp_mgr.load()
        if cp:
            remaining = len(batches) - (cp.last_completed_batch + 1)
            print(f"\nCheckpoint found: {cp.last_completed_batch + 1}"
                  f"/{cp.total_batches} batches done")
            print(f"  KEEP: {cp.words_kept} | REJECT: {cp.words_rejected}"
                  f" | ERR: {cp.words_errored}")
            print(f"  Remaining batches: {remaining}")
        return

    if args.reset:
        cp_mgr = CheckpointManager(args.output_dir)
        writer = OutputWriter(args.output_dir)
        cp_mgr.reset()
        writer.clear_all()
        print("Checkpoint and output files cleared.")
        return

    if args.retry_errors:
        writer = OutputWriter(args.output_dir)
        error_words = load_error_words(writer.errors_path)
        if not error_words:
            print("No errors to retry.")
            return
        print(f"Retrying {len(error_words)} errored words...")
        # Clear errors file before retry
        writer.errors_path.unlink(missing_ok=True)
        run_classification(
            error_words,
            args.batch_size,
            args.model,
            args.base_url,
            args.output_dir,
            args.max_retries,
        )
        return

    # Check for existing checkpoint
    cp_mgr = CheckpointManager(args.output_dir)
    cp = cp_mgr.load()
    start_batch = 0

    if cp is not None:
        # Validate checkpoint compatibility
        if cp.batch_size != args.batch_size or cp.model != args.model:
            print(
                f"Warning: checkpoint was created with batch_size={cp.batch_size}, "
                f"model={cp.model}."
            )
            print(
                f"Current args: batch_size={args.batch_size}, model={args.model}."
            )
            print("Use --reset to start over, or match the checkpoint settings.")
            sys.exit(1)

        start_batch = cp.last_completed_batch + 1
        if start_batch >= len(batches):
            print("All batches already completed. Use --reset to start over.")
            return
        print(
            f"Resuming from batch {start_batch}/{len(batches)} "
            f"({cp.words_processed:,} words processed)"
        )
    else:
        print(f"Starting classification of {len(words):,} words in "
              f"{len(batches):,} batches")
        cp = None

    run_classification(
        words,
        args.batch_size,
        args.model,
        args.base_url,
        args.output_dir,
        args.max_retries,
        start_batch=start_batch,
        checkpoint=cp,
    )


if __name__ == "__main__":
    main()
