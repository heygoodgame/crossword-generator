#!/usr/bin/env python3
"""Classify dictionary words as KEEP or REJECT using an LLM.

Loads words scored 50-60 from the Jeff Chen word list and sends batches
to Ollama for crossword fill quality classification. Results are written
incrementally to kept-{run}.txt and rejected-{run}.txt with checkpoint/resume
support.

Multi-run workflow for reliability (majority vote):
    uv run python scripts/classify_words.py --run 1
    uv run python scripts/classify_words.py --run 2
    uv run python scripts/classify_words.py --run 3
    uv run python scripts/classify_words.py --run 3 --skip-decided
    uv run python scripts/classify_words.py --tally

Other options:
    uv run python scripts/classify_words.py --batch-size 100     # Larger batches
    uv run python scripts/classify_words.py --model llama3.1     # Different model
    uv run python scripts/classify_words.py --workers 4          # Parallel LLM calls
    uv run python scripts/classify_words.py --dry-run
    uv run python scripts/classify_words.py --reset              # Clear all runs
    uv run python scripts/classify_words.py --reset --run 2      # Clear only run 2
    uv run python scripts/classify_words.py --retry-errors
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import ollama

DICT_PATH = Path("dictionaries/XwiJeffChenList.txt")

LAZY_BATCH_THRESHOLD = 0.7  # retry if >70% share the same reason

SYSTEM_PROMPT = """\
You are a crossword puzzle construction expert evaluating word quality for \
crossword grid fill.

IMPORTANT: Crossword entries are written WITHOUT SPACES. You must mentally \
insert spaces to recognize phrases, names, and compound words. For example:
- ABANDONSSHIP → "ABANDONS SHIP" (common phrase — KEEP)
- ABBIEHOFFMAN → "ABBIE HOFFMAN" (famous activist — KEEP)
- ABARRELOFLAUGHS → "A BARREL OF LAUGHS" (common expression — KEEP)
- AARONBURR → "AARON BURR" (US Vice President — KEEP)
Always try to parse an entry as a phrase or name before classifying it.

KEEP — suitable for crossword fill:
- Common everyday English words solvers recognize
- Lively, contemporary vocabulary
- Well-known proper nouns (see proper noun rules below)
- Multi-word phrases people actually say (entries have no spaces)
- Words with interesting letter patterns

REJECT — unsuitable for crossword fill:
- Obscure crosswordese (exists only for convenient letter patterns)
- Abbreviations not used in everyday speech
- Incomplete partial phrases
- Roman numerals used as filler
- Archaic or variant spellings
- Foreign words that aren't common English loanwords
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

Evaluate each word independently. Give a specific reason for each word \
that references the word itself — do not use the same generic reason \
(like "not a word") for multiple entries."""

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


def shuffle_words(
    words: list[tuple[str, int]], run: int
) -> list[tuple[str, int]]:
    """Return a shuffled copy of words, deterministic per run number."""
    shuffled = list(words)
    random.Random(run).shuffle(shuffled)
    return shuffled


def read_output_file(path: Path) -> list[tuple[str, int, str]]:
    """Read a kept/rejected output file. Returns list of (word, score, reason)."""
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


def load_decided_words(output_dir: Path, current_run: int) -> set[str]:
    """Read prior run outputs, return set of decided words."""
    votes: dict[str, list[str]] = defaultdict(list)
    for r in range(1, current_run):
        for word, _, _ in read_output_file(output_dir / f"kept-{r}.txt"):
            votes[word].append("KEEP")
        for word, _, _ in read_output_file(output_dir / f"rejected-{r}.txt"):
            votes[word].append("REJECT")
    return {
        w
        for w, v in votes.items()
        if v.count("KEEP") >= 2 or v.count("REJECT") >= 2
    }


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
    def __init__(self, output_dir: Path, run: int = 1) -> None:
        self.path = output_dir / f"checkpoint-{run}.json"

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
    def __init__(self, output_dir: Path, run: int = 1) -> None:
        self.output_dir = output_dir
        self.kept_path = output_dir / f"kept-{run}.txt"
        self.rejected_path = output_dir / f"rejected-{run}.txt"
        self.errors_path = output_dir / f"errors-{run}.txt"
        self._lock = threading.Lock()

    def ensure_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append_kept(self, entries: list[tuple[str, int, str]]) -> None:
        with self._lock, open(self.kept_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def append_rejected(self, entries: list[tuple[str, int, str]]) -> None:
        with self._lock, open(self.rejected_path, "a") as f:
            for word, score, reason in entries:
                f.write(f"{word};{score};{reason}\n")

    def append_errors(self, entries: list[tuple[str, int, str]]) -> None:
        with self._lock, open(self.errors_path, "a") as f:
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

        # Detect lazy batch: if >70% share the same reason, retry once
        if self._is_lazy_batch(results):
            retry = self._call_llm(word_list)
            if retry and not self._is_lazy_batch(retry):
                results = retry

        kept, rejected, classified = self._process_results(
            results, score_map
        )

        # Find missing words and retry once
        missing = [w for w in word_list if w not in classified]
        if missing:
            retry_results = self._call_llm(missing)
            if retry_results:
                k2, r2, c2 = self._process_results(
                    retry_results, score_map
                )
                kept.extend(k2)
                rejected.extend(r2)
                classified.update(c2)

        # Anything still missing goes to errors
        still_missing = [w for w in word_list if w not in classified]
        errors = [
            (w, score_map[w], "missing_from_response")
            for w in still_missing
        ]

        return kept, rejected, errors

    @staticmethod
    def _is_lazy_batch(results: list[dict]) -> bool:
        """Check if a batch has suspiciously uniform reasons."""
        if len(results) < 5:
            return False
        reasons = [
            r.get("reason", "").lower().strip() for r in results
        ]
        most_common_count = Counter(reasons).most_common(1)[0][1]
        return most_common_count / len(results) > LAZY_BATCH_THRESHOLD

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
    workers: int = 1,
    start_batch: int = 0,
    checkpoint: Checkpoint | None = None,
    run: int = 1,
) -> None:
    batches = make_batches(words, batch_size)
    total_batches = len(batches)

    writer = OutputWriter(output_dir, run=run)
    writer.ensure_dir()
    cp_mgr = CheckpointManager(output_dir, run=run)
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Process in groups of `workers` batches
            for group_start in range(start_batch, total_batches, workers):
                if interrupted:
                    break

                group_end = min(group_start + workers, total_batches)
                futures = {
                    executor.submit(
                        classifier.classify_batch, batches[idx]
                    ): idx
                    for idx in range(group_start, group_end)
                }

                for future in as_completed(futures):
                    if interrupted:
                        break
                    kept, rejected, errors = future.result()

                    if kept:
                        writer.append_kept(kept)
                    if rejected:
                        writer.append_rejected(rejected)
                    if errors:
                        writer.append_errors(errors)

                    progress.update(
                        len(kept), len(rejected), len(errors)
                    )
                    progress.display()

                # Update checkpoint after full group completes
                checkpoint.last_completed_batch = group_end - 1
                group_words = sum(
                    len(batches[i])
                    for i in range(group_start, group_end)
                )
                checkpoint.words_processed += group_words
                checkpoint.words_kept = progress.kept
                checkpoint.words_rejected = progress.rejected
                checkpoint.words_errored = progress.errored
                cp_mgr.save(checkpoint)
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    sys.stderr.write("\n")
    if interrupted:
        print(
            f"\nInterrupted at batch "
            f"{checkpoint.last_completed_batch + 1}"
            f"/{total_batches}. Resume by re-running the script."
        )
    else:
        print(f"\nDone. KEEP: {checkpoint.words_kept} | "
              f"REJECT: {checkpoint.words_rejected} | "
              f"ERR: {checkpoint.words_errored}")


def run_tally(output_dir: Path) -> None:
    """Read all per-run output files, tally votes, write final results."""
    votes: dict[str, dict[str, int]] = defaultdict(lambda: {"keep": 0, "reject": 0})
    scores: dict[str, int] = {}
    runs_found: set[int] = set()

    # Find all numbered kept/rejected files (exclude *-final.txt)
    for path in sorted(output_dir.glob("kept-*.txt")):
        if path.name == "kept-final.txt":
            continue
        run_str = path.stem.split("-", 1)[1]
        try:
            run_num = int(run_str)
        except ValueError:
            continue
        runs_found.add(run_num)
        for word, score, _ in read_output_file(path):
            votes[word]["keep"] += 1
            scores[word] = score

    for path in sorted(output_dir.glob("rejected-*.txt")):
        if path.name == "rejected-final.txt":
            continue
        run_str = path.stem.split("-", 1)[1]
        try:
            run_num = int(run_str)
        except ValueError:
            continue
        runs_found.add(run_num)
        for word, score, _ in read_output_file(path):
            votes[word]["reject"] += 1
            scores[word] = score

    if not votes:
        print("No run output files found.")
        return

    kept_final = output_dir / "kept-final.txt"
    rejected_final = output_dir / "rejected-final.txt"

    kept_count = 0
    rejected_count = 0

    with open(kept_final, "w") as kf, open(rejected_final, "w") as rf:
        for word in sorted(votes):
            v = votes[word]
            keep_n = v["keep"]
            reject_n = v["reject"]
            score = scores[word]
            line = f"{word};{score};{keep_n};{reject_n}\n"
            if keep_n >= reject_n:
                kf.write(line)
                kept_count += 1
            else:
                rf.write(line)
                rejected_count += 1

    total = len(votes)
    print(f"Tally across {len(runs_found)} runs ({sorted(runs_found)})")
    print(f"Total words: {total:,}")
    print(f"KEEP (majority): {kept_count:,}")
    print(f"REJECT (majority): {rejected_count:,}")
    print(f"Written to {kept_final} and {rejected_final}")


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent LLM calls (default: 1)",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run number for multi-run majority vote (default: 1)",
    )
    parser.add_argument(
        "--skip-decided",
        action="store_true",
        help="Skip words already decided by prior runs (2+ votes same way)",
    )
    parser.add_argument(
        "--tally",
        action="store_true",
        help="Combine all run results with majority vote, write final files",
    )
    args = parser.parse_args()

    # Tally mode: no classification, just combine results
    if args.tally:
        run_tally(args.output_dir)
        return

    run_num = args.run if args.run is not None else 1

    # Load words
    words = load_target_words(DICT_PATH, args.min_score, args.max_score)

    if args.dry_run:
        # Apply skip-decided filtering for accurate count
        skipped = 0
        if args.skip_decided:
            decided = load_decided_words(args.output_dir, run_num)
            original_count = len(words)
            words = [(w, s) for w, s in words if w not in decided]
            skipped = original_count - len(words)

        shuffled = shuffle_words(words, run_num)
        batches = make_batches(shuffled, args.batch_size)

        print(f"Dictionary: {DICT_PATH}")
        print(f"Score range: {args.min_score}-{args.max_score}")
        print(f"Run: {run_num}")
        print(f"Words found: {len(words):,}")
        if skipped:
            print(f"Skipped (already decided): {skipped:,}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total batches: {len(batches):,}")
        print(f"Model: {args.model}")
        print(f"Output dir: {args.output_dir}")

        cp_mgr = CheckpointManager(args.output_dir, run=run_num)
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
        if args.run is not None:
            # Clear only the specified run
            cp_mgr = CheckpointManager(args.output_dir, run=run_num)
            writer = OutputWriter(args.output_dir, run=run_num)
            cp_mgr.reset()
            writer.clear_all()
            print(f"Run {run_num} checkpoint and output files cleared.")
        else:
            # Clear all runs
            output_dir = args.output_dir
            cleared = 0
            for pattern in ("kept-*.txt", "rejected-*.txt", "errors-*.txt",
                            "checkpoint-*.json"):
                for path in output_dir.glob(pattern):
                    path.unlink(missing_ok=True)
                    cleared += 1
            print(f"All run files cleared ({cleared} files removed).")
        return

    if args.retry_errors:
        writer = OutputWriter(args.output_dir, run=run_num)
        error_words = load_error_words(writer.errors_path)
        if not error_words:
            print(f"No errors to retry for run {run_num}.")
            return
        print(f"Retrying {len(error_words)} errored words (run {run_num})...")
        # Clear errors file before retry
        writer.errors_path.unlink(missing_ok=True)
        run_classification(
            error_words,
            args.batch_size,
            args.model,
            args.base_url,
            args.output_dir,
            args.max_retries,
            workers=args.workers,
            run=run_num,
        )
        return

    # Apply skip-decided filtering
    if args.skip_decided:
        decided = load_decided_words(args.output_dir, run_num)
        original_count = len(words)
        words = [(w, s) for w, s in words if w not in decided]
        skipped = original_count - len(words)
        print(f"Skipping {skipped:,} already-decided words, "
              f"classifying {len(words):,}")

    # Shuffle words deterministically per run
    words = shuffle_words(words, run_num)
    batches = make_batches(words, args.batch_size)

    # Check for existing checkpoint
    cp_mgr = CheckpointManager(args.output_dir, run=run_num)
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
            print(f"Run {run_num}: all batches already completed. "
                  f"Use --reset --run {run_num} to start over.")
            return
        print(
            f"Run {run_num}: resuming from batch {start_batch}/{len(batches)} "
            f"({cp.words_processed:,} words processed)"
        )
    else:
        print(f"Run {run_num}: starting classification of {len(words):,} words "
              f"in {len(batches):,} batches")
        cp = None

    run_classification(
        words,
        args.batch_size,
        args.model,
        args.base_url,
        args.output_dir,
        args.max_retries,
        workers=args.workers,
        start_batch=start_batch,
        checkpoint=cp,
        run=run_num,
    )


if __name__ == "__main__":
    main()
