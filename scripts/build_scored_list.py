#!/usr/bin/env python3
"""Build HggScoredCrosswordList.txt from kept/rejected word lists.

Reads the curated kept and rejected word lists with review metadata,
applies scoring rules based on review outcomes, and writes a single
scored list for use in the pipeline.

Scoring rules:
  Kept list:
    score 60                    → 60
    score 50, reject_count = 0  → 58
    score 50, reject_count > 0  → 56
  Reject list:
    score 60, keep_count > 0    → 55
    score 60, keep_count = 0    → 53
    score 50, keep_count > 0    → 52
    score 50, keep_count = 0    → 45
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DICT_DIR = Path(__file__).resolve().parent.parent / "dictionaries"
KEPT_FILE = DICT_DIR / "HggCuratedCrosswordList-KeptWithReason.txt"
REJECT_FILE = DICT_DIR / "HggCuratedCrosswordList-RejectWithReason.txt"
OUTPUT_FILE = DICT_DIR / "HggScoredCrosswordList.txt"


def score_kept(original_score: int, reject_count: int) -> int:
    if original_score == 60:
        return 60
    if reject_count == 0:
        return 58
    return 56


def score_rejected(original_score: int, keep_count: int) -> int:
    if original_score == 60:
        return 55 if keep_count > 0 else 53
    return 52 if keep_count > 0 else 45


def parse_line(line: str) -> tuple[str, int, int, int]:
    """Parse a line into (word, score, keep_count, reject_count)."""
    parts = line.split(";")
    word = parts[0]
    score = int(parts[1])
    keep_count = int(parts[2])
    reject_count = int(parts[3])
    return word, score, keep_count, reject_count


def build_scored_list() -> None:
    scored: dict[str, int] = {}
    bucket_counts: dict[int, int] = {60: 0, 58: 0, 56: 0, 55: 0, 53: 0, 52: 0, 45: 0}

    # Process kept list
    kept_count = 0
    with open(KEPT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, original_score, _keep_count, reject_count = parse_line(line)
            s = score_kept(original_score, reject_count)
            scored[word] = s
            bucket_counts[s] += 1
            kept_count += 1

    # Process reject list
    reject_count_total = 0
    with open(REJECT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, original_score, keep_count, _reject_count = parse_line(line)
            s = score_rejected(original_score, keep_count)
            scored[word] = s
            bucket_counts[s] += 1
            reject_count_total += 1

    # Write output sorted alphabetically
    with open(OUTPUT_FILE, "w") as f:
        for word in sorted(scored):
            f.write(f"{word};{scored[word]}\n")

    log.info("Kept entries:     %d", kept_count)
    log.info("Rejected entries: %d", reject_count_total)
    log.info("Total output:     %d", len(scored))
    log.info("")
    log.info("Score distribution:")
    for score in sorted(bucket_counts, reverse=True):
        log.info("  %d: %d entries", score, bucket_counts[score])
    log.info("")
    log.info("Wrote %s", OUTPUT_FILE)


if __name__ == "__main__":
    build_scored_list()
