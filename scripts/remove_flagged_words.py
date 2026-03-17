#!/usr/bin/env python3
"""Remove flagged words from Hgg dictionaries.

Loads two removal lists (not-family-friendly and spelling variants) and
strips matching entries from both HggScoredCrosswordList.txt and
HggCuratedCrosswordList.txt. Removed entries are written to sibling
-Removed.txt files with the reason for removal.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DICT_DIR = Path(__file__).resolve().parent.parent / "dictionaries"

REMOVAL_SOURCES = [
    (DICT_DIR / "XwiJeffChenList-NotFamilyFriendly.txt", "not-family-friendly"),
    (DICT_DIR / "XwiJeffChenList-VariantsToRemove.txt", None),  # reason from file
]

DICTIONARIES = [
    DICT_DIR / "HggScoredCrosswordList.txt",
    DICT_DIR / "HggCuratedCrosswordList.txt",
]


def load_removal_set() -> dict[str, str]:
    """Build {word: reason} from both removal lists."""
    removals: dict[str, str] = {}
    for path, default_reason in REMOVAL_SOURCES:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                word = parts[0]
                if default_reason:
                    reason = default_reason
                elif len(parts) > 2:
                    reason = parts[2]
                else:
                    reason = "unknown"
                removals[word] = reason
    log.info("Loaded %d words to remove", len(removals))
    return removals


def process_dictionary(
    dict_path: Path, removals: dict[str, str], *, dry_run: bool
) -> None:
    """Remove flagged words from a single dictionary file."""
    kept: list[str] = []
    removed: list[str] = []

    with open(dict_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word = line.split(";")[0]
            if word in removals:
                removed.append(f"{line};{removals[word]}")
            else:
                kept.append(line)

    original_count = len(kept) + len(removed)
    log.info("")
    log.info("%s", dict_path.name)
    log.info("  Original:  %d entries", original_count)
    log.info("  Removed:   %d entries", len(removed))
    log.info("  Remaining: %d entries", len(kept))

    if dry_run:
        log.info("  (dry run — no files modified)")
        return

    with open(dict_path, "w") as f:
        for entry in kept:
            f.write(f"{entry}\n")

    removed_path = dict_path.with_name(dict_path.stem + "-Removed.txt")
    with open(removed_path, "w") as f:
        for entry in sorted(removed):
            f.write(f"{entry}\n")

    log.info("  Wrote %s", dict_path.name)
    log.info("  Wrote %s", removed_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove flagged words from Hgg dictionaries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts without modifying files",
    )
    args = parser.parse_args()

    removals = load_removal_set()
    for dict_path in DICTIONARIES:
        process_dictionary(dict_path, removals, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
