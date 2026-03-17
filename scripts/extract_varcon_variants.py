#!/usr/bin/env python3
"""Extract spelling variants from Varcon for crossword dictionary removal.

Parses dictionaries/varcon.txt to identify words that are NOT
American-preferred spellings, then intersects with XwiJeffChenList.txt
to produce a removal list.

Usage:
    uv run python scripts/extract_varcon_variants.py
    uv run python scripts/extract_varcon_variants.py --dry-run
    uv run python scripts/extract_varcon_variants.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VARCON_PATH = REPO_ROOT / "dictionaries" / "varcon.txt"
DICTIONARY_PATH = REPO_ROOT / "dictionaries" / "XwiJeffChenList.txt"
OUTPUT_PATH = REPO_ROOT / "dictionaries" / "XwiJeffChenList-VariantsToRemove.txt"

# Variant modifiers — words with these are not preferred
VARIANT_MODIFIERS = {"v", "V", "-", "x"}


def parse_tags(tag_str: str) -> list[tuple[str, str]]:
    """Parse a space-separated tag string into (region, modifier) pairs.

    Examples:
        "A Bv C" -> [("A", ""), ("B", "v"), ("C", "")]
        "Av B" -> [("A", "v"), ("B", "")]
        "_v" -> [("_", "v")]
        "A. Bv" -> [("A", "."), ("B", "v")]
        "A B 1 2" -> [("A", ""), ("B", "")] (numbers are sense indices, ignored)
    """
    pairs = []
    for token in tag_str.split():
        # Skip numeric sense indices like "1", "2", "3"
        if token.isdigit():
            continue
        # Region is the first character, rest is the modifier
        if not token:
            continue
        region = token[0]
        modifier = token[1:] if len(token) > 1 else ""
        if region in ("A", "B", "Z", "C", "D", "_"):
            pairs.append((region, modifier))
    return pairs


def _has_bare_preferred(tags: list[tuple[str, str]]) -> bool:
    """Check if tags include a bare A or _ (unambiguously preferred)."""
    return any(
        region in ("A", "_") and modifier == ""
        for region, modifier in tags
    )


def is_american_preferred(tags: list[tuple[str, str]]) -> bool:
    """Check if any tag marks this word as American or non-regional preferred.

    A bare A/_ tag is always preferred. A. or _. (disputed/equal) is
    only treated as preferred if it was NOT demoted by a line-level
    sibling with a bare A/_ tag (demoted tags have modifier ".*").
    """
    for region, modifier in tags:
        if region in ("A", "_") and modifier in ("", "."):
            return True
    return False


def _parse_line_entries(
    line: str,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Parse a single varcon data line into [(word, tags), ...].

    Returns only lowercase, non-proper-noun, non-empty entries.
    """
    entries = line.split(" / ")
    results = []
    for entry in entries:
        entry = entry.strip()
        if ": " not in entry:
            continue
        tag_str, raw_word = entry.split(": ", 1)
        raw_word = raw_word.strip()

        # Skip proper nouns (capitalized words)
        if raw_word[0].isupper():
            continue

        # Normalize: lowercase, strip non-alpha chars
        word = re.sub(r"[^a-z]", "", raw_word.lower())
        if not word:
            continue

        tags = parse_tags(tag_str.strip())
        if not tags:
            continue

        results.append((word, tags))
    return results


def parse_varcon(
    varcon_path: Path,
) -> dict[str, list[tuple[str, str]]]:
    """Parse varcon.txt into {word: [(region, modifier), ...]}.

    When a line has an entry with bare A/_ (truly preferred), any
    sibling entries with only A./_.  (disputed) are demoted: their
    "." modifier is replaced with ".*" so they are no longer treated
    as preferred.
    """
    word_tags: dict[str, list[tuple[str, str]]] = {}

    with open(varcon_path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")

            # Skip blank lines, comments
            if not line or line.startswith("#"):
                continue

            # Strip pipe sections (sense/POS annotations)
            if " | " in line:
                line = line[: line.index(" | ")]

            # Skip possessives
            if "'s" in line:
                continue

            parsed = _parse_line_entries(line)
            if not parsed:
                continue

            # Check if any entry on this line is truly preferred
            line_has_preferred = any(
                _has_bare_preferred(tags) for _, tags in parsed
            )

            for word, tags in parsed:
                if line_has_preferred:
                    # Demote disputed (.) to non-preferred when a
                    # sibling is truly preferred
                    tags = [
                        (r, ".*" if m == "." else m)
                        for r, m in tags
                    ]

                if word not in word_tags:
                    word_tags[word] = []
                word_tags[word].extend(tags)

    return word_tags


def find_variants(
    word_tags: dict[str, list[tuple[str, str]]],
) -> set[str]:
    """Identify words that are NOT American-preferred spellings."""
    variants = set()
    for word, tags in word_tags.items():
        if not is_american_preferred(tags):
            variants.add(word)
    return variants


def load_dictionary(dict_path: Path) -> dict[str, str]:
    """Load dictionary file, return {word: score_str} mapping."""
    entries = {}
    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            word, score = line.split(";", 1)
            entries[word.lower()] = score
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Varcon spelling variants for removal."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts without writing output file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each variant found with its varcon tags.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Parse varcon
    logger.info("Parsing %s ...", VARCON_PATH)
    word_tags = parse_varcon(VARCON_PATH)
    logger.info("Found %d unique words in varcon.", len(word_tags))

    # Find variants
    variants = find_variants(word_tags)
    logger.info("Found %d variant (non-American-preferred) words.", len(variants))

    # Load dictionary
    logger.info("Loading dictionary from %s ...", DICTIONARY_PATH)
    dictionary = load_dictionary(DICTIONARY_PATH)
    logger.info("Dictionary contains %d entries.", len(dictionary))

    # Intersect
    to_remove = {}
    for word in sorted(variants):
        if word in dictionary:
            to_remove[word] = dictionary[word]

    logger.info(
        "Found %d variant words present in the dictionary.", len(to_remove)
    )

    if args.verbose:
        logger.info("\n--- Variants to remove ---")
        for word in sorted(to_remove):
            tags = word_tags[word]
            tag_summary = " ".join(
                f"{r}{m}" for r, m in tags
            )
            logger.info("  %s;%s  [varcon: %s]", word, to_remove[word], tag_summary)

    if args.dry_run:
        logger.info("\nDry run — no file written.")
        return

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for word in sorted(to_remove):
            f.write(f"{word};{to_remove[word]}\n")

    logger.info("Wrote %d entries to %s", len(to_remove), OUTPUT_PATH)


if __name__ == "__main__":
    main()
