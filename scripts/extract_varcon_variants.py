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
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Parse a varcon data line into [(normalized, original, tags), ...].

    ``normalized`` is the lowercase, alpha-only form for matching.
    ``original`` is the raw word as it appears in varcon.
    """
    entries = line.split(" / ")
    results: list[tuple[str, str, list[tuple[str, str]]]] = []
    for entry in entries:
        entry = entry.strip()
        if ": " not in entry:
            continue
        tag_str, raw_word = entry.split(": ", 1)
        raw_word = raw_word.strip()

        # Normalize: lowercase, strip non-alpha chars
        word = re.sub(r"[^a-z]", "", raw_word.lower())
        if not word:
            continue

        tags = parse_tags(tag_str.strip())
        if not tags:
            continue

        results.append((word, raw_word, tags))
    return results


REGION_NAMES = {
    "A": "American",
    "B": "British",
    "Z": "British -ize",
    "C": "Canadian",
    "D": "Australian",
    "_": "non-regional",
}

MODIFIER_NAMES = {
    "": "preferred",
    ".": "disputed",
    ".*": "disputed (demoted)",
    "v": "variant",
    "V": "seldom-used variant",
    "-": "possible",
    "x": "improper",
}


def _describe_tags(tags: list[tuple[str, str]]) -> str:
    """Produce a human-readable description of a word's tags."""
    parts = []
    for region, modifier in tags:
        r = REGION_NAMES.get(region, region)
        m = MODIFIER_NAMES.get(modifier, modifier)
        parts.append(f"{r} {m}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return ", ".join(unique)


def _format_preferred(original: str) -> str:
    """Format a preferred form, marking proper nouns."""
    if original[0].isupper():
        return f"{original} (proper)"
    return original


def _format_reason(
    original_forms: list[str],
    tags: list[tuple[str, str]],
    preferred_originals: list[str],
) -> str:
    """Build a reason string with original case and proper-noun markers.

    Example: 'British variant; preferred: color'
    Example: 'British preferred; preferred: Acer (proper)'
    """
    # Mark if this word itself is only from proper nouns
    desc = _describe_tags(tags)
    if any(f[0].isupper() for f in original_forms):
        desc = f"[proper] {desc}"

    if preferred_originals:
        # Deduplicate preferred forms, preserving order
        seen: set[str] = set()
        unique_pref = []
        for p in preferred_originals:
            if p not in seen:
                seen.add(p)
                unique_pref.append(p)
        alts = [_format_preferred(p) for p in unique_pref]
        if alts:
            return f"{desc}; preferred: {', '.join(alts)}"
    return desc


def parse_varcon(
    varcon_path: Path,
) -> tuple[
    dict[str, list[tuple[str, str]]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    """Parse varcon.txt.

    Returns:
        word_tags: {normalized: [(region, modifier), ...]}
        word_preferred: {normalized: [preferred_original, ...]}
        word_originals: {normalized: [original_form, ...]}

    When a line has an entry with bare A/_ (truly preferred), any
    sibling entries with only A./_.  (disputed) are demoted: their
    "." modifier is replaced with ".*" so they are no longer treated
    as preferred.
    """
    word_tags: dict[str, list[tuple[str, str]]] = {}
    word_preferred: dict[str, list[str]] = {}
    word_originals: dict[str, list[str]] = {}

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
                _has_bare_preferred(tags)
                for _, _, tags in parsed
            )

            # Collect preferred original forms on this line
            preferred_on_line = [
                orig
                for _, orig, tags in parsed
                if _has_bare_preferred(tags)
            ]

            for word, orig, tags in parsed:
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

                if word not in word_preferred:
                    word_preferred[word] = []
                word_preferred[word].extend(preferred_on_line)

                if word not in word_originals:
                    word_originals[word] = []
                word_originals[word].append(orig)

    return word_tags, word_preferred, word_originals


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
    word_tags, word_preferred, word_originals = parse_varcon(VARCON_PATH)
    logger.info("Found %d unique words in varcon.", len(word_tags))

    # Find variants
    variants = find_variants(word_tags)
    logger.info("Found %d variant (non-American-preferred) words.", len(variants))

    # Load dictionary
    logger.info("Loading dictionary from %s ...", DICTIONARY_PATH)
    dictionary = load_dictionary(DICTIONARY_PATH)
    logger.info("Dictionary contains %d entries.", len(dictionary))

    # Intersect and build reasons
    to_remove: dict[str, tuple[str, str]] = {}
    for word in sorted(variants):
        if word in dictionary:
            reason = _format_reason(
                word_originals.get(word, [word]),
                word_tags[word],
                word_preferred.get(word, []),
            )
            to_remove[word] = (dictionary[word], reason)

    logger.info(
        "Found %d variant words present in the dictionary.", len(to_remove)
    )

    if args.verbose:
        logger.info("\n--- Variants to remove ---")
        for word in sorted(to_remove):
            score, reason = to_remove[word]
            logger.info("  %s;%s;%s", word, score, reason)

    if args.dry_run:
        logger.info("\nDry run — no file written.")
        return

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for word in sorted(to_remove):
            score, reason = to_remove[word]
            f.write(f"{word};{score};{reason}\n")

    logger.info("Wrote %d entries to %s", len(to_remove), OUTPUT_PATH)


if __name__ == "__main__":
    main()
