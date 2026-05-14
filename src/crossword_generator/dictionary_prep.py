"""Dictionary normalization utilities for batch generation experiments."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DictionaryPrepSummary:
    """Summary of a dictionary normalization run."""

    input_path: Path
    input_paths: tuple[Path, ...]
    output_path: Path
    score: int
    total_lines: int
    nonempty_lines: int
    output_count: int
    skipped_malformed: int
    skipped_invalid_word: int
    skipped_excluded: int
    duplicates: int


def prepare_flat_dictionary(
    input_path: Path | str,
    output_path: Path | str,
    *,
    score: int = 55,
    extra_input_paths: Sequence[Path | str] = (),
    exclude_words: Collection[str] = (),
) -> DictionaryPrepSummary:
    """Normalize plain or scored word lists to ``WORD;SCORE`` lines.

    Input lines may be either plain words or ``word;score`` rows. Existing
    source scores are validated when present, but the output score is always
    the flat score passed here.
    """
    input_paths = (Path(input_path),) + tuple(Path(p) for p in extra_input_paths)
    dst = Path(output_path)
    exclusions = {word.upper() for word in exclude_words}

    words: set[str] = set()
    total_lines = 0
    nonempty_lines = 0
    skipped_malformed = 0
    skipped_invalid_word = 0
    skipped_excluded = 0
    duplicates = 0

    for src in input_paths:
        for raw_line in src.read_text().splitlines():
            total_lines += 1
            line = raw_line.strip()
            if not line:
                continue
            nonempty_lines += 1

            parsed = _parse_word(line)
            if parsed is None:
                skipped_malformed += 1
                continue

            word = parsed.upper()
            if not _is_valid_crossword_word(word):
                skipped_invalid_word += 1
                continue

            if word in exclusions:
                skipped_excluded += 1
                continue

            if word in words:
                duplicates += 1
                continue

            words.add(word)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(f"{word};{score}\n" for word in sorted(words)))

    return DictionaryPrepSummary(
        input_path=input_paths[0],
        input_paths=input_paths,
        output_path=dst,
        score=score,
        total_lines=total_lines,
        nonempty_lines=nonempty_lines,
        output_count=len(words),
        skipped_malformed=skipped_malformed,
        skipped_invalid_word=skipped_invalid_word,
        skipped_excluded=skipped_excluded,
        duplicates=duplicates,
    )


def format_summary(summary: DictionaryPrepSummary) -> str:
    """Format a summary for CLI/log output."""
    additional_inputs = ", ".join(str(p) for p in summary.input_paths[1:])
    return "\n".join(
        [
            f"Input: {summary.input_path}",
            *(
                [f"Additional inputs: {additional_inputs}"]
                if len(summary.input_paths) > 1
                else []
            ),
            f"Output: {summary.output_path}",
            f"Score used: {summary.score}",
            f"Total lines: {summary.total_lines}",
            f"Non-empty input rows: {summary.nonempty_lines}",
            f"Output rows: {summary.output_count}",
            f"Skipped malformed rows: {summary.skipped_malformed}",
            f"Skipped invalid words: {summary.skipped_invalid_word}",
            f"Skipped excluded words: {summary.skipped_excluded}",
            f"Duplicates skipped: {summary.duplicates}",
        ]
    )


def load_excluded_words(paths: Sequence[Path | str]) -> set[str]:
    """Load excluded words from plain or semicolon-delimited word lists."""
    words: set[str] = set()
    for path in paths:
        src = Path(path)
        for raw_line in src.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parsed = _parse_word(line.split(";", 1)[0])
            if parsed is not None:
                words.add(parsed.upper())
    return words


def _parse_word(line: str) -> str | None:
    if ";" not in line:
        return line

    parts = line.split(";")
    if len(parts) != 2:
        return None

    word, score = parts[0].strip(), parts[1].strip()
    if not word:
        return None

    try:
        int(score)
    except ValueError:
        return None

    return word


def _is_valid_crossword_word(word: str) -> bool:
    return word.isascii() and word.isalpha()
