"""Dictionary normalization utilities for batch generation experiments."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
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
    skipped_below_source_score: int
    skipped_outside_length: int
    duplicates: int


@dataclass
class _WordCollection:
    input_paths: tuple[Path, ...]
    words: set[str]
    total_lines: int = 0
    nonempty_lines: int = 0
    skipped_malformed: int = 0
    skipped_invalid_word: int = 0
    skipped_excluded: int = 0
    skipped_below_source_score: int = 0
    skipped_outside_length: int = 0
    duplicates: int = 0

    def extend(self, other: _WordCollection) -> None:
        self.input_paths += other.input_paths
        for word in other.words:
            if word in self.words:
                self.duplicates += 1
            else:
                self.words.add(word)
        self.total_lines += other.total_lines
        self.nonempty_lines += other.nonempty_lines
        self.skipped_malformed += other.skipped_malformed
        self.skipped_invalid_word += other.skipped_invalid_word
        self.skipped_excluded += other.skipped_excluded
        self.skipped_below_source_score += other.skipped_below_source_score
        self.skipped_outside_length += other.skipped_outside_length
        self.duplicates += other.duplicates


def prepare_flat_dictionary(
    input_path: Path | str,
    output_path: Path | str,
    *,
    score: int = 55,
    extra_input_paths: Sequence[Path | str] = (),
    exclude_words: Collection[str] = (),
    min_source_score_by_length: Mapping[int, int] | None = None,
    flat_score_input_paths: Collection[Path | str] = (),
    min_word_length: int | None = None,
    max_word_length: int | None = None,
) -> DictionaryPrepSummary:
    """Normalize plain or scored word lists to ``WORD;SCORE`` lines.

    Input lines may be either plain words or ``word;score`` rows. Existing
    source scores are validated when present, and can be used for
    length-specific filtering. Paths listed in ``flat_score_input_paths`` are
    previously flattened dictionaries, so their scores are ignored for
    source-score filtering. The output score is always the flat score passed
    here.
    """
    input_paths = (Path(input_path),) + tuple(Path(p) for p in extra_input_paths)
    dst = Path(output_path)
    collection = _collect_words(
        input_paths,
        exclude_words=exclude_words,
        min_source_score_by_length=min_source_score_by_length,
        flat_score_input_paths=flat_score_input_paths,
        min_word_length=min_word_length,
        max_word_length=max_word_length,
    )
    _write_flat_words(dst, collection.words, score)

    return _summary_from_collection(collection, dst, score)


def prepare_length_mixed_flat_dictionary(
    short_input_path: Path | str,
    long_input_path: Path | str,
    output_path: Path | str,
    *,
    score: int = 55,
    short_extra_input_paths: Sequence[Path | str] = (),
    short_max_length: int = 5,
    long_min_length: int = 6,
    exclude_words: Collection[str] = (),
    min_source_score_by_length: Mapping[int, int] | None = None,
    flat_score_input_paths: Collection[Path | str] = (),
) -> DictionaryPrepSummary:
    """Build a flat dictionary from accessible short fill plus longer hard fill."""
    short_paths = (Path(short_input_path),) + tuple(
        Path(p) for p in short_extra_input_paths
    )
    long_paths = (Path(long_input_path),)
    dst = Path(output_path)

    short_collection = _collect_words(
        short_paths,
        exclude_words=exclude_words,
        flat_score_input_paths=flat_score_input_paths,
        max_word_length=short_max_length,
    )
    long_collection = _collect_words(
        long_paths,
        exclude_words=exclude_words,
        min_source_score_by_length=min_source_score_by_length,
        flat_score_input_paths=flat_score_input_paths,
        min_word_length=long_min_length,
    )
    short_collection.extend(long_collection)
    _write_flat_words(dst, short_collection.words, score)

    return _summary_from_collection(short_collection, dst, score)


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
            f"Skipped below source-score floor: {summary.skipped_below_source_score}",
            f"Skipped outside length bounds: {summary.skipped_outside_length}",
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
            parsed = _parse_word_and_score(line.split(";", 1)[0])
            if parsed is not None:
                word, _score = parsed
                words.add(word.upper())
    return words


def _parse_word_and_score(line: str) -> tuple[str, int | None] | None:
    if ";" not in line:
        return line, None

    parts = line.split(";")
    if len(parts) != 2:
        return None

    word, score = parts[0].strip(), parts[1].strip()
    if not word:
        return None

    try:
        parsed_score = int(score)
    except ValueError:
        return None

    return word, parsed_score


def _is_valid_crossword_word(word: str) -> bool:
    return word.isascii() and word.isalpha()


def _collect_words(
    input_paths: Sequence[Path],
    *,
    exclude_words: Collection[str] = (),
    min_source_score_by_length: Mapping[int, int] | None = None,
    flat_score_input_paths: Collection[Path | str] = (),
    min_word_length: int | None = None,
    max_word_length: int | None = None,
) -> _WordCollection:
    exclusions = {word.upper() for word in exclude_words}
    min_source_scores = dict(min_source_score_by_length or {})
    flat_score_paths = {Path(path).resolve() for path in flat_score_input_paths}
    collection = _WordCollection(input_paths=tuple(input_paths), words=set())

    for src in input_paths:
        for raw_line in src.read_text().splitlines():
            collection.total_lines += 1
            line = raw_line.strip()
            if not line:
                continue
            collection.nonempty_lines += 1

            parsed = _parse_word_and_score(line)
            if parsed is None:
                collection.skipped_malformed += 1
                continue

            parsed_word, source_score = parsed
            if src.resolve() in flat_score_paths:
                source_score = None
            word = parsed_word.upper()
            if not _is_valid_crossword_word(word):
                collection.skipped_invalid_word += 1
                continue

            if min_word_length is not None and len(word) < min_word_length:
                collection.skipped_outside_length += 1
                continue
            if max_word_length is not None and len(word) > max_word_length:
                collection.skipped_outside_length += 1
                continue

            min_source_score = min_source_scores.get(len(word))
            if (
                min_source_score is not None
                and source_score is not None
                and source_score < min_source_score
            ):
                collection.skipped_below_source_score += 1
                continue

            if word in exclusions:
                collection.skipped_excluded += 1
                continue

            if word in collection.words:
                collection.duplicates += 1
                continue

            collection.words.add(word)

    return collection


def _write_flat_words(output_path: Path, words: Collection[str], score: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{word};{score}\n" for word in sorted(words)))


def _summary_from_collection(
    collection: _WordCollection,
    output_path: Path,
    score: int,
) -> DictionaryPrepSummary:
    return DictionaryPrepSummary(
        input_path=collection.input_paths[0],
        input_paths=collection.input_paths,
        output_path=output_path,
        score=score,
        total_lines=collection.total_lines,
        nonempty_lines=collection.nonempty_lines,
        output_count=len(collection.words),
        skipped_malformed=collection.skipped_malformed,
        skipped_invalid_word=collection.skipped_invalid_word,
        skipped_excluded=collection.skipped_excluded,
        skipped_below_source_score=collection.skipped_below_source_score,
        skipped_outside_length=collection.skipped_outside_length,
        duplicates=collection.duplicates,
    )
