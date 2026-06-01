"""Build and publish effective crossword dictionaries."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crossword_generator.data_store import AUTHOR, request_admin_json
from crossword_generator.dictionary_prep import (
    DictionaryPrepSummary,
    load_excluded_words,
    prepare_hgg_easy_dictionary,
    prepare_sixty_dictionary,
)

PUBLISH_PATH = "/admin/crossword-effective-dictionaries/publish"
SCHEMA_VERSION = 1
MAX_PAYLOAD_BYTES = 1_000_000
WORD_RE = re.compile(r"^[A-Z]+$")


class EffectiveDictionaryError(RuntimeError):
    """Raised when effective dictionary build, validation, or publish fails."""


@dataclass(frozen=True)
class SourceFile:
    """Input file used to build an effective dictionary snapshot."""

    role: str
    path: Path


@dataclass(frozen=True)
class EffectiveDictionary:
    """One generated effective dictionary."""

    slug: str
    label: str
    path: Path
    score: int
    min_length: int
    max_length: int
    entries: tuple[tuple[str, int], ...]

    @property
    def contents(self) -> str:
        return "".join(f"{word};{score}\n" for word, score in self.entries)

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def sha256(self) -> str:
        return _sha256_text(self.contents)

    @property
    def length_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for word, _score in self.entries:
            length = str(len(word))
            counts[length] = counts.get(length, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: int(item[0])))


@dataclass(frozen=True)
class EffectiveDictionaryBuild:
    """Result of building the effective dictionary snapshot locally."""

    easy: EffectiveDictionary
    sixty: EffectiveDictionary
    source_files: tuple[SourceFile, ...]
    source_fingerprint: str
    easy_summary: DictionaryPrepSummary
    sixty_summary: DictionaryPrepSummary
    project_root: Path


@dataclass(frozen=True)
class EffectiveDictionaryPublishResult:
    """Result returned after publishing an effective dictionary snapshot."""

    action: str
    response: dict[str, Any]


def build_effective_dictionaries(
    *,
    project_root: Path,
    output_dir: Path,
    easy_source: Path,
    easy_extra_sources: tuple[Path, ...],
    easy_exclude_sources: tuple[Path, ...],
    hard_source: Path,
    sixty_source: Path,
    score: int = 50,
) -> EffectiveDictionaryBuild:
    """Build HGG Easy and HGG 60 into output_dir and validate the result.

    ``sixty_source`` is the scored master list (XwiJeffChenList.txt) that
    provides the source-score 60 entries. ``easy_source`` / ``hard_source``
    are Jeff's plain fill lists (HGGXW-Easy.txt / HGGXW-Hard.txt) and carry
    no scores, so the 60-pointers must come from ``sixty_source``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbs_easy = project_root / "dictionaries" / "HggThumbsDownEasy.txt"
    base_excluded_words = load_excluded_words(easy_exclude_sources)
    extra_easy_exclude = (
        load_excluded_words([thumbs_easy]) if thumbs_easy.exists() else set()
    )
    excluded_easy_words = base_excluded_words | extra_easy_exclude

    easy_path = output_dir / "hgg-easy.txt"
    sixty_path = output_dir / "hgg-60.txt"

    easy_summary = prepare_hgg_easy_dictionary(
        easy_source,
        sixty_source,
        easy_path,
        score=score,
        extra_input_paths=easy_extra_sources,
        exclude_words=excluded_easy_words,
    )
    sixty_summary = prepare_sixty_dictionary(
        sixty_source,
        sixty_path,
        score=60,
        exclude_words=base_excluded_words,
    )

    easy = EffectiveDictionary(
        slug="hgg-easy",
        label="HGG Easy",
        path=easy_path,
        score=score,
        min_length=3,
        max_length=9,
        entries=_load_entries(easy_path),
    )
    sixty = EffectiveDictionary(
        slug="hgg-60",
        label="HGG 60",
        path=sixty_path,
        score=60,
        min_length=7,
        max_length=9,
        entries=_load_entries(sixty_path),
    )

    source_files = (
        SourceFile("easy-source", easy_source),
        *(SourceFile("easy-extra-source", path) for path in easy_extra_sources),
        *(SourceFile("easy-exclude-source", path) for path in easy_exclude_sources),
        *(
            (SourceFile("easy-thumbs-down", thumbs_easy),)
            if thumbs_easy.exists()
            else ()
        ),
        SourceFile("hard-source", hard_source),
        SourceFile("sixty-source", sixty_source),
    )
    build = EffectiveDictionaryBuild(
        easy=easy,
        sixty=sixty,
        source_files=source_files,
        source_fingerprint=_source_fingerprint(source_files),
        easy_summary=easy_summary,
        sixty_summary=sixty_summary,
        project_root=project_root,
    )
    validate_effective_dictionaries(build, sixty_source=sixty_source)
    return build


def validate_effective_dictionaries(
    build: EffectiveDictionaryBuild,
    *,
    sixty_source: Path,
) -> None:
    """Validate the HGG Easy and HGG 60 invariants before publish.

    ``sixty_source`` is the scored master list; source-score 60 entries are
    read from it to assert none leaked into hgg-easy.
    """
    for dictionary in (build.easy, build.sixty):
        if not dictionary.entries:
            raise EffectiveDictionaryError(f"{dictionary.slug} is empty")
        words_seen: set[str] = set()
        for word, score in dictionary.entries:
            if WORD_RE.fullmatch(word) is None:
                raise EffectiveDictionaryError(
                    f"{dictionary.slug} has invalid word: {word!r}"
                )
            if word in words_seen:
                raise EffectiveDictionaryError(
                    f"{dictionary.slug} has duplicate word: {word}"
                )
            words_seen.add(word)
            if score != dictionary.score:
                raise EffectiveDictionaryError(
                    f"{dictionary.slug} has {word};{score}, "
                    f"expected score {dictionary.score}"
                )
            if not dictionary.min_length <= len(word) <= dictionary.max_length:
                raise EffectiveDictionaryError(
                    f"{dictionary.slug} has {word} with length {len(word)}, "
                    f"expected {dictionary.min_length}-{dictionary.max_length}"
                )

    source_sixties = _collect_source_score_words(
        sixty_source,
        min_score=60,
        min_word_length=3,
        max_word_length=9,
    )
    easy_words = {word for word, _score in build.easy.entries}
    sixty_words = {word for word, _score in build.sixty.entries}
    easy_sixties = easy_words & source_sixties
    if easy_sixties:
        sample = ", ".join(sorted(easy_sixties)[:5])
        raise EffectiveDictionaryError(
            f"hgg-easy includes source-score 60 word(s): {sample}"
        )
    overlap = easy_words & sixty_words
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise EffectiveDictionaryError(
            f"hgg-easy and hgg-60 overlap unexpectedly: {sample}"
        )


def make_effective_dictionary_payload(
    build: EffectiveDictionaryBuild,
    *,
    published_at: datetime | None = None,
    generator_commit: str | None = None,
) -> dict[str, Any]:
    """Package both effective dictionaries into one atomic publish payload."""
    timestamp = published_at or datetime.now(UTC)
    published = timestamp.isoformat().replace("+00:00", "Z")
    commit = generator_commit or _current_git_commit()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "published_at": published,
        "generator_commit": commit,
        "author": AUTHOR,
        "source_fingerprint": build.source_fingerprint,
        "source_files": [
            _source_file_metadata(item, build.project_root)
            for item in build.source_files
        ],
        "dictionaries": {
            build.easy.slug: _dictionary_payload(build.easy),
            build.sixty.slug: _dictionary_payload(build.sixty),
        },
        "runtime_recipes": [
            {
                "puzzle": "easy/5",
                "uses": ["hgg-easy"],
            },
            {
                "puzzle": "easy/7",
                "uses": ["hgg-easy"],
            },
            {
                "puzzle": "hard/5",
                "uses": ["hgg-easy"],
                "note": "Same as Easy pending Jeff guidance.",
            },
            {
                "puzzle": "hard/7",
                "uses": ["hgg-easy", "hgg-60"],
                "constraint": "Exactly one 7-letter hgg-60 entry.",
            },
            {
                "puzzle": "easy/9",
                "uses": ["hgg-easy"],
            },
            {
                "puzzle": "hard/9",
                "uses": ["hgg-easy", "hgg-60"],
                "constraint": (
                    "Only 8-9 letter entries use hgg-60; 7-letter hgg-60 "
                    "entries are reserved for hard/7."
                ),
            },
        ],
        "metadata": {
            "artifact_type": "crossword_effective_dictionaries",
            "author": AUTHOR,
            "schema_version": SCHEMA_VERSION,
            "generator_commit": commit,
            "published_at": published,
            "source_fingerprint": build.source_fingerprint,
            "dictionary_slugs": [build.easy.slug, build.sixty.slug],
            "hgg_easy_count": build.easy.count,
            "hgg_60_count": build.sixty.count,
            "hgg_easy_sha256": build.easy.sha256,
            "hgg_60_sha256": build.sixty.sha256,
            "active": True,
        },
    }
    size = len(json.dumps(payload).encode())
    if size > MAX_PAYLOAD_BYTES:
        raise EffectiveDictionaryError(
            f"Effective dictionary payload exceeds {MAX_PAYLOAD_BYTES} bytes: {size}"
        )
    return payload


def publish_effective_dictionaries(
    build: EffectiveDictionaryBuild,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    generator_commit: str | None = None,
) -> EffectiveDictionaryPublishResult:
    """Publish the active effective-dictionaries snapshot through the admin API."""
    payload = make_effective_dictionary_payload(
        build,
        generator_commit=generator_commit,
    )
    response = request_admin_json(
        "POST",
        PUBLISH_PATH,
        {"snapshot": payload},
        api_base=api_base,
        token=token,
        timeout=timeout,
    )
    action = str(response.get("action", "published"))
    return EffectiveDictionaryPublishResult(action=action, response=response)


def _dictionary_payload(dictionary: EffectiveDictionary) -> dict[str, Any]:
    return {
        "slug": dictionary.slug,
        "label": dictionary.label,
        "score": dictionary.score,
        "min_length": dictionary.min_length,
        "max_length": dictionary.max_length,
        "entry_count": dictionary.count,
        "length_counts": dictionary.length_counts,
        "sha256": dictionary.sha256,
        "format": "word_semicolon_score",
        "contents": dictionary.contents,
    }


def _source_file_metadata(source: SourceFile, project_root: Path) -> dict[str, Any]:
    body = source.path.read_bytes()
    return {
        "role": source.role,
        "path": _relative_source_path(source.path, project_root),
        "line_count": len(source.path.read_text().splitlines()),
        "byte_count": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def _relative_source_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _source_fingerprint(source_files: tuple[SourceFile, ...]) -> str:
    digest = hashlib.sha256()
    for source in source_files:
        digest.update(source.role.encode())
        digest.update(b"\0")
        digest.update(str(source.path).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(source.path.read_bytes()).hexdigest().encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_entries(path: Path) -> tuple[tuple[str, int], ...]:
    entries: list[tuple[str, int]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        word, score_text = line.split(";", 1)
        entries.append((word.strip().upper(), int(score_text.strip())))
    return tuple(entries)


def _collect_source_score_words(
    path: Path,
    *,
    min_score: int,
    min_word_length: int,
    max_word_length: int,
) -> set[str]:
    words: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or ";" not in line:
            continue
        word_text, score_text = line.split(";", 1)
        try:
            score = int(score_text.strip())
        except ValueError:
            continue
        word = word_text.strip().upper()
        if (
            score >= min_score
            and WORD_RE.fullmatch(word) is not None
            and min_word_length <= len(word) <= max_word_length
        ):
            words.add(word)
    return words


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
