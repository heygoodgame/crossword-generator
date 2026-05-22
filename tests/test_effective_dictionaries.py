"""Tests for effective dictionary snapshot build and publish."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crossword_generator import data_store
from crossword_generator.effective_dictionaries import (
    PUBLISH_PATH,
    EffectiveDictionaryBuild,
    build_effective_dictionaries,
    make_effective_dictionary_payload,
    publish_effective_dictionaries,
)


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    dictionaries = tmp_path / "dictionaries"
    dictionaries.mkdir()
    easy_source = dictionaries / "easy.txt"
    easy_extra = dictionaries / "extra.txt"
    exclude_source = dictionaries / "exclude.txt"
    hard_source = dictionaries / "hard.txt"
    thumbs_easy = dictionaries / "HggThumbsDownEasy.txt"

    easy_source.write_text("APPLE;55\nBANANA;55\nALCHEMY;55\nNOVICE;55\n")
    easy_extra.write_text("ZOOMING\n")
    exclude_source.write_text("BANANA;removed\n")
    hard_source.write_text(
        "ALCHEMY;60\nAIRLIFT;60\nNOVICE;50\nTOOLONGWORD;60\n"
    )
    thumbs_easy.write_text("APPLE;easy thumbs-down\n")
    return easy_source, easy_extra, exclude_source, hard_source, dictionaries


def _build(tmp_path: Path) -> EffectiveDictionaryBuild:
    easy_source, easy_extra, exclude_source, hard_source, _dictionaries = (
        _write_sources(tmp_path)
    )
    return build_effective_dictionaries(
        project_root=tmp_path,
        output_dir=tmp_path / "out",
        easy_source=easy_source,
        easy_extra_sources=(easy_extra,),
        easy_exclude_sources=(exclude_source,),
        hard_source=hard_source,
    )


def test_build_effective_dictionaries_applies_excludes_and_scores(
    tmp_path: Path,
) -> None:
    build = _build(tmp_path)

    assert build.easy.count == 2
    assert build.easy.entries == (("NOVICE", 50), ("ZOOMING", 50))
    assert build.sixty.entries == (("AIRLIFT", 60), ("ALCHEMY", 60))
    assert build.easy.length_counts == {"6": 1, "7": 1}
    assert build.sixty.length_counts == {"7": 2}


def test_make_payload_packages_both_dictionaries_as_one_snapshot(
    tmp_path: Path,
) -> None:
    build = _build(tmp_path)

    payload = make_effective_dictionary_payload(
        build,
        generator_commit="abc123",
    )

    recipes = {recipe["puzzle"]: recipe for recipe in payload["runtime_recipes"]}
    assert recipes["easy/5"]["game_key"] == "minicrossword"
    assert recipes["hard/7"]["game_key"] == "minicrossword"
    assert recipes["easy/9"]["game_key"] == "midicrossword"
    assert payload["metadata"]["hgg_easy_count"] == 2
    assert payload["metadata"]["hgg_60_count"] == 2
    assert payload["dictionaries"]["hgg-easy"]["contents"] == (
        "NOVICE;50\nZOOMING;50\n"
    )
    assert payload["dictionaries"]["hgg-60"]["contents"] == (
        "AIRLIFT;60\nALCHEMY;60\n"
    )


def test_publish_effective_dictionaries_posts_one_atomic_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    build = _build(tmp_path)
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        calls.append((method, path, body))
        if method == "POST":
            return {"action": "published"}
        raise AssertionError(f"Unexpected call: {method} {path}")

    monkeypatch.setattr(data_store, "_request_json", fake_request)

    result = publish_effective_dictionaries(
        build,
        token="token",
        generator_commit="abc123",
    )

    assert result.action == "published"
    assert calls[0][0] == "POST"
    assert calls[0][1] == PUBLISH_PATH
    posted = calls[0][2]
    assert posted is not None
    assert set(posted["snapshot"]["dictionaries"]) == {"hgg-easy", "hgg-60"}
