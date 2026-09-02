"""Tests for `crossword-generator consolidate-list`."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from crossword_generator import consolidate_list as cl
from crossword_generator.cli import main


def test_diff_words_ignores_score_changes() -> None:
    old = "APPLE;55\nBANANA;55\n"
    new = "APPLE;60\nCHERRY;55\n"
    added, removed = cl.diff_words(old, new)
    assert added == {"CHERRY"}
    assert removed == {"BANANA"}


def test_diff_words_ignores_blank_lines() -> None:
    added, removed = cl.diff_words("APPLE\n\n", "APPLE\nBERRY\n")
    assert added == {"BERRY"}
    assert removed == set()


def test_consolidate_one_writes_file_and_acks_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_rel = "dictionaries/sample.txt"
    target = tmp_path / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("OLDWORD;55\nKEEPER;55\n")

    calls: list[tuple[str, str]] = []

    def fake_metadata(slug, **_kw):
        calls.append(("metadata", slug))
        return {"slug": slug, "file_path": target_rel}

    def fake_contents(slug, **_kw):
        calls.append(("download", slug))
        return "KEEPER;55\nNEWBIE;55\n"

    def fake_mark(slug, **_kw):
        calls.append(("mark", slug))

    monkeypatch.setattr(cl, "fetch_list_metadata", fake_metadata)
    monkeypatch.setattr(cl, "fetch_list_contents", fake_contents)
    monkeypatch.setattr(cl, "mark_consolidated", fake_mark)

    summary = cl.consolidate_one("sample", tmp_path)

    assert summary.added == 1
    assert summary.removed == 1
    assert summary.wrote is True
    assert target.read_text() == "KEEPER;55\nNEWBIE;55\n"
    assert calls == [("metadata", "sample"), ("download", "sample"), ("mark", "sample")]


def test_consolidate_one_dry_run_skips_write_and_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_rel = "dictionaries/sample.txt"
    target = tmp_path / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("OLDWORD;55\n")

    marked: list[str] = []
    monkeypatch.setattr(
        cl,
        "fetch_list_metadata",
        lambda slug, **_kw: {"slug": slug, "file_path": target_rel},
    )
    monkeypatch.setattr(
        cl,
        "fetch_list_contents",
        lambda _slug, **_kw: "NEWBIE;55\n",
    )
    monkeypatch.setattr(
        cl,
        "mark_consolidated",
        lambda slug, **_kw: marked.append(slug),
    )

    summary = cl.consolidate_one("sample", tmp_path, dry_run=True)

    assert summary.added == 1
    assert summary.removed == 1
    assert summary.wrote is False
    assert target.read_text() == "OLDWORD;55\n"
    assert marked == []


def test_consolidate_list_command_iterates_all_slugs_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_list(**_kw):
        return [
            {"slug": "list-a", "file_path": "dictionaries/a.txt"},
            {"slug": "list-b", "file_path": "dictionaries/b.txt"},
        ]

    def fake_consolidate(slug, project_root, **_kw):
        seen.append(slug)
        return cl.ListSummary(
            slug=slug,
            file_path=f"dictionaries/{slug}.txt",
            added=0,
            removed=0,
            total_after=0,
            wrote=False,
        )

    monkeypatch.setattr(cl, "list_registered_lists", fake_list)
    monkeypatch.setattr(cl, "consolidate_one", fake_consolidate)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["consolidate-list", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert seen == ["list-a", "list-b"]
    assert "Dry run" in result.output


def test_dedupe_lines_keeps_first_occurrence_and_counts() -> None:
    body = "ALLAY\nBABES\nCDS\nALLAY\nBABES\nDREGS\n"
    cleaned, repeated = cl.dedupe_lines(body)
    assert cleaned == "ALLAY\nBABES\nCDS\nDREGS\n"
    assert repeated == 2
    # A changed score is a different row, not a repeat; blank lines pass.
    assert cl.dedupe_lines("WORD;55\nWORD;60\n\n") == ("WORD;55\nWORD;60\n\n", 0)


def test_consolidate_one_dedupes_repeated_download_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """hey-you's chunked download repeats rows at chunk boundaries."""
    target_rel = "dictionaries/sample.txt"
    target = tmp_path / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        cl, "fetch_list_metadata", lambda slug, **_kw: {"file_path": target_rel}
    )
    monkeypatch.setattr(
        cl, "fetch_list_contents", lambda slug, **_kw: "ALLAY\nBABES\nALLAY\nCDS\n"
    )
    monkeypatch.setattr(cl, "mark_consolidated", lambda slug, **_kw: None)

    with caplog.at_level("WARNING"):
        summary = cl.consolidate_one("sample", tmp_path)

    assert target.read_text() == "ALLAY\nBABES\nCDS\n"
    assert summary.total_after == 3
    assert "repeated 1 row(s)" in caplog.text
