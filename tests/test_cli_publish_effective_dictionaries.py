"""Tests for publish-effective-dictionaries CLI wiring."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from crossword_generator import cli as cli_module
from crossword_generator import consolidate_list as cl
from crossword_generator.cli import main


def test_publish_effective_dictionaries_dry_run_does_not_write_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dictionaries = tmp_path / "dictionaries"
    dictionaries.mkdir()
    easy_source = dictionaries / "easy.txt"
    easy_extra = dictionaries / "extra.txt"
    exclude_source = dictionaries / "exclude.txt"
    hard_source = dictionaries / "hard.txt"
    sixty_source = dictionaries / "sixty.txt"
    easy_output = dictionaries / "hgg-easy.txt"
    sixty_output = dictionaries / "hgg-60.txt"

    easy_source.write_text("APPLE;55\nALCHEMY;55\nNOVICE;55\n")
    easy_extra.write_text("ZOOMING\n")
    exclude_source.write_text("APPLE;removed\n")
    # Plain hard fill list; scores live in the master sixty source.
    hard_source.write_text("ALCHEMY\nAIRLIFT\nNOVICE\n")
    sixty_source.write_text("ALCHEMY;60\nAIRLIFT;60\nNOVICE;50\n")
    monkeypatch.setattr(cli_module, "find_project_root", lambda: tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "publish-effective-dictionaries",
            "--easy-source", str(easy_source),
            "--easy-extra-source", str(easy_extra),
            "--easy-exclude-source", str(exclude_source),
            "--hard-source", str(hard_source),
            "--sixty-source", str(sixty_source),
            "--easy-output", str(easy_output),
            "--sixty-output", str(sixty_output),
            "--generator-commit", "abc123",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "Snapshot payload:" in result.output
    assert "Dry run: no API call made" in result.output
    assert not easy_output.exists()
    assert not sixty_output.exists()


def test_refresh_dictionaries_consolidates_and_rebuilds_local_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dictionaries = tmp_path / "dictionaries"
    dictionaries.mkdir()
    easy_source = dictionaries / "HGGXW-Easy.txt"
    hard_source = dictionaries / "HGGXW-Hard.txt"
    sixty_source = dictionaries / "XwiJeffChenList.txt"
    (dictionaries / "XwiJeffChenList-NotFamilyFriendly.txt").write_text("")
    (dictionaries / "HggGeneratedSafetyExclude.txt").write_text("")

    easy_source.write_text("SLOAN\nAPPLE\n")
    hard_source.write_text("")
    sixty_source.write_text("SLOAN;50\nAPPLE;50\nFEATURES;60\n")

    def fake_list(**_kw):
        return [
            {"slug": "hgg-easy", "file_path": "dictionaries/HGGXW-Easy.txt"},
            {"slug": "hgg-hard", "file_path": "dictionaries/HGGXW-Hard.txt"},
        ]

    def fake_consolidate(slug, project_root, **_kw):
        if slug == "hgg-easy":
            easy_source.write_text("APPLE\n")
            return cl.ListSummary(
                slug=slug,
                file_path="dictionaries/HGGXW-Easy.txt",
                added=0,
                removed=1,
                total_after=1,
                wrote=True,
            )
        hard_source.write_text("SLOAN\n")
        return cl.ListSummary(
            slug=slug,
            file_path="dictionaries/HGGXW-Hard.txt",
            added=1,
            removed=0,
            total_after=1,
            wrote=True,
        )

    monkeypatch.setattr(cli_module, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(cl, "list_registered_lists", fake_list)
    monkeypatch.setattr(cl, "consolidate_one", fake_consolidate)

    result = CliRunner().invoke(main, ["refresh-dictionaries"])

    assert result.exit_code == 0, result.output
    assert "Refreshing crossword word lists" in result.output
    assert "Rebuilt local effective dictionaries" in result.output
    assert (dictionaries / "hgg-easy.txt").read_text() == "APPLE;50\n"
    assert "SLOAN;50\n" in (dictionaries / "hgg-hard.txt").read_text()
