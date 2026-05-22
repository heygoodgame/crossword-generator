"""Tests for publish-effective-dictionaries CLI wiring."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from crossword_generator import cli as cli_module
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
    easy_output = dictionaries / "hgg-easy.txt"
    sixty_output = dictionaries / "hgg-60.txt"

    easy_source.write_text("APPLE;55\nALCHEMY;55\nNOVICE;55\n")
    easy_extra.write_text("ZOOMING\n")
    exclude_source.write_text("APPLE;removed\n")
    hard_source.write_text("ALCHEMY;60\nAIRLIFT;60\nNOVICE;50\n")
    monkeypatch.setattr(cli_module, "find_project_root", lambda: tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "publish-effective-dictionaries",
            "--easy-source", str(easy_source),
            "--easy-extra-source", str(easy_extra),
            "--easy-exclude-source", str(exclude_source),
            "--hard-source", str(hard_source),
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
