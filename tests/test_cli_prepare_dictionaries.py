"""Tests for dictionary preparation CLI wiring."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from crossword_generator.cli import main


def _invoke(
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> tuple[Path, Path, Path, "CliRunner.invoke"]:
    """Lay out a minimal source set under tmp_path and run prepare-dictionaries."""
    easy_source = tmp_path / "easy.txt"
    hard_source = tmp_path / "hard.txt"
    exclude_source = tmp_path / "exclude.txt"
    easy_output = tmp_path / "easy-out.txt"
    hard_output = tmp_path / "hard-out.txt"
    hard_7_output = tmp_path / "hard-7-out.txt"
    easy_source.write_text("APPLE;55\nULSTER;55\nTYPHOID;55\nNEVERHARD;55\n")
    hard_source.write_text("PUZZLE;60\nSEVENOK;60\nNEVERHARD;60\n")
    exclude_source.write_text("ULSTER\nTYPHOID\n")

    args = [
        "prepare-dictionaries",
        "--easy-source", str(easy_source),
        "--easy-exclude-source", str(exclude_source),
        "--hard-source", str(hard_source),
        "--easy-output", str(easy_output),
        "--hard-output", str(hard_output),
        "--hard-7-output", str(hard_7_output),
        *(extra_args or []),
    ]
    result = CliRunner().invoke(main, args)
    return easy_output, hard_output, hard_7_output, result


def test_prepare_dictionaries_writes_easy_and_hard(tmp_path: Path) -> None:
    """Without overrides, hard excludes mirror easy excludes from .txt sources."""
    easy_output, hard_output, _, result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert "APPLE;55" in easy_output.read_text().splitlines()
    assert "ULSTER;55" not in hard_output.read_text()
    assert "TYPHOID;55" not in hard_output.read_text()


def test_prepare_dictionaries_auto_discovers_thumbs_down_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HggThumbsDownEasy/Hard.txt under dictionaries/ are unioned in automatically."""
    from crossword_generator import cli as cli_module

    dictionaries = tmp_path / "dictionaries"
    dictionaries.mkdir()
    (dictionaries / "HggThumbsDownEasy.txt").write_text("APPLE;easy thumbs-down\n")
    (dictionaries / "HggThumbsDownHard.txt").write_text("NEVERHARD;hard thumbs-down\n")
    monkeypatch.setattr(cli_module, "find_project_root", lambda: tmp_path)

    easy_output, hard_output, _, result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert "Thumbs-down lists auto-discovered" in result.output
    # APPLE is in the easy thumbs-down list → must not appear in easy output
    assert "APPLE;55" not in easy_output.read_text()
    # NEVERHARD is in the hard thumbs-down list → must not appear in hard output,
    # but should still appear in easy output (hard exclusion ≠ easy exclusion).
    assert "NEVERHARD" in easy_output.read_text()
    assert "NEVERHARD" not in hard_output.read_text()


def test_prepare_dictionaries_no_thumbs_down_files_means_no_auto_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without HggThumbsDown*.txt on disk, the auto-discover line doesn't print."""
    from crossword_generator import cli as cli_module

    (tmp_path / "dictionaries").mkdir()
    monkeypatch.setattr(cli_module, "find_project_root", lambda: tmp_path)

    _, _, _, result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert "Thumbs-down lists auto-discovered" not in result.output
