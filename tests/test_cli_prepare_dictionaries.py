"""Tests for dictionary preparation CLI wiring."""

from pathlib import Path

from click.testing import CliRunner

from crossword_generator.cli import main


def test_prepare_dictionaries_writes_easy_and_hard(tmp_path: Path) -> None:
    """Without overrides, hard excludes mirror easy excludes from .txt sources."""
    easy_source = tmp_path / "easy.txt"
    hard_source = tmp_path / "hard.txt"
    exclude_source = tmp_path / "exclude.txt"
    easy_output = tmp_path / "easy-out.txt"
    hard_output = tmp_path / "hard-out.txt"
    hard_7_output = tmp_path / "hard-7-out.txt"
    easy_source.write_text("APPLE;55\nULSTER;55\nTYPHOID;55\n")
    hard_source.write_text("PUZZLE;60\nSEVENOK;60\n")
    exclude_source.write_text("ULSTER\nTYPHOID\n")

    result = CliRunner().invoke(
        main,
        [
            "prepare-dictionaries",
            "--easy-source",
            str(easy_source),
            "--easy-exclude-source",
            str(exclude_source),
            "--hard-source",
            str(hard_source),
            "--easy-output",
            str(easy_output),
            "--hard-output",
            str(hard_output),
            "--hard-7-output",
            str(hard_7_output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert easy_output.read_text().splitlines() == ["APPLE;55"]
    assert "ULSTER;55" not in hard_output.read_text()
    assert "TYPHOID;55" not in hard_output.read_text()
