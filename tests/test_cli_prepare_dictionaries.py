"""Tests for dictionary preparation CLI wiring."""

from pathlib import Path

from click.testing import CliRunner

from crossword_generator import data_store
from crossword_generator.cli import main
from crossword_generator.data_store import WordListOverrides


def test_prepare_dictionaries_applies_word_list_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    easy_source = tmp_path / "easy.txt"
    hard_source = tmp_path / "hard.txt"
    easy_output = tmp_path / "easy-out.txt"
    hard_output = tmp_path / "hard-out.txt"
    hard_7_output = tmp_path / "hard-7-out.txt"
    easy_source.write_text("APPLE;55\nULSTER;55\nTYPHOID;55\n")
    hard_source.write_text("PUZZLE;60\nSEVENOK;60\n")

    overrides = {
        "easy": WordListOverrides(
            list_scope="easy",
            include=frozenset({"NEWEASY"}),
            exclude=frozenset({"ULSTER"}),
        ),
        "hard": WordListOverrides(
            list_scope="hard",
            include=frozenset({"HARDLY"}),
            exclude=frozenset(),
        ),
        "all": WordListOverrides(
            list_scope="all",
            include=frozenset(),
            exclude=frozenset({"TYPHOID"}),
        ),
    }
    calls: list[str] = []

    def fake_fetch(list_scope: str):
        calls.append(list_scope)
        return overrides[list_scope]

    monkeypatch.setattr(data_store, "fetch_word_list_overrides", fake_fetch)

    result = CliRunner().invoke(
        main,
        [
            "prepare-dictionaries",
            "--easy-source",
            str(easy_source),
            "--hard-source",
            str(hard_source),
            "--easy-output",
            str(easy_output),
            "--hard-output",
            str(hard_output),
            "--hard-7-output",
            str(hard_7_output),
            "--apply-overrides",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["easy", "hard", "all"]
    assert easy_output.read_text().splitlines() == [
        "APPLE;55",
        "NEWEASY;55",
    ]
    assert "HARDLY;55" in hard_output.read_text().splitlines()
    assert "HARDLY;55" in hard_7_output.read_text().splitlines()
    assert "ULSTER;55" not in hard_output.read_text()
    assert "TYPHOID;55" not in hard_output.read_text()
