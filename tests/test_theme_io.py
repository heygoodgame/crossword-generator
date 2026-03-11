"""Tests for theme file I/O utilities."""

from __future__ import annotations

from pathlib import Path

from crossword_generator.models import ThemeConcept
from crossword_generator.theme_io import (
    list_themes,
    load_theme,
    load_topic_set,
    save_theme,
)


def _make_theme(topic: str = "Things that fly") -> ThemeConcept:
    return ThemeConcept(
        topic=topic,
        wordplay_type="literal",
        seed_entries=["EAGLE", "KITE", "HAWK"],
        revealer="SOAR",
        revealer_clue="Take flight",
    )


class TestSaveAndLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        theme = _make_theme()
        path = save_theme(
            theme, grid_size=9, model_name="test-llm", output_dir=tmp_path
        )

        assert path.exists()
        assert path.suffix == ".json"

        loaded = load_theme(path)
        assert loaded.theme.topic == "Things that fly"
        assert loaded.theme.seed_entries == ["EAGLE", "KITE", "HAWK"]
        assert loaded.theme.revealer == "SOAR"
        assert loaded.grid_size == 9
        assert loaded.generator_model == "test-llm"
        assert loaded.generated_at != ""

    def test_filename_contains_slug(self, tmp_path: Path) -> None:
        theme = _make_theme("Things that are red")
        path = save_theme(theme, grid_size=9, model_name="m", output_dir=tmp_path)
        assert "things-that-are-red" in path.name

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "dir"
        theme = _make_theme()
        path = save_theme(theme, grid_size=9, model_name="m", output_dir=nested)
        assert nested.exists()
        assert path.exists()


class TestListThemes:
    def test_lists_saved_themes(self, tmp_path: Path) -> None:
        save_theme(_make_theme("Topic A"), 9, "m", tmp_path)
        save_theme(_make_theme("Topic B"), 9, "m", tmp_path)

        themes = list_themes(tmp_path)
        assert len(themes) == 2
        topics = {t.theme.topic for t in themes}
        assert topics == {"Topic A", "Topic B"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert list_themes(tmp_path) == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert list_themes(tmp_path / "nope") == []

    def test_skips_invalid_files(self, tmp_path: Path) -> None:
        save_theme(_make_theme(), 9, "m", tmp_path)
        # Write an invalid JSON file
        (tmp_path / "bad.json").write_text("not valid json")

        themes = list_themes(tmp_path)
        assert len(themes) == 1


class TestLoadTopicSet:
    def test_returns_topic_strings(self, tmp_path: Path) -> None:
        save_theme(_make_theme("Fruits"), 9, "m", tmp_path)
        save_theme(_make_theme("Animals"), 9, "m", tmp_path)

        topics = load_topic_set(tmp_path)
        assert topics == {"Fruits", "Animals"}

    def test_empty_when_no_files(self, tmp_path: Path) -> None:
        assert load_topic_set(tmp_path) == set()
