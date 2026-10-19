"""Shared test fixtures for the crossword generator test suite."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_content_shadow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test off the live Jev study files and the TypeSafe API."""
    monkeypatch.setenv("CROSSWORD_SHADOW_STUDY_DIR", str(tmp_path / "shadow-study"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def dictionary_path(project_root: Path) -> Path:
    """Return the path to the HGG curated crossword word list."""
    return project_root / "dictionaries" / "HggScoredCrosswordList.txt"
