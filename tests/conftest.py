"""Shared test fixtures for the crossword generator test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def dictionary_path(project_root: Path) -> Path:
    """Return the path to the HGG curated crossword word list."""
    return project_root / "dictionaries" / "HggCuratedCrosswordList.txt"
