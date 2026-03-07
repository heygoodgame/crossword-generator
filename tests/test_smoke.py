"""Smoke tests to verify project setup."""

from pathlib import Path


def test_dictionary_exists(dictionary_path: Path) -> None:
    """Verify the Jeff Chen word list is present."""
    assert dictionary_path.exists()
    assert dictionary_path.stat().st_size > 0


def test_models_import() -> None:
    """Verify core models can be imported."""
    from crossword_generator.models import PuzzleEnvelope, PuzzleType

    envelope = PuzzleEnvelope()
    assert envelope.puzzle_type == PuzzleType.MINI
    assert envelope.grid_size == 5


def test_cli_exists() -> None:
    """Verify CLI entrypoint can be imported."""
    from crossword_generator.cli import main

    assert main is not None
