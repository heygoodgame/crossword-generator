"""I/O utilities for standalone theme files."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from crossword_generator.models import ThemeConcept, ThemeFile

logger = logging.getLogger(__name__)


def save_theme(
    theme: ThemeConcept,
    grid_size: int,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save a theme to a JSON file.

    Args:
        theme: The theme concept to save.
        grid_size: Grid size the theme was generated for.
        model_name: LLM model name used for generation.
        output_dir: Directory to write the file into.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(theme.topic)
    short_id = uuid.uuid4().hex[:8]
    filename = f"{slug}_{short_id}.json"

    theme_file = ThemeFile(
        theme=theme,
        grid_size=grid_size,
        generated_at=datetime.now(tz=UTC).isoformat(),
        generator_model=model_name,
    )

    path = output_dir / filename
    path.write_text(theme_file.model_dump_json(indent=2))
    logger.info("Saved theme to %s", path)
    return path


def load_theme(path: Path) -> ThemeFile:
    """Load and validate a theme file.

    Args:
        path: Path to the JSON theme file.

    Returns:
        A validated ThemeFile instance.
    """
    return ThemeFile.model_validate_json(path.read_text())


def list_themes(directory: Path) -> list[ThemeFile]:
    """List all theme files in a directory.

    Args:
        directory: Directory to scan for .json theme files.

    Returns:
        List of loaded ThemeFile instances (invalid files are skipped).
    """
    if not directory.exists():
        return []

    themes: list[ThemeFile] = []
    for path in sorted(directory.glob("*.json")):
        try:
            themes.append(load_theme(path))
        except Exception:
            logger.debug("Skipping invalid theme file: %s", path)
    return themes


def load_topic_set(directory: Path) -> set[str]:
    """Load the set of previously-generated topic strings.

    Args:
        directory: Directory containing theme JSON files.

    Returns:
        Set of topic strings from all valid theme files.
    """
    return {tf.theme.topic for tf in list_themes(directory)}


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:40] if slug else "theme"
