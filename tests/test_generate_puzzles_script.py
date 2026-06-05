"""Tests for the week-based batch generation helper."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run_helper(project_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(project_root / "generate-puzzles.sh"), *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )


def test_generate_puzzles_script_two_week_default(project_root: Path) -> None:
    result = _run_helper(project_root, "2", "--dry-run", "--seed", "123456")

    assert result.returncode == 0
    assert "Bucket counts: 5=10,7=4,9=14" in result.stdout
    assert "Buckets: easy/5,easy/7,easy/9" in result.stdout
    assert "--bucket-counts 5=10\\,7=4\\,9=14" in result.stdout
    assert "--seed-start 123456" in result.stdout
    assert "--llm claude" in result.stdout


def test_generate_puzzles_script_custom_batch_and_hard(project_root: Path) -> None:
    result = _run_helper(
        project_root,
        "3",
        "--dry-run",
        "--difficulty",
        "hard",
        "--batch-id",
        "manual-batch",
        "--output-root",
        "output/custom",
        "--seed",
        "42",
        "--",
        "--verbose",
    )

    assert result.returncode == 0
    assert "Bucket counts: 5=15,7=6,9=21" in result.stdout
    assert "Buckets: hard/5,hard/7,hard/9" in result.stdout
    assert "Batch id: manual-batch" in result.stdout
    assert "Output root: output/custom" in result.stdout
    assert "--verbose" in result.stdout


def test_generate_puzzles_script_rejects_bad_weeks(project_root: Path) -> None:
    result = _run_helper(project_root, "0", "--dry-run")

    assert result.returncode == 2
    assert "WEEKS must be a positive integer" in result.stderr
