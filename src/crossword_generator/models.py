"""Core data models for the crossword generation pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PuzzleType(StrEnum):
    MINI = "mini"
    MIDI = "midi"


class GridCell(BaseModel):
    """A single cell in the crossword grid."""

    row: int
    col: int
    letter: str | None = None
    is_black: bool = False
    cell_number: int | None = None


class ClueEntry(BaseModel):
    """A clue for a single crossword entry."""

    number: int
    direction: str  # "across" or "down"
    answer: str
    clue: str = ""
    quality_score: float | None = None


class ThemeConcept(BaseModel):
    """A theme concept for a themed (midi) puzzle."""

    topic: str = ""
    wordplay_type: str = ""
    seed_entries: list[str] = Field(default_factory=list)
    candidate_entries: list[str] = Field(default_factory=list)
    revealer: str = ""
    revealer_clue: str = ""


class WordGrade(BaseModel):
    """Quality grade for a single word in the grid."""

    word: str
    length: int
    direction: str
    number: int
    dictionary_score: int | None = None
    penalties: dict[str, float] = Field(default_factory=dict)
    adjusted_score: float = 0.0


class FillGradeReport(BaseModel):
    """Aggregate fill quality report for a completed grid."""

    overall_score: float  # 0-100
    word_count: int
    passing: bool
    word_grades: list[WordGrade] = Field(default_factory=list)
    penalties_applied: dict[str, float] = Field(default_factory=dict)
    summary: str = ""


class ClueGrade(BaseModel):
    """Quality grade for a single clue from LLM evaluation."""

    number: int
    direction: str
    answer: str
    score: float  # 0-100
    feedback: str = ""


class ClueGradeReport(BaseModel):
    """Aggregate clue quality report from LLM evaluation."""

    overall_score: float  # 0-100
    clue_count: int
    passing: bool
    clue_grades: list[ClueGrade] = Field(default_factory=list)
    summary: str = ""


class FillResult(BaseModel):
    """Result of grid filling."""

    grid: list[list[str]] = Field(default_factory=list)
    quality_score: float | None = None
    filler_used: str = ""
    attempt_number: int = 1
    grade_report: FillGradeReport | None = None


class PuzzleEnvelope(BaseModel):
    """The JSON contract passed between all pipeline steps.

    Each step reads the envelope, adds its output, and writes it back.
    """

    puzzle_type: PuzzleType = PuzzleType.MINI
    grid_size: int = 5

    # Step outputs (populated as the pipeline progresses)
    theme: ThemeConcept | None = None
    fill: FillResult | None = None
    clues: list[ClueEntry] = Field(default_factory=list)
    clue_grade_report: ClueGradeReport | None = None

    # Metadata
    step_history: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
