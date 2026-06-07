"""Tests for the per-step LLM batch report."""

from __future__ import annotations

from pathlib import Path

from crossword_generator.llm.log_browser import LLMLogRecord
from crossword_generator.llm.log_report import build_report, format_report


def _record(
    step: str,
    model: str,
    *,
    cost: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation: int = 0,
    cache_read: int = 0,
    error: dict | None = None,
) -> LLMLogRecord:
    return LLMLogRecord(
        raw={},
        source_path=Path("x.jsonl"),
        line_number=1,
        request_id="r",
        started_at=None,
        finished_at=None,
        duration_seconds=1.0,
        provider="claude",
        model=model,
        context={"step": step},
        request={},
        response={},
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
        cost={"estimated_cost_usd": cost},
        error=error,
    )


def test_aggregates_by_step_and_model() -> None:
    records = [
        _record("clue_generation", "claude-opus-4-8", cost=0.10, output_tokens=100),
        _record("clue_generation", "claude-opus-4-8", cost=0.20, output_tokens=200),
        _record("clue_grading", "claude-sonnet-4-6", cost=0.05),
    ]
    report = build_report(records)
    assert len(report.steps) == 2
    gen = next(s for s in report.steps if s.step == "clue_generation")
    assert gen.calls == 2
    assert abs(gen.cost_usd - 0.30) < 1e-9
    assert gen.output_tokens == 300
    assert abs(report.total_cost_usd - 0.35) < 1e-9
    assert report.total_calls == 3


def test_cache_hit_rate() -> None:
    record = _record(
        "clue_generation",
        "claude-sonnet-4-6",
        cost=0.0,
        input_tokens=600,
        cache_creation=0,
        cache_read=400,
    )
    report = build_report([record])
    gen = report.steps[0]
    # 400 read / (600 + 0 + 400) = 0.4
    assert abs(gen.cache_hit_rate - 0.4) < 1e-9


def test_skips_error_records() -> None:
    records = [
        _record("clue_generation", "claude-opus-4-8", cost=0.10),
        _record(
            "clue_generation",
            "claude-opus-4-8",
            cost=0.99,
            error={"message": "boom"},
        ),
    ]
    report = build_report(records)
    assert report.total_calls == 1
    assert abs(report.total_cost_usd - 0.10) < 1e-9


def test_format_report_renders_rows() -> None:
    report = build_report([_record("clue_grading", "claude-sonnet-4-6", cost=0.05)])
    text = format_report(report)
    assert "clue_grading" in text
    assert "claude-sonnet-4-6" in text
    assert "TOTAL" in text
