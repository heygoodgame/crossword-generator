"""Per-step cost and cache-effectiveness report for a batch's LLM logs.

Graduated from the one-off audit of batch ``weekly-1w-20260607``. Summarizes,
per pipeline step, the model used, call count, token usage, estimated cost, and
the realized prompt-cache hit rate — so caching regressions and cost surprises
are visible after every batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from crossword_generator.llm.log_browser import LLMLogRecord, load_llm_logs


def _usage_int(usage: dict[str, int | float], key: str) -> int:
    value = usage.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass
class StepStats:
    """Aggregated stats for one (step, model) pair."""

    step: str
    model: str
    calls: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    duration_seconds: float = 0.0

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache (0.0-1.0)."""
        total = self.total_prompt_tokens
        return self.cache_read_tokens / total if total else 0.0


@dataclass
class BatchReport:
    """Per-step report plus batch totals."""

    steps: list[StepStats] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_calls(self) -> int:
        return sum(s.calls for s in self.steps)


def build_report(records: list[LLMLogRecord]) -> BatchReport:
    """Aggregate LLM log records into a per-(step, model) report."""
    by_key: dict[tuple[str, str], StepStats] = {}
    for record in records:
        if record.has_error:
            continue
        step = record.step or "?"
        model = record.model or "?"
        key = (step, model)
        stats = by_key.get(key)
        if stats is None:
            stats = StepStats(step=step, model=model)
            by_key[key] = stats

        stats.calls += 1
        stats.cost_usd += record.estimated_cost_usd or 0.0
        usage = record.usage or {}
        stats.input_tokens += _usage_int(usage, "input_tokens")
        stats.output_tokens += _usage_int(usage, "output_tokens")
        stats.cache_creation_tokens += _usage_int(usage, "cache_creation_input_tokens")
        stats.cache_read_tokens += _usage_int(usage, "cache_read_input_tokens")
        duration = record.duration_seconds
        if isinstance(duration, (int, float)):
            stats.duration_seconds += float(duration)

    steps = sorted(by_key.values(), key=lambda s: (s.step, s.model))
    return BatchReport(steps=steps)


def report_for_path(target: Path | None) -> BatchReport:
    """Load logs from a file or directory and build the report."""
    records, _problems = load_llm_logs(target)
    return build_report(records)


def format_report(report: BatchReport) -> str:
    """Render a human-readable table."""
    lines: list[str] = []
    header = (
        f"{'STEP':<22}{'MODEL':<26}{'N':>4}{'COST$':>9}"
        f"{'IN':>9}{'OUT':>9}{'C_WR':>8}{'C_RD':>8}{'HIT%':>7}{'SEC':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in report.steps:
        lines.append(
            f"{s.step:<22}{s.model:<26}{s.calls:>4}{s.cost_usd:>9.4f}"
            f"{s.input_tokens:>9}{s.output_tokens:>9}"
            f"{s.cache_creation_tokens:>8}{s.cache_read_tokens:>8}"
            f"{s.cache_hit_rate * 100:>6.1f}%{s.duration_seconds:>8.1f}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':<48}{report.total_calls:>4}{report.total_cost_usd:>9.4f}")
    return "\n".join(lines)
