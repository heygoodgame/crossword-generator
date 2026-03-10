#!/usr/bin/env python3
"""Benchmark CSP filler quality across grid sizes and seeds.

Runs N seeds across configurable grid sizes, grades each fill,
optionally retries, and reports score distributions.

Usage:
    uv run python scripts/benchmark_fill_quality.py
    uv run python scripts/benchmark_fill_quality.py --seeds 100 --sizes 5 7
    uv run python scripts/benchmark_fill_quality.py --seeds 500 --sizes 5 7 9 \
        --max-retries 10 --retries-by-size 7:5,9:3 --min-score 60 --workers 12
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

from crossword_generator.config import (
    CSPFillerConfig,
    find_project_root,
    load_config,
)
from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FillError
from crossword_generator.fillers.csp import CSPFiller
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.models import PuzzleType

logger = logging.getLogger(__name__)

SIZE_TO_PUZZLE_TYPE: dict[int, PuzzleType] = {
    5: PuzzleType.MINI,
    7: PuzzleType.MINI,
    9: PuzzleType.MIDI,
    10: PuzzleType.MIDI,
    11: PuzzleType.MIDI,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AttemptResult:
    """A single fill+grade attempt."""

    seed: int
    grid_size: int
    attempt: int
    success: bool
    score: float | None
    passing: bool
    word_count: int
    time_seconds: float
    error: str | None


@dataclass
class SeedResult:
    """Aggregated result for one seed across all retry attempts."""

    seed: int
    grid_size: int
    max_retries: int
    attempts: list[AttemptResult]
    best_score: float | None
    best_attempt: int | None
    final_passing: bool


@dataclass
class SizeSummary:
    """Statistics for one grid size across all seeds."""

    grid_size: int
    max_retries: int
    total_seeds: int
    total_attempts: int
    fill_failures: int
    scores: list[float]
    all_attempt_scores: list[float]
    pass_rate: float
    mean_score: float
    median_score: float
    stdev_score: float
    min_score: float
    max_score: float
    p10: float
    p25: float
    p75: float
    p90: float
    mean_time: float
    total_time: float


# ---------------------------------------------------------------------------
# Worker state (initialized per process)
# ---------------------------------------------------------------------------

_worker_filler: CSPFiller | None = None
_worker_grader: FillGrader | None = None


def _init_worker(
    csp_config_dict: dict,
    dictionary_path: str,
    min_word_score: int,
    min_2letter_score: int,
    min_passing_score: int,
) -> None:
    """Initialize per-worker CSPFiller and FillGrader."""
    global _worker_filler, _worker_grader  # noqa: PLW0603
    csp_config = CSPFillerConfig.model_validate(csp_config_dict)
    dictionary = Dictionary.load(
        Path(dictionary_path),
        min_word_score=min_word_score,
        min_2letter_score=min_2letter_score,
    )
    _worker_filler = CSPFiller(csp_config, dictionary)
    _worker_grader = FillGrader(dictionary, min_passing_score=min_passing_score)


# ---------------------------------------------------------------------------
# Core work function
# ---------------------------------------------------------------------------


def _run_seed(
    seed: int,
    grid_size: int,
    puzzle_type_value: str,
    max_retries: int,
    min_passing_score: int,
) -> SeedResult:
    """Run one seed with retries. Uses module-level worker state."""
    assert _worker_filler is not None
    assert _worker_grader is not None

    puzzle_type = PuzzleType(puzzle_type_value)
    spec = get_grid_spec(puzzle_type, grid_size, seed=seed)
    attempts: list[AttemptResult] = []
    best_score: float | None = None
    best_attempt: int | None = None

    for attempt_num in range(1, max_retries + 1):
        fill_seed = seed if attempt_num == 1 else seed + attempt_num * 10_000
        start = time.monotonic()

        try:
            filled = _worker_filler.fill(spec, seed=fill_seed)
            elapsed = time.monotonic() - start
            report = _worker_grader.grade(filled.grid)

            result = AttemptResult(
                seed=seed,
                grid_size=grid_size,
                attempt=attempt_num,
                success=True,
                score=report.overall_score,
                passing=report.passing,
                word_count=report.word_count,
                time_seconds=elapsed,
                error=None,
            )
        except FillError as e:
            elapsed = time.monotonic() - start
            result = AttemptResult(
                seed=seed,
                grid_size=grid_size,
                attempt=attempt_num,
                success=False,
                score=None,
                passing=False,
                word_count=0,
                time_seconds=elapsed,
                error=str(e),
            )

        attempts.append(result)

        if result.score is not None:
            if best_score is None or result.score > best_score:
                best_score = result.score
                best_attempt = attempt_num

        if result.passing:
            break

    return SeedResult(
        seed=seed,
        grid_size=grid_size,
        max_retries=max_retries,
        attempts=attempts,
        best_score=best_score,
        best_attempt=best_attempt,
        final_passing=any(a.passing for a in attempts),
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _parse_size_map(s: str) -> dict[int, int]:
    """Parse 'size:value,...' format, e.g. '5:10,7:5,9:3'."""
    result: dict[int, int] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        size_str, val_str = pair.split(":")
        result[int(size_str)] = int(val_str)
    return result


def _build_csp_config(config, args: argparse.Namespace) -> CSPFillerConfig:
    """Build CSPFillerConfig with CLI overrides applied."""
    updates: dict = {}

    if args.timeout is not None:
        updates["timeout"] = args.timeout

    if args.timeout_by_size is not None:
        updates["timeout_by_size"] = _parse_size_map(args.timeout_by_size)

    if args.quality_tiers is not None:
        updates["quality_tiers"] = args.quality_tiers

    if args.dictionary is not None:
        updates["dictionary_path"] = args.dictionary

    if updates:
        return config.fill.csp.model_copy(update=updates)
    return config.fill.csp


def _get_retries_for_size(
    size: int,
    retries_by_size: dict[int, int] | None,
    default: int,
) -> int:
    """Return max retries for a given grid size."""
    if retries_by_size and size in retries_by_size:
        return retries_by_size[size]
    return default


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def _print_progress(
    completed: int,
    total: int,
    result: SeedResult,
    verbose: bool,
) -> None:
    """Print progress to stderr."""
    pct = completed * 100 / total
    best = f"{result.best_score:.1f}" if result.best_score is not None else "FAIL"

    if verbose:
        attempts_str = "/".join(
            f"{a.score:.1f}" if a.score is not None else "X"
            for a in result.attempts
        )
        sys.stderr.write(
            f"[{completed:>5}/{total}] {pct:5.1f}% | "
            f"{result.grid_size}x{result.grid_size} seed={result.seed:>5} "
            f"best={best} attempts=[{attempts_str}]\n"
        )
    else:
        sys.stderr.write(f"\r[{completed:>5}/{total}] {pct:5.1f}% complete")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    args: argparse.Namespace,
) -> dict[int, list[SeedResult]]:
    """Execute the benchmark, return results grouped by grid size."""
    config = load_config()
    csp_config = _build_csp_config(config, args)
    project_root = find_project_root()
    dict_path = args.dictionary or str(project_root / config.dictionary.path)

    retries_by_size = (
        _parse_size_map(args.retries_by_size) if args.retries_by_size else None
    )

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Build work items: (seed, grid_size, puzzle_type_value, max_retries)
    work_items: list[tuple[int, int, str, int, int]] = []
    for size in args.sizes:
        puzzle_type = SIZE_TO_PUZZLE_TYPE[size]
        retries = _get_retries_for_size(size, retries_by_size, args.max_retries)
        for seed in seeds:
            work_items.append(
                (seed, size, puzzle_type.value, retries, args.min_score)
            )

    total = len(work_items)
    completed = 0
    results_by_size: dict[int, list[SeedResult]] = {}

    init_args = (
        csp_config.model_dump(),
        dict_path,
        config.dictionary.min_word_score,
        config.dictionary.min_2letter_score,
        args.min_score,
    )

    if args.workers <= 1:
        _init_worker(*init_args)
        for seed, size, pt_val, retries, min_score in work_items:
            result = _run_seed(seed, size, pt_val, retries, min_score)
            results_by_size.setdefault(size, []).append(result)
            completed += 1
            _print_progress(completed, total, result, args.verbose)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=init_args,
        ) as executor:
            future_to_item = {
                executor.submit(
                    _run_seed, seed, size, pt_val, retries, min_score
                ): (seed, size)
                for seed, size, pt_val, retries, min_score in work_items
            }
            for future in as_completed(future_to_item):
                result = future.result()
                results_by_size.setdefault(result.grid_size, []).append(result)
                completed += 1
                _print_progress(completed, total, result, args.verbose)

    # Clear progress line
    if not args.verbose:
        sys.stderr.write("\r" + " " * 60 + "\r")
        sys.stderr.flush()

    return results_by_size


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute percentile from a sorted list."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (n - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def compute_summary(
    grid_size: int,
    seed_results: list[SeedResult],
) -> SizeSummary:
    """Compute aggregate statistics for one grid size."""
    best_scores = [
        r.best_score for r in seed_results if r.best_score is not None
    ]
    all_attempt_scores = [
        a.score
        for r in seed_results
        for a in r.attempts
        if a.score is not None
    ]
    fill_failures = sum(1 for r in seed_results if r.best_score is None)
    total_attempts = sum(len(r.attempts) for r in seed_results)
    pass_count = sum(1 for r in seed_results if r.final_passing)
    all_times = [a.time_seconds for r in seed_results for a in r.attempts]
    max_retries = seed_results[0].max_retries if seed_results else 0

    sorted_scores = sorted(best_scores)

    return SizeSummary(
        grid_size=grid_size,
        max_retries=max_retries,
        total_seeds=len(seed_results),
        total_attempts=total_attempts,
        fill_failures=fill_failures,
        scores=best_scores,
        all_attempt_scores=all_attempt_scores,
        pass_rate=pass_count / len(seed_results) if seed_results else 0.0,
        mean_score=mean(best_scores) if best_scores else 0.0,
        median_score=median(best_scores) if best_scores else 0.0,
        stdev_score=stdev(best_scores) if len(best_scores) >= 2 else 0.0,
        min_score=min(best_scores) if best_scores else 0.0,
        max_score=max(best_scores) if best_scores else 0.0,
        p10=_percentile(sorted_scores, 10),
        p25=_percentile(sorted_scores, 25),
        p75=_percentile(sorted_scores, 75),
        p90=_percentile(sorted_scores, 90),
        mean_time=mean(all_times) if all_times else 0.0,
        total_time=sum(all_times),
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_histogram(
    scores: list[float],
    *,
    bin_width: int = 2,
    bar_max_width: int = 50,
) -> str:
    """Render a text histogram of score distribution."""
    if not scores:
        return "  (no data)"

    bins: dict[int, int] = {}
    for s in scores:
        b = int(s // bin_width) * bin_width
        b = min(b, 100 - bin_width)
        bins[b] = bins.get(b, 0) + 1

    max_count = max(bins.values()) if bins else 1

    # Find the range of non-empty bins (with 1 bin padding)
    non_empty = [b for b, c in bins.items() if c > 0]
    lo = max(0, min(non_empty) - bin_width)
    hi = min(100, max(non_empty) + 2 * bin_width)

    lines = ["  Score Distribution:"]
    for b in range(lo, hi, bin_width):
        count = bins.get(b, 0)
        bar_len = int(count / max_count * bar_max_width) if max_count > 0 else 0
        bar = "#" * bar_len
        label = f"  {b:>3}-{b + bin_width:<3}"
        lines.append(f"{label} |{bar} {count}")

    return "\n".join(lines)


def format_report(
    summaries: list[SizeSummary],
    args: argparse.Namespace,
) -> str:
    """Format the full benchmark report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("CSP Filler Quality Benchmark")
    lines.append("=" * 70)
    lines.append(
        f"Seeds: {args.seeds} "
        f"(range {args.seed_start}..{args.seed_start + args.seeds - 1})"
    )
    lines.append(f"Grid sizes: {args.sizes}")
    lines.append(f"Default max retries: {args.max_retries}")
    if args.retries_by_size:
        lines.append(f"Per-size retries: {args.retries_by_size}")
    lines.append(f"Min passing score: {args.min_score}")
    lines.append(f"Workers: {args.workers}")
    lines.append("")

    for summary in summaries:
        sz = summary.grid_size
        lines.append(f"--- {sz}x{sz} (max_retries={summary.max_retries}) ---")
        lines.append(f"  Seeds tested:      {summary.total_seeds}")
        lines.append(f"  Total attempts:    {summary.total_attempts}")
        lines.append(f"  Fill failures:     {summary.fill_failures}")
        lines.append(f"  Pass rate:         {summary.pass_rate:.1%}")
        lines.append(f"  Mean score:        {summary.mean_score:.2f}")
        lines.append(f"  Median score:      {summary.median_score:.2f}")
        lines.append(f"  Std deviation:     {summary.stdev_score:.2f}")
        lines.append(f"  Min score:         {summary.min_score:.2f}")
        lines.append(f"  Max score:         {summary.max_score:.2f}")
        lines.append(f"  P10:               {summary.p10:.2f}")
        lines.append(f"  P25:               {summary.p25:.2f}")
        lines.append(f"  P75:               {summary.p75:.2f}")
        lines.append(f"  P90:               {summary.p90:.2f}")
        lines.append(f"  Mean time/attempt: {summary.mean_time:.2f}s")
        lines.append(f"  Total time:        {summary.total_time:.1f}s")
        lines.append("")
        lines.append(_format_histogram(summary.scores))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_csv(
    results_by_size: dict[int, list[SeedResult]],
    output_path: Path,
) -> None:
    """Save all attempt-level results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "grid_size",
            "seed",
            "attempt",
            "max_retries",
            "success",
            "score",
            "passing",
            "word_count",
            "time_seconds",
            "error",
        ])
        for size in sorted(results_by_size):
            for seed_result in sorted(
                results_by_size[size], key=lambda r: r.seed
            ):
                for attempt in seed_result.attempts:
                    writer.writerow([
                        attempt.grid_size,
                        attempt.seed,
                        attempt.attempt,
                        seed_result.max_retries,
                        attempt.success,
                        f"{attempt.score:.2f}" if attempt.score is not None else "",
                        attempt.passing,
                        attempt.word_count,
                        f"{attempt.time_seconds:.3f}",
                        attempt.error or "",
                    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CSP filler quality across grid sizes and seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=500,
        help="Number of seeds to test per grid size (default: 500)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 7, 9],
        help="Grid sizes to benchmark (default: 5 7 9)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Max fill attempts per seed, global default (default: 10)",
    )
    parser.add_argument(
        "--retries-by-size",
        type=str,
        default=None,
        help="Per-size retry overrides as 'size:retries,...' e.g. '7:5,9:3'",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=60,
        help="Min passing score for fill grading (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="CSP timeout in seconds (overrides config for all sizes)",
    )
    parser.add_argument(
        "--timeout-by-size",
        type=str,
        default=None,
        help="Per-size timeouts as 'size:seconds,...' e.g. '5:30,7:120,9:300'",
    )
    parser.add_argument(
        "--quality-tiers",
        type=int,
        nargs="+",
        default=None,
        help="CSP quality tiers (default: from config)",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default=None,
        help="Dictionary path (default: from config)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="CSV output path (default: auto-timestamped in output/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed value (default: 0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-seed results during execution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate sizes
    for size in args.sizes:
        if size not in SIZE_TO_PUZZLE_TYPE:
            valid = ", ".join(str(s) for s in sorted(SIZE_TO_PUZZLE_TYPE))
            print(
                f"Error: unsupported grid size {size}. Valid: {valid}",
                file=sys.stderr,
            )
            sys.exit(1)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Print config summary to stderr
    sys.stderr.write(
        f"Benchmark: {args.seeds} seeds x {args.sizes} sizes, "
        f"max_retries={args.max_retries}"
    )
    if args.retries_by_size:
        sys.stderr.write(f" (overrides: {args.retries_by_size})")
    sys.stderr.write(f", min_score={args.min_score}, workers={args.workers}\n")

    start_time = time.monotonic()
    results_by_size = run_benchmark(args)
    wall_time = time.monotonic() - start_time

    # Compute and print summaries
    summaries = [
        compute_summary(size, results_by_size.get(size, []))
        for size in sorted(args.sizes)
        if results_by_size.get(size)
    ]

    report = format_report(summaries, args)
    report += f"\nTotal wall time: {wall_time:.1f}s\n"
    print(report)

    # Save CSV
    if args.output:
        csv_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path("output") / f"benchmark_fill_quality_{ts}.csv"

    save_csv(results_by_size, csv_path)
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
