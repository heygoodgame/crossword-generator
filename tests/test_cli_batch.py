"""Tests for batch CLI helpers."""

import logging
import threading

from crossword_generator.cli import (
    _batch_bucket_configs,
    _extract_grid_variant,
    _failure_category,
    _parse_batch_count_overrides,
    _summarize_batch_results,
    _ThreadFilter,
)


def _make_record(thread_id: int) -> logging.LogRecord:
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="m",
        args=(),
        exc_info=None,
    )
    record.thread = thread_id
    return record


def test_thread_filter_passes_only_matching_thread() -> None:
    me = threading.get_ident()
    log_filter = _ThreadFilter(me)
    assert log_filter.filter(_make_record(me)) is True
    assert log_filter.filter(_make_record(me + 1)) is False


def test_summarize_batch_results_by_bucket() -> None:
    results: list[dict[str, object]] = [
        {
            "difficulty": "easy",
            "size": 5,
            "success": True,
            "runtime_seconds": 10.0,
            "clue_score": 80.0,
        },
        {
            "difficulty": "easy",
            "size": 5,
            "success": False,
            "runtime_seconds": 20.0,
            "clue_score": None,
        },
        {
            "difficulty": "hard",
            "size": 9,
            "success": True,
            "runtime_seconds": 30.0,
            "clue_score": 70.0,
        },
    ]

    summary = _summarize_batch_results(results)

    assert summary["easy-5x5"] == {
        "total": 2,
        "successes": 1,
        "failures": 1,
        "success_rate": 0.5,
        "average_runtime_seconds": 15.0,
        "average_clue_score": 80.0,
    }
    assert summary["hard-9x9"]["success_rate"] == 1.0


def test_extract_grid_variant_from_log_messages() -> None:
    assert _extract_grid_variant("Trying grid variant 25 (seed=26)") == 25
    assert (
        _extract_grid_variant(
            "Grid variant 10 skipped: slot lengths [8, 9] unsupported"
        )
        == 10
    )
    assert _extract_grid_variant("No variant here") is None


def test_failure_category_for_incompatible_patterns() -> None:
    category = _failure_category(
        {
            "success": False,
            "skipped_incompatible_variants": 3,
            "fill_attempts": 0,
            "error_message": "All grid variants exhausted",
        }
    )

    assert category == "incompatible_grid_patterns"


def test_hard_7x7_batch_uses_dedicated_config(tmp_path) -> None:
    configs = {
        f"{difficulty}/{size}": config_path.name
        for difficulty, size, _, config_path in _batch_bucket_configs(tmp_path)
    }

    assert configs["hard/7"] == "config.hard7.yaml"
    assert configs["hard/5"] == "config.hard5.yaml"
    assert configs["hard/9"] == "config.hard9.yaml"
    assert configs["easy/9"] == "config.easy9.yaml"


def test_parse_batch_count_overrides_applies_size_ratio(tmp_path) -> None:
    selected_buckets = _batch_bucket_configs(tmp_path)

    counts = _parse_batch_count_overrides("5=5,7=2,9=7", selected_buckets, 3)

    assert counts == {
        "easy/5": 5,
        "easy/7": 2,
        "easy/9": 7,
        "hard/5": 5,
        "hard/7": 2,
        "hard/9": 7,
    }


def test_parse_batch_count_overrides_allows_exact_bucket_override(tmp_path) -> None:
    selected_buckets = _batch_bucket_configs(tmp_path)

    counts = _parse_batch_count_overrides(
        "5=5,7=2,9=7,hard/9=8",
        selected_buckets,
        3,
    )

    assert counts["easy/9"] == 7
    assert counts["hard/9"] == 8
