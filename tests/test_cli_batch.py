"""Tests for batch CLI helpers."""

import logging
import threading
from pathlib import Path

from crossword_generator.cli import (
    _apply_dictionary_overrides,
    _batch_bucket_configs,
    _extract_grid_variant,
    _failure_category,
    _parse_batch_count_overrides,
    _referenced_dictionary_filenames,
    _run_duplicate_sweep,
    _summarize_batch_results,
    _ThreadFilter,
    _UsedAnswerSet,
    _write_filtered_dictionary,
)
from crossword_generator.clue_history import (
    ClueHistoryIndex,
    duplicate_error_message,
)
from crossword_generator.config import find_project_root, load_config
from crossword_generator.models import ClueEntry, PuzzleEnvelope, PuzzleType


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


def test_write_filtered_dictionary_removes_scheduled_words(
    tmp_path,
) -> None:
    dictionaries = tmp_path / "dictionaries"
    dictionaries.mkdir()
    (dictionaries / "hgg-60.txt").write_text(
        "moonwalk;60\njackpots;60\nzucchini;60\n"
    )
    output_root = tmp_path / "batch"
    output_root.mkdir()

    path, removed = _write_filtered_dictionary(
        tmp_path,
        output_root,
        "hgg-60.txt",
        ["MOONWALK", " jackpots ", "NOTINLIST"],
        "hgg-60-scheduled-filtered.txt",
    )

    assert removed == 2
    assert path == str(output_root / "hgg-60-scheduled-filtered.txt")
    assert (output_root / "hgg-60-scheduled-filtered.txt").read_text() == (
        "zucchini;60\n"
    )


def test_apply_dictionary_overrides_swaps_sixty_references() -> None:
    project_root = find_project_root()
    config = load_config(project_root / "config.hard7.yaml")
    override = "/tmp/some-batch/hgg-60-scheduled-filtered.txt"

    _apply_dictionary_overrides(config, {"hgg-60.txt": override})

    assert override in config.dictionary.additional_paths
    assert override in config.dictionary.themed_additional_paths
    assert override in config.fill.csp.additional_dictionary_paths
    for paths in (
        config.dictionary.additional_paths,
        config.dictionary.themed_additional_paths,
        config.fill.csp.additional_dictionary_paths,
    ):
        assert all("hgg-60.txt" != Path(p).name for p in paths)


def test_apply_dictionary_overrides_swaps_primary_paths() -> None:
    project_root = find_project_root()
    config = load_config(project_root / "config.easy.yaml")
    base_name = Path(config.dictionary.path).name
    override = f"/tmp/some-batch/{Path(base_name).stem}-recent-filtered.txt"

    _apply_dictionary_overrides(config, {base_name: override})

    assert config.dictionary.path == override
    assert config.fill.csp.dictionary_path == override


def test_referenced_dictionary_filenames_skips_zero_count_buckets() -> None:
    project_root = find_project_root()
    buckets = [
        b for b in _batch_bucket_configs(project_root)
        if f"{b[0]}/{b[1]}" in ("easy/5", "hard/7")
    ]
    counts = {"easy/5": 1, "hard/7": 0}

    filenames = _referenced_dictionary_filenames(buckets, counts)

    easy_config = load_config(project_root / "config.easy.yaml")
    assert Path(easy_config.dictionary.path).name in filenames
    # hard/7 has count 0, so its hgg-60 merge is not referenced.
    assert "hgg-60.txt" not in filenames


class _StubExporter:
    """Records export calls without writing real .ipuz files."""

    file_extension = ".ipuz"

    def __init__(self) -> None:
        self.exports: list[tuple[object, Path]] = []

    def export(self, envelope: object, output_dir: Path) -> Path:
        raise NotImplementedError

    def export_to_file(self, envelope: object, path: Path) -> Path:
        self.exports.append((envelope, path))
        return path


class _StubClueStep:
    """Stands in for ClueWithGradingStep.repair_external_duplicates."""

    def __init__(self, replacement: str | None) -> None:
        self._replacement = replacement
        self.calls = 0

    def repair_external_duplicates(self, envelope, hits):  # noqa: ANN001
        self.calls += 1
        if self._replacement is None:
            # Stuck: soft-error every hit, like the real step.
            return envelope.model_copy(
                update={
                    "errors": [
                        *envelope.errors,
                        *(duplicate_error_message(h) for h in hits),
                    ]
                }
            )
        new_clues = []
        hit_keys = {(h.clue.number, h.clue.direction) for h in hits}
        for clue in envelope.clues:
            if (clue.number, clue.direction) in hit_keys:
                new_clues.append(clue.model_copy(update={"clue": self._replacement}))
            else:
                new_clues.append(clue)
        return envelope.model_copy(update={"clues": new_clues})


def _sweep_envelope(clue_text: str) -> PuzzleEnvelope:
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=3,
        clues=[
            ClueEntry(number=1, direction="across", answer="CAT", clue=clue_text)
        ],
    )


def _sweep_result(
    seed: int,
    envelope: PuzzleEnvelope,
    clue_step: object,
    output_path: Path,
) -> dict[str, object]:
    return {
        "difficulty": "easy",
        "size": 5,
        "seed": seed,
        "success": True,
        "error_message": None,
        "failure_category": None,
        "_sweep": {
            "envelope": envelope,
            "clue_step": clue_step,
            "output_path": output_path,
        },
    }


def test_duplicate_sweep_repairs_cross_puzzle_collision(tmp_path: Path) -> None:
    """Second puzzle with the same clue gets repaired and re-exported; the
    first keeps its clue untouched."""
    env_a = _sweep_envelope("Feline pet")
    env_b = _sweep_envelope("Feline pet")
    history = ClueHistoryIndex()
    history.add_clues(env_a.clues)
    history.add_clues(env_b.clues)

    step = _StubClueStep("Fresh feline clue")
    exporter = _StubExporter()
    results = [
        _sweep_result(1, env_a, step, tmp_path / "a.ipuz"),
        _sweep_result(2, env_b, step, tmp_path / "b.ipuz"),
    ]

    stats = _run_duplicate_sweep(results, history, exporter=exporter)

    assert stats == {"checked": 2, "repaired": 1, "unresolved": 0}
    assert step.calls == 1
    assert [path for _, path in exporter.exports] == [tmp_path / "b.ipuz"]
    repaired_env = exporter.exports[0][0]
    assert repaired_env.clues[0].clue == "Fresh feline clue"
    # Sweep context is stripped so the manifest stays JSON-serializable.
    assert all("_sweep" not in r for r in results)
    assert results[1]["error_message"] is None
    # The replacement is registered in the shared history.
    assert "Fresh feline clue" in history.clues_for_answer("CAT")


def test_duplicate_sweep_unresolved_blocks_upload(tmp_path: Path) -> None:
    """A collision that repair can't clear becomes a DUPLICATE: soft error in
    the manifest result, which the upload guard reads."""
    env_a = _sweep_envelope("Feline pet")
    env_b = _sweep_envelope("Feline pet")
    history = ClueHistoryIndex()
    history.add_clues(env_a.clues)
    history.add_clues(env_b.clues)

    step = _StubClueStep(None)  # stuck
    exporter = _StubExporter()
    results = [
        _sweep_result(1, env_a, step, tmp_path / "a.ipuz"),
        _sweep_result(2, env_b, step, tmp_path / "b.ipuz"),
    ]

    stats = _run_duplicate_sweep(results, history, exporter=exporter)

    assert stats == {"checked": 2, "repaired": 0, "unresolved": 1}
    assert results[0]["error_message"] is None
    assert "DUPLICATE:" in str(results[1]["error_message"])


def test_duplicate_sweep_distinct_clues_untouched(tmp_path: Path) -> None:
    env_a = _sweep_envelope("Feline pet")
    env_b = _sweep_envelope("Whiskers wearer")
    history = ClueHistoryIndex()
    history.add_clues(env_a.clues)
    history.add_clues(env_b.clues)

    step = _StubClueStep("unused")
    exporter = _StubExporter()
    results = [
        _sweep_result(1, env_a, step, tmp_path / "a.ipuz"),
        _sweep_result(2, env_b, step, tmp_path / "b.ipuz"),
    ]

    stats = _run_duplicate_sweep(results, history, exporter=exporter)

    assert stats == {"checked": 2, "repaired": 0, "unresolved": 0}
    assert step.calls == 0
    assert exporter.exports == []


def test_batch_bucket_order_is_most_constrained_first() -> None:
    """9x9 midis generate before 7x7 then 5x5 minis so minis defer to midis."""
    order = [(d, s) for d, s, _, _ in _batch_bucket_configs(Path("."))]

    sizes_in_order = [s for _, s in order]
    # Every 9 must precede every 7, and every 7 must precede every 5.
    assert sizes_in_order == sorted(sizes_in_order, reverse=True)
    assert order[0][1] == 9 and order[-1][1] == 5


def test_used_answer_set_mini_excludes_three_letter_midi_keeps() -> None:
    """A 3-letter answer a midi used is excluded for minis, kept for midis."""
    used = _UsedAnswerSet()
    used.add(["ICE", "OCEAN", " ", "AT"])  # AT (<3) dropped; ICE/OCEAN kept

    # Mini floor (3): excludes the 3-letter glue too.
    assert used.snapshot(min_length=3) == {"ICE", "OCEAN"}
    # Midi floor (4): keeps its 3-letter glue fillable.
    assert used.snapshot(min_length=4) == {"OCEAN"}
    # Sub-3 answers are never tracked.
    assert "AT" not in used.snapshot(min_length=3)


def test_used_answer_set_is_thread_safe() -> None:
    used = _UsedAnswerSet()

    def worker(tid: int) -> None:
        for i in range(100):
            used.add([f"WORD{tid}X{i}"])

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(used.snapshot(min_length=4)) == 8 * 100
