"""Tests for batch CLI helpers."""

import json
import logging
import threading
from pathlib import Path

from click.testing import CliRunner

from crossword_generator import cli as cli_module
from crossword_generator.cli import (
    _OPT_IN_DIFFICULTIES,
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
    main,
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
    # A default batch (no --buckets) excludes opt-in difficulties like starter,
    # so the size-ratio override only fans out across easy/hard.
    selected_buckets = [
        b for b in _batch_bucket_configs(tmp_path) if b[0] not in _OPT_IN_DIFFICULTIES
    ]

    counts = _parse_batch_count_overrides("5=5,7=2,9=7", selected_buckets, 3)

    assert counts == {
        "easy/5": 5,
        "easy/7": 2,
        "easy/9": 7,
        "hard/5": 5,
        "hard/7": 2,
        "hard/9": 7,
    }


def test_starter_bucket_is_registered_but_opt_in(tmp_path) -> None:
    buckets = _batch_bucket_configs(tmp_path)
    configs = {f"{d}/{s}": path.name for d, s, _, path in buckets}

    # Registered (so --buckets starter/5 resolves) ...
    assert configs["starter/5"] == "config.starter.yaml"
    # ... but excluded from a default full-batch selection.
    assert "starter" in _OPT_IN_DIFFICULTIES
    default = [b for b in buckets if b[0] not in _OPT_IN_DIFFICULTIES]
    assert all(d != "starter" for d, _, _, _ in default)


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

    path, removed, removed_variants = _write_filtered_dictionary(
        tmp_path,
        output_root,
        "hgg-60.txt",
        ["MOONWALK", " jackpots ", "NOTINLIST"],
        "hgg-60-scheduled-filtered.txt",
    )

    assert removed == 2
    assert removed_variants == 0
    assert path == str(output_root / "hgg-60-scheduled-filtered.txt")
    assert (output_root / "hgg-60-scheduled-filtered.txt").read_text() == (
        "zucchini;60\n"
    )


def test_write_filtered_dictionary_counts_variant_rows_separately(
    tmp_path,
) -> None:
    dictionaries = tmp_path / "dictionaries"
    dictionaries.mkdir()
    (dictionaries / "hgg-easy.txt").write_text(
        "party;55\nparties;55\npartied;50\nart;60\narts;60\nzebra;55\n"
    )
    output_root = tmp_path / "batch"
    output_root.mkdir()

    path, removed, removed_variants = _write_filtered_dictionary(
        tmp_path,
        output_root,
        "hgg-easy.txt",
        ["PARTY", "ART"],
        "hgg-easy-recent-filtered.txt",
        # ART is a 3-letter base, so the caller would not have expanded it;
        # ARTS is passed here only to prove the variant set is honoured and
        # counted separately from base exclusions.
        variant_answers=["PARTIES", "PARTIED", "ARTS", "PARTY"],
    )

    assert removed == 2
    assert removed_variants == 3
    assert (output_root / "hgg-easy-recent-filtered.txt").read_text() == (
        "zebra;55\n"
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


def test_generate_pilot_batch_refreshes_dictionaries_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    refreshed: list[Path] = []
    project_root = find_project_root()

    def fake_refresh(**kwargs):
        refreshed.append(kwargs["project_root"])

    def fake_run_batch_item(**kwargs):
        output_path = (
            kwargs["output_root"]
            / kwargs["difficulty"]
            / f"{kwargs['size']}x{kwargs['size']}"
            / f"seed-{kwargs['seed']:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '{"solution":[["A","P","P","L","E"]],"clues":{"Across":[],"Down":[]}}'
        )
        return {
            "difficulty": kwargs["difficulty"],
            "size": kwargs["size"],
            "seed": kwargs["seed"],
            "success": True,
            "runtime_seconds": 0.0,
            "output_path": str(output_path),
            "clue_score": 80.0,
        }

    monkeypatch.setattr(
        cli_module, "_refresh_dictionaries_for_generation", fake_refresh
    )
    monkeypatch.setattr(cli_module, "_run_batch_item", fake_run_batch_item)

    result = CliRunner().invoke(
        main,
        [
            "generate-pilot-batch",
            "--output-root", str(tmp_path / "batch"),
            "--batch-id", "test-batch",
            "--buckets", "easy/5",
            "--count", "1",
            "--no-avoid-existing-clues",
            "--no-exclude-recent-answers",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
        ],
    )

    assert result.exit_code == 0, result.output
    assert refreshed == [project_root]


def test_generate_pilot_batch_allows_dictionary_refresh_opt_out(
    tmp_path,
    monkeypatch,
) -> None:
    refreshed = False

    def fake_refresh(**_kwargs):
        nonlocal refreshed
        refreshed = True

    def fake_run_batch_item(**kwargs):
        output_path = (
            kwargs["output_root"]
            / kwargs["difficulty"]
            / f"{kwargs['size']}x{kwargs['size']}"
            / f"seed-{kwargs['seed']:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '{"solution":[["A","P","P","L","E"]],"clues":{"Across":[],"Down":[]}}'
        )
        return {
            "difficulty": kwargs["difficulty"],
            "size": kwargs["size"],
            "seed": kwargs["seed"],
            "success": True,
            "runtime_seconds": 0.0,
            "output_path": str(output_path),
            "clue_score": 80.0,
        }

    monkeypatch.setattr(
        cli_module, "_refresh_dictionaries_for_generation", fake_refresh
    )
    monkeypatch.setattr(cli_module, "_run_batch_item", fake_run_batch_item)

    result = CliRunner().invoke(
        main,
        [
            "generate-pilot-batch",
            "--output-root", str(tmp_path / "batch"),
            "--batch-id", "test-batch",
            "--buckets", "easy/5",
            "--count", "1",
            "--no-refresh-dictionaries",
            "--no-avoid-existing-clues",
            "--no-exclude-recent-answers",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
        ],
    )

    assert result.exit_code == 0, result.output
    assert refreshed is False


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


def test_generate_pilot_batch_daily_exclusions_merge_windows_and_pass_counts(
    tmp_path,
    monkeypatch,
) -> None:
    """Long window for 4+ letters, short window for glue, variants expanded,
    and the server usage counts reach the fill step with the daily defaults."""
    from crossword_generator.data_store import RecentDailyAnswers

    fetch_calls: list[dict[str, object]] = []
    run_kwargs: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        fetch_calls.append(kwargs)
        if kwargs.get("window_days") == 7:
            return RecentDailyAnswers(
                answers=["ETA", "WALK", "ART"],
                window_days=7,
                first_unscheduled_date="2026-08-22",
                since_date="2026-08-15",
                forward_days=13,
            )
        return RecentDailyAnswers(
            answers=["ERA", "PARTY", "WALK"],
            window_days=30,
            first_unscheduled_date="2026-08-22",
            since_date="2026-07-23",
            forward_days=13,
            counts={"ETA": 7, "WALK": 2},
            count_window_days=90,
        )

    def fake_run_batch_item(**kwargs):
        run_kwargs.append(kwargs)
        output_path = (
            kwargs["output_root"] / "easy" / "5x5" / f"seed-{kwargs['seed']:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '{"solution":[["A","P","P","L","E"]],"clues":{"Across":[],"Down":[]}}'
        )
        return {
            "difficulty": "easy",
            "size": 5,
            "seed": kwargs["seed"],
            "success": True,
            "runtime_seconds": 0.0,
            "output_path": str(output_path),
            "clue_score": 80.0,
        }

    import crossword_generator.data_store as data_store_module

    monkeypatch.setattr(
        data_store_module, "fetch_recent_daily_answers", fake_fetch
    )
    monkeypatch.setattr(cli_module, "_run_batch_item", fake_run_batch_item)

    result = CliRunner().invoke(
        main,
        [
            "generate-pilot-batch",
            "--output-root", str(tmp_path / "batch"),
            "--batch-id", "test-batch",
            "--buckets", "easy/5",
            "--count", "1",
            "--seed-start", "1",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
        ],
    )

    assert result.exit_code == 0, result.output
    # Two fetches: the 30-day (with counts) and the 7-day glue window.
    assert [call["window_days"] for call in fetch_calls] == [30, 7]
    assert fetch_calls[0]["count_window_days"] == 90
    assert fetch_calls[0]["forward_days"] == 13

    # Filtered dictionary: 4+ letter words from the long window, 3-letter
    # words from the short window only, plus inflectional variants.
    filtered = tmp_path / "batch" / "hgg-easy-recent-filtered.txt"
    words = {
        line.split(";", 1)[0].strip().upper()
        for line in filtered.read_text().splitlines()
        if line.strip()
    }
    assert {"PARTY", "WALK", "ETA", "ART"}.isdisjoint(words)
    # Variants come from the 7-day list only: WALK's are excluded, but
    # PARTY (30-day list only) keeps its inflections.
    assert {"WALKS", "WALKED", "WALKING"}.isdisjoint(words)
    assert "PARTIES" in words
    # ERA is a 3-letter answer that appears only in the 30-day list; glue
    # follows the 7-day window, so it stays available.
    assert "ERA" in words

    (kwargs,) = run_kwargs
    assert kwargs["answer_usage_counts"] == {"ETA": 7, "WALK": 2}
    assert kwargs["answer_novelty_candidates"] == 4
    assert kwargs["answer_usage_penalty"] is None  # config default (4.0)

    manifest = json.loads((tmp_path / "batch" / "manifest.json").read_text())
    recent = manifest["exclude_recent_answers"]
    assert recent["window_days"] == 30
    assert recent["short_window_days"] == 7
    assert recent["exclude_answer_variants"] is True
    assert recent["variant_rows_removed_by_dictionary"]["hgg-easy.txt"] >= 1
    assert manifest["daily_usage_penalty"]["applied"] is True
    assert manifest["daily_usage_penalty"]["novelty_candidates"] == 4


def test_generate_pilot_batch_degrades_without_server_counts(
    tmp_path,
    monkeypatch,
) -> None:
    from crossword_generator.data_store import RecentDailyAnswers

    run_kwargs: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        return RecentDailyAnswers(
            answers=["WALK"],
            window_days=kwargs.get("window_days") or 7,
            first_unscheduled_date="2026-08-22",
            since_date="2026-07-23",
            forward_days=13,
        )

    def fake_run_batch_item(**kwargs):
        run_kwargs.append(kwargs)
        output_path = (
            kwargs["output_root"] / "easy" / "5x5" / f"seed-{kwargs['seed']:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '{"solution":[["A","P","P","L","E"]],"clues":{"Across":[],"Down":[]}}'
        )
        return {
            "difficulty": "easy",
            "size": 5,
            "seed": kwargs["seed"],
            "success": True,
            "runtime_seconds": 0.0,
            "output_path": str(output_path),
            "clue_score": 80.0,
        }

    import crossword_generator.data_store as data_store_module

    monkeypatch.setattr(
        data_store_module, "fetch_recent_daily_answers", fake_fetch
    )
    monkeypatch.setattr(cli_module, "_run_batch_item", fake_run_batch_item)

    result = CliRunner().invoke(
        main,
        [
            "generate-pilot-batch",
            "--output-root", str(tmp_path / "batch"),
            "--batch-id", "test-batch",
            "--buckets", "easy/5",
            "--count", "1",
            "--seed-start", "1",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "no daily usage counts" in result.output
    (kwargs,) = run_kwargs
    assert kwargs["answer_usage_counts"] is None
    manifest = json.loads((tmp_path / "batch" / "manifest.json").read_text())
    assert manifest["daily_usage_penalty"]["applied"] is False
