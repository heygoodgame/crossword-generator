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


def test_used_answer_set_short_cap_excludes_after_n_puzzles() -> None:
    """Short glue is excluded only once it has appeared in `short_cap` puzzles."""
    used = _UsedAnswerSet()
    used.add(["ALL", "OWE", "OCEAN"])  # puzzle 1
    used.add(["ALL", "ALL", "TED"])  # puzzle 2 (repeat within one puzzle counts once)

    # Midi floor without a cap: 3-letter glue never excluded.
    assert used.snapshot(min_length=4, short_cap=0) == {"OCEAN"}
    # Cap 2: ALL has been in two puzzles, so it is now excluded; OWE/TED stay.
    assert used.snapshot(min_length=4, short_cap=2) == {"OCEAN", "ALL"}
    # Cap 1 behaves like a hard floor of 3.
    assert used.snapshot(min_length=4, short_cap=1) == {"OCEAN", "ALL", "OWE", "TED"}
    # Answers at/above the floor never depend on the cap.
    assert "OCEAN" in used.snapshot(min_length=4, short_cap=99)


def test_used_answer_set_short_counts_only_below_floor() -> None:
    used = _UsedAnswerSet()
    used.add(["ALL", "OCEAN"])
    used.add(["ALL", "ATE"])

    assert used.short_counts(min_length=4) == {"ALL": 2, "ATE": 1}
    # With a mini floor of 3 nothing is "short" — it is all hard-excluded.
    assert used.short_counts(min_length=3) == {}


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


def test_generate_pilot_batch_short_glue_soft_penalty_and_cap(
    tmp_path,
    monkeypatch,
) -> None:
    """Midi batch-mates' 3-letter glue is penalised, then capped, not excluded
    outright; 4+ letter answers are hard-excluded after one use."""
    from crossword_generator.data_store import RecentDailyAnswers

    run_kwargs: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        return RecentDailyAnswers(
            answers=[],
            window_days=kwargs.get("window_days") or 7,
            first_unscheduled_date="2026-10-01",
            since_date="2026-09-01",
            forward_days=13,
            counts={"ETA": 3} if kwargs.get("window_days") == 30 else None,
            count_window_days=90,
        )

    # 3x3 solution: rows ALL / TED / OWE and matching columns, so every
    # puzzle in the batch contributes the same 3-letter glue plus one
    # distinct 4+ letter answer via a wider row.
    def fake_run_batch_item(**kwargs):
        run_kwargs.append(kwargs)
        seed = kwargs["seed"]
        output_path = (
            kwargs["output_root"] / "easy" / "9x9" / f"seed-{seed:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        long_word = ["OCEAN", "PLANT", "STORM"][seed - 1]
        solution = [
            ["A", "L", "L", "#", "#"],
            ["T", "E", "D", "#", "#"],
            ["O", "W", "E", "#", "#"],
            ["#", "#", "#", "#", "#"],
            list(long_word),
        ]
        output_path.write_text(
            json.dumps({"solution": solution, "clues": {"Across": [], "Down": []}})
        )
        return {
            "difficulty": "easy",
            "size": 9,
            "seed": seed,
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
            "--buckets", "easy/9",
            "--count", "3",
            "--seed-start", "1",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
            "--intra-batch-short-window", "0",
            "--intra-batch-short-cap", "2",
            "--intra-batch-short-penalty", "8",
        ],
    )

    assert result.exit_code == 0, result.output
    first, second, third = run_kwargs

    # Puzzle 1: nothing used yet; only the server counts reach the CSP.
    assert first["excluded_fill_words"] == set()
    assert first["answer_usage_counts"] == {"ETA": 3}

    # Puzzle 2: OCEAN (4+) hard-excluded; the glue is *penalised* (8 per
    # prior use, on top of the schedule count) but still fillable.
    assert second["excluded_fill_words"] == {"OCEAN"}
    counts = second["answer_usage_counts"]
    assert counts["ETA"] == 3
    assert counts["ALL"] == 8 and counts["OWE"] == 8 and counts["ATO"] == 8
    assert "ALL" not in second["excluded_fill_words"]
    assert second["answer_novelty_candidates"] == 4

    # Puzzle 3: the glue has now appeared in two puzzles, so the cap of 2
    # excludes it; the penalty has doubled for anything still short.
    assert {"OCEAN", "PLANT", "ALL", "TED", "OWE"} <= third["excluded_fill_words"]
    assert third["answer_usage_counts"]["ALL"] == 16

    manifest = json.loads((tmp_path / "batch" / "manifest.json").read_text())
    assert manifest["intra_batch_short_policy"] == {
        "window_days": 0,
        "cap": 2,
        "penalty": 8.0,
        "penalty_applied": True,
    }


def test_generate_pilot_batch_short_window_follows_seed_order(
    tmp_path,
    monkeypatch,
) -> None:
    """Glue is hard-excluded only from batch-mates within +/-2 days (seed
    rank), so the set schedules in seed order; older glue stays available."""
    from crossword_generator.data_store import RecentDailyAnswers

    run_kwargs: list[dict[str, object]] = []
    # Distinct 3x3 glue block per puzzle so exclusions are attributable.
    blocks = {
        1: ["ALL", "TED", "OWE"],
        2: ["EGG", "RYE", "APR"],
        3: ["GPS", "OLE", "HMM"],
        4: ["PAL", "ILL", "TRI"],
    }

    def fake_fetch(**kwargs):
        return RecentDailyAnswers(
            answers=[],
            window_days=kwargs.get("window_days") or 7,
            first_unscheduled_date="2026-10-01",
            since_date="2026-09-01",
            forward_days=13,
        )

    def fake_run_batch_item(**kwargs):
        run_kwargs.append(kwargs)
        seed = kwargs["seed"]
        output_path = (
            kwargs["output_root"] / "easy" / "9x9" / f"seed-{seed:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [list(r) + ["#", "#"] for r in blocks[seed]]
        rows.append(["#"] * 5)
        rows.append(list(["OCEAN", "PLANT", "STORM", "CLOUD"][seed - 1]))
        output_path.write_text(
            json.dumps({"solution": rows, "clues": {"Across": [], "Down": []}})
        )
        return {
            "difficulty": "easy",
            "size": 9,
            "seed": seed,
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
            "--buckets", "easy/9",
            "--count", "4",
            "--seed-start", "1",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
            "--intra-batch-short-window", "2",
            "--intra-batch-short-cap", "0",
            "--intra-batch-short-penalty", "0",
        ],
    )

    assert result.exit_code == 0, result.output
    excluded = [kw["excluded_fill_words"] for kw in run_kwargs]

    # Day 2 sees day 1's glue; day 3 sees days 1-2; day 4 sees days 2-3 only
    # (day 1 is 3 days back, outside the scheduler's window).
    assert "ALL" in excluded[1] and "ALL" in excluded[2]
    assert "ALL" not in excluded[3]
    assert {"EGG", "GPS"} <= excluded[3]
    # 4+ letter answers are always excluded once used, regardless of day.
    assert {"OCEAN", "PLANT", "STORM"} <= excluded[3]
    # With penalty 0 no synthetic counts reach the CSP.
    assert all(kw["answer_usage_counts"] is None for kw in run_kwargs)


def test_generate_pilot_batch_prior_manifest_seeds_used_answers(
    tmp_path,
    monkeypatch,
) -> None:
    """A continuation run treats a prior manifest's puzzles as batch-mates."""
    from crossword_generator.data_store import RecentDailyAnswers

    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior_results = []
    for seed, long_word in ((1, "OCEAN"), (2, "PLANT")):
        path = prior_dir / f"seed-{seed}.ipuz"
        path.write_text(
            json.dumps(
                {
                    "solution": [
                        ["A", "L", "L", "#", "#"],
                        ["T", "E", "D", "#", "#"],
                        ["O", "W", "E", "#", "#"],
                        ["#", "#", "#", "#", "#"],
                        list(long_word),
                    ],
                    "clues": {"Across": [], "Down": []},
                }
            )
        )
        prior_results.append(
            {
                "difficulty": "easy",
                "size": 9,
                "seed": seed,
                "success": True,
                "output_path": path.name,
            }
        )
    # A failed prior result must be ignored even though it has no file.
    prior_results.append(
        {
            "difficulty": "easy",
            "size": 9,
            "seed": 99,
            "success": False,
            "output_path": "missing.ipuz",
        }
    )
    prior_manifest = prior_dir / "manifest.json"
    prior_manifest.write_text(json.dumps({"results": prior_results}))

    run_kwargs: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        return RecentDailyAnswers(
            answers=[],
            window_days=kwargs.get("window_days") or 7,
            first_unscheduled_date="2026-10-01",
            since_date="2026-09-01",
            forward_days=13,
        )

    def fake_run_batch_item(**kwargs):
        run_kwargs.append(kwargs)
        output_path = (
            kwargs["output_root"] / "easy" / "9x9" / f"seed-{kwargs['seed']:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"solution":[["S","T","O","R","M"]]}')
        return {
            "difficulty": "easy",
            "size": 9,
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
            "--buckets", "easy/9",
            "--count", "1",
            "--seed-start", "3",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
            "--prior-batch-manifest", str(prior_manifest),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Seeded intra-batch used answers from 2 prior puzzle(s)" in result.output
    (kwargs,) = run_kwargs
    # Both prior long answers are hard-excluded; the priors occupy days 1-2
    # and this puzzle is day 3, so their glue is inside the +/-2-day window.
    assert {"OCEAN", "PLANT", "ALL", "TED", "OWE"} <= kwargs["excluded_fill_words"]
    # No server counts, but the seeded glue still drives the soft penalty
    # (2 prior uses x default penalty 2).
    assert kwargs["answer_usage_counts"]["ALL"] == 4

    manifest = json.loads((tmp_path / "batch" / "manifest.json").read_text())
    assert manifest["prior_batch_manifests"]["puzzles_seeded"] == 2


def test_generate_pilot_batch_prior_manifest_refills_middle_day(
    tmp_path,
    monkeypatch,
) -> None:
    """Days are seed ranks over priors + this run, so reusing a dropped seed
    refills that day and sees only its +/-2-day neighbours' glue."""
    from crossword_generator.data_store import RecentDailyAnswers

    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    glue = {
        1: ["ALL", "TED", "OWE"],
        2: ["EGG", "RYE", "APR"],
        3: ["GPS", "OLE", "HMM"],
        7: ["PAL", "ILL", "TRI"],
        8: ["CUE", "PHD", "MET"],
        9: ["ATM", "CPR", "TNT"],
    }
    prior_results = []
    for seed, rows in glue.items():
        path = prior_dir / f"seed-{seed}.ipuz"
        solution = [list(r) + ["#", "#"] for r in rows] + [["#"] * 5]
        path.write_text(json.dumps({"solution": solution, "clues": {}}))
        prior_results.append(
            {
                "difficulty": "easy",
                "size": 9,
                "seed": seed,
                "success": True,
                "output_path": path.name,
            }
        )
    prior_manifest = prior_dir / "manifest.json"
    prior_manifest.write_text(json.dumps({"results": prior_results}))

    run_kwargs: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        return RecentDailyAnswers(
            answers=[],
            window_days=kwargs.get("window_days") or 7,
            first_unscheduled_date="2026-10-01",
            since_date="2026-09-01",
            forward_days=13,
        )

    def fake_run_batch_item(**kwargs):
        run_kwargs.append(kwargs)
        output_path = (
            kwargs["output_root"] / "easy" / "9x9" / f"seed-{kwargs['seed']:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"solution":[["S","T","O","R","M"]]}')
        return {
            "difficulty": "easy",
            "size": 9,
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

    # Prior seeds 1,2,3,7,8,9 plus this run's seed 5 rank as days 1-7 —
    # seed 5 refills day 4 and must see only days 2-6.
    result = CliRunner().invoke(
        main,
        [
            "generate-pilot-batch",
            "--output-root", str(tmp_path / "batch"),
            "--batch-id", "test-batch",
            "--buckets", "easy/9",
            "--count", "1",
            "--seed-start", "5",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
            "--intra-batch-short-penalty", "0",
            "--prior-batch-manifest", str(prior_manifest),
        ],
    )
    assert result.exit_code == 0, result.output
    (kwargs,) = run_kwargs
    excluded = kwargs["excluded_fill_words"]
    # Days 2, 3, 5, 6 (seeds 2, 3, 7, 8) are within +/-2 of day 4; days 1
    # and 7 (seeds 1, 9) are not.
    assert {"EGG", "GPS", "PAL", "CUE"} <= excluded
    assert {"ALL", "ATM"}.isdisjoint(excluded)
    assert "Seeded intra-batch used answers from 6 prior puzzle(s)" in result.output

    # Reusing a seed that is still in the prior manifest is an error.
    clash = CliRunner().invoke(
        main,
        [
            "generate-pilot-batch",
            "--output-root", str(tmp_path / "batch2"),
            "--batch-id", "test-batch",
            "--buckets", "easy/9",
            "--count", "1",
            "--seed-start", "7",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
            "--prior-batch-manifest", str(prior_manifest),
        ],
    )
    assert clash.exit_code != 0
    assert "already appear in a --prior-batch-manifest" in clash.output


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


def _target_slot(day: int, answers: set[str], epoch, **kw):
    import datetime as dt

    from crossword_generator.schedule_targeting import ScheduledSlot

    fields = dict(game_key="midicrossword", track="easy", size=9)
    fields.update(kw)
    return ScheduledSlot(
        day_number=day,
        date=epoch + dt.timedelta(days=day - 1),
        answers=frozenset(answers),
        **fields,
    )


def test_generate_pilot_batch_targets_open_days_with_per_day_exclusions(
    tmp_path,
    monkeypatch,
) -> None:
    """--target-* fills each open day of one game/track, excluding exactly the
    answers the scheduler would reject on THAT day, and stamps target_date."""
    import datetime as dt

    from crossword_generator.data_store import (
        RecentDailyAnswers,
        records_from_manifest,
    )

    epoch = dt.date(2099, 1, 1)
    # Easy midi scheduled on days 100, 101, 103, 106; hole at 102, 104, 105.
    slots = [
        _target_slot(100, {"SNORE", "DOG"}, epoch),
        _target_slot(101, {"MARIA"}, epoch),
        _target_slot(103, {"STEAK", "CBS"}, epoch),
        _target_slot(106, {"OCEAN"}, epoch),
        # Hard track and a mini also constrain (cross-track, cross-game) but
        # do not occupy easy days.
        _target_slot(104, {"UFOS", "ACT"}, epoch, track="hard"),
        _target_slot(112, {"PLANT"}, epoch, game_key="minicrossword", size=5),
    ]
    run_kwargs: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        return RecentDailyAnswers(
            answers=["JUNKWORD", "ZAP"],
            window_days=kwargs.get("window_days") or 30,
            first_unscheduled_date="2099-01-01",
            since_date="2098-12-01",
            forward_days=13,
            counts={"ETA": 9},
            count_window_days=90,
        )

    def fake_run_batch_item(**kwargs):
        run_kwargs.append(kwargs)
        seed = kwargs["seed"]
        output_path = (
            kwargs["output_root"] / "easy" / "9x9" / f"seed-{seed:03d}.ipuz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [list("ABC") + ["#", "#"], ["#"] * 5, list(f"WORD{seed}")]
        output_path.write_text(
            json.dumps({"solution": rows, "clues": {"Across": [], "Down": []}})
        )
        return {
            "difficulty": "easy",
            "size": 9,
            "seed": seed,
            "success": True,
            "runtime_seconds": 0.0,
            "output_path": str(output_path),
            "clue_score": 80.0,
            "fill_score": 50.0,
        }

    import crossword_generator.data_store as data_store_module

    monkeypatch.setattr(
        data_store_module, "fetch_recent_daily_answers", fake_fetch
    )
    monkeypatch.setattr(
        cli_module, "_load_daily_schedule_slots", lambda **kwargs: slots
    )
    monkeypatch.setattr(cli_module, "_run_batch_item", fake_run_batch_item)

    start = (epoch + dt.timedelta(days=99)).isoformat()  # day 100
    through = (epoch + dt.timedelta(days=105)).isoformat()  # day 106
    result = CliRunner().invoke(
        main,
        [
            "generate-pilot-batch",
            "--output-root", str(tmp_path / "batch"),
            "--batch-id", "target-batch",
            "--target-game", "midicrossword",
            "--target-track", "easy",
            "--target-from", start,
            "--target-through", through,
            "--seed-start", "1",
            "--no-avoid-existing-clues",
            "--no-refresh-dictionaries",
            "--no-exclude-scheduled-sixty",
            "--no-llm-log",
            "--intra-batch-short-penalty", "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Targeting 3 open midicrossword easy day(s)" in result.output
    assert "skipping the first-unscheduled-slot recent-answer window" in result.output

    # One work item per open day, in date order, all 9x9 easy.
    assert [kw["seed"] for kw in run_kwargs] == [1, 2, 3]
    assert all(kw["size"] == 9 and kw["difficulty"] == "easy" for kw in run_kwargs)
    excluded = [kw["excluded_fill_words"] for kw in run_kwargs]

    # Day 102: SNORE (day 100) and MARIA (101), STEAK (103), UFOS (104),
    # OCEAN (106) all within +/-6; DOG (100) within +/-2; CBS (103) within
    # +/-2 — but ACT (104) is 2 days away too. PLANT (mini, 112) is out.
    day_102 = {"SNORE", "MARIA", "STEAK", "UFOS", "OCEAN", "DOG", "CBS", "ACT"}
    assert day_102 <= excluded[0]
    assert "PLANT" not in excluded[0]
    # Day 105: DOG (100) is 5 days away -> short glue stays available; CBS
    # (103) and ACT (104) are within 2.
    assert "DOG" not in excluded[2]
    assert {"CBS", "ACT", "SNORE", "OCEAN"} <= excluded[2]
    # Variants of regular-window answers are excluded (OCEANS), never glue.
    assert "OCEANS" in excluded[0]
    # The anchored recent-answer list is NOT applied in targeted mode...
    assert "JUNKWORD" not in excluded[0] and "ZAP" not in excluded[0]
    # ...but its usage counts still feed the soft penalty.
    assert run_kwargs[0]["answer_usage_counts"] == {"ETA": 9}
    # Intra-batch dedup: puzzle 2 sees puzzle 1's 4+ letter answer.
    assert "WORD1" in excluded[1]

    manifest = json.loads((tmp_path / "batch" / "manifest.json").read_text())
    assert manifest["target"]["game_key"] == "midicrossword"
    assert [d["day_number"] for d in manifest["target"]["days"]] == [102, 104, 105]
    assert manifest["bucket_counts"] == {"easy/9": 3}
    results = manifest["results"]
    assert [r["target_day_number"] for r in results] == [102, 104, 105]
    assert results[0]["target_date"] == (epoch + dt.timedelta(days=101)).isoformat()
    assert results[0]["target_track"] == "easy"
    assert results[0]["schedule_excluded_count"] >= 8

    records = records_from_manifest(tmp_path / "batch" / "manifest.json")
    assert records[0]["metadata"]["target_date"] == results[0]["target_date"]
    assert records[0]["metadata"]["publish_slot"] == results[0]["target_date"]
    assert records[0]["metadata"]["target_day_number"] == 102
    assert records[0]["metadata"]["target_track"] == "easy"


def test_generate_pilot_batch_target_dates_rejects_occupied_and_picks_mini_size(
    tmp_path,
    monkeypatch,
) -> None:
    import datetime as dt

    epoch = dt.date(2099, 1, 1)  # a Thursday
    slots = [
        _target_slot(2, {"AAAA"}, epoch, game_key="minicrossword", size=5),
        _target_slot(5, {"BBBB"}, epoch, game_key="midicrossword"),
    ]
    monkeypatch.setattr(
        cli_module, "_load_daily_schedule_slots", lambda **kwargs: slots
    )
    seen: list[tuple[str, int, int]] = []

    def fake_run_batch_item(**kwargs):
        seen.append((kwargs["difficulty"], kwargs["size"], kwargs["seed"]))
        return {
            "difficulty": kwargs["difficulty"],
            "size": kwargs["size"],
            "seed": kwargs["seed"],
            "success": False,
            "runtime_seconds": 0.0,
            "output_path": "",
            "error_message": "skipped",
        }

    monkeypatch.setattr(cli_module, "_run_batch_item", fake_run_batch_item)
    common = [
        "generate-pilot-batch",
        "--output-root", str(tmp_path / "batch"),
        "--batch-id", "target-batch",
        "--target-game", "minicrossword",
        "--target-track", "easy",
        "--seed-start", "7",
        "--no-avoid-existing-clues",
        "--no-refresh-dictionaries",
        "--no-exclude-scheduled-sixty",
        "--no-exclude-recent-answers",
        "--no-llm-log",
    ]
    # Day 2 (2099-01-02) is already scheduled for the mini easy track.
    occupied = CliRunner().invoke(
        main, [*common, "--target-dates", "2099-01-02,2099-01-03"]
    )
    assert occupied.exit_code != 0
    assert "already scheduled for minicrossword easy: 2099-01-02" in occupied.output

    # 2099-01-03 is a Saturday (7x7), 2099-01-05 a Monday (5x5).
    ok = CliRunner().invoke(
        main, [*common, "--target-dates", "2099-01-03,2099-01-05"]
    )
    assert ok.exit_code == 0, ok.output
    assert seen == [("easy", 7, 7), ("easy", 5, 8)]

    # --buckets is incompatible with targeting.
    bad = CliRunner().invoke(
        main, [*common, "--target-through", "2099-01-05", "--buckets", "easy/5"]
    )
    assert bad.exit_code != 0
    assert "drop --buckets" in bad.output
