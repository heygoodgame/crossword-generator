import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crossword_generator import content_shadow_study as study
from crossword_generator.content_shadow import ContentShadow
from tests.test_content_shadow import Adapter, envelope


def make_batch(path, *, probability=0.56, final=True, fail=False, clue=None):
    e = envelope()
    if clue:
        e.clues[0].clue = clue
    api = Adapter(fail=fail)
    original = api.evaluate

    def evaluate(state, questions):
        r = original(state, questions)
        r["probabilities"] = dict.fromkeys(questions, probability)
        return r

    api.evaluate = evaluate
    observer = ContentShadow(path / "content-shadow.json", adapter=api)
    observer.observe(e, phase="final" if final else "generated", quality_flags=set())
    return path


@pytest.fixture
def directory(tmp_path, monkeypatch):
    path = tmp_path / "study"
    monkeypatch.setenv("CROSSWORD_SHADOW_STUDY_DIR", str(path))
    return path


def test_auto_requires_key_respects_opt_out_and_checkpoint(
    directory, tmp_path, monkeypatch
):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert study.collection_status() == (False, "no TYPESAFE_API_KEY")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    assert study.collection_status() == (True, "collecting")
    assert study.collection_status(False)[0] is False
    monkeypatch.setattr(study, "TARGET", 1)
    result = study.update_study(make_batch(tmp_path / "batch"))
    assert result["phase"] == "review_due"
    assert study.collection_status()[0] is False
    assert study.collection_status(True)[0] is False
    assert "CHECKPOINT" in study.progress_line(result)


def test_copies_and_reingestion_do_not_inflate_progress(directory, tmp_path):
    batch = make_batch(tmp_path / "batch")
    first = study.update_study(batch)
    assert first["final_clues"] == 1
    assert first["awaiting_review"] == 1
    copy = tmp_path / "archive"
    shutil.copytree(batch, copy)
    assert study.update_study(copy)["final_clues"] == 1
    assert study.update_study(batch)["observed_cases"] == 1
    # Another generation of the identical final answer/clue also counts once.
    assert study.update_study(make_batch(tmp_path / "rerun"))["final_clues"] == 1


def test_borderline_and_sample_queue_preserves_reviews(
    directory, tmp_path, monkeypatch
):
    batch = make_batch(tmp_path / "borderline")
    study.update_study(batch)
    path = directory / "review.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["review_reason"] == "borderline"
    rows[0]["editor_verdict"] = "acceptable"
    rows[0]["editor_notes"] = "Editor decision survives another batch"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(study, "_sample", lambda _: True)
    result = study.update_study(
        make_batch(tmp_path / "clean", probability=0.1, clue="New clue")
    )
    assert result["final_clues"] == 2
    assert result["review_cases"] == 2 and result["awaiting_review"] == 1
    assert "random_unflagged" in path.read_text()
    assert "Editor decision survives another batch" in path.read_text()


def test_only_complete_successful_final_clues_count(directory, tmp_path):
    for name, kwargs in [("unfinished", {"final": False}), ("error", {"fail": True})]:
        assert (
            study.update_study(make_batch(tmp_path / name, **kwargs))["final_clues"]
            == 0
        )
    failed = make_batch(tmp_path / "failed")
    (failed / "manifest.json").write_text(
        json.dumps(
            {
                "output_root": str(failed),
                "results": [
                    {
                        "success": False,
                        "content_shadow_path": str(failed / "content-shadow.json"),
                    }
                ],
            }
        )
    )
    assert study.update_study(failed)["final_clues"] == 0


def test_concurrent_batches_and_explicit_decision(directory, tmp_path, monkeypatch):
    batches = [make_batch(tmp_path / str(i), clue=f"Clue {i}") for i in range(2)]
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(study.update_study, batches))
    assert study.status()["final_clues"] == 2
    result = study.update_study(decision="retire", notes="Reviewed and retired")
    assert result["phase"] == "retire"
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    assert study.collection_status()[0] is False
    with pytest.raises(ValueError):
        study.update_study(decision="revise", notes="")


def test_negative_sampling_reproducible_and_small():
    cases = [str(i) for i in range(1000)]
    chosen = {c for c in cases if study._sample(c)}
    assert chosen == {c for c in reversed(cases) if study._sample(c)}
    assert 20 < len(chosen) < 80


def test_console_progress_includes_zero_flag_batches(directory, tmp_path, capsys):
    from crossword_generator.cli import _announce_shadow

    _announce_shadow(reasons=["no TYPESAFE_API_KEY"])
    output = capsys.readouterr().out
    assert "0/500" in output and "0 awaiting review" in output
    assert "no TYPESAFE_API_KEY" in output and "Review:" in output
    _announce_shadow(make_batch(tmp_path / "batch", probability=0.1))
    output = capsys.readouterr().out
    assert "1/1 final clues scored" in output
    assert "0 additional flags" in output and "0 request errors" in output


def test_runs_that_did_not_opt_in_leave_the_study_alone(
    directory, tmp_path, monkeypatch
):
    from click.testing import CliRunner

    from crossword_generator import cli
    from crossword_generator.config import Config

    # A key in the environment is not consent: observation is off by default.
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    enabled = Config().grading.clue.content_shadow.enabled
    assert study.collection_status(enabled) == (False, study.DISABLED)
    monkeypatch.setattr(cli, "_load_existing_clue_history", lambda *a, **kw: 0)
    monkeypatch.setattr(
        cli,
        "_run_batch_item",
        lambda **kw: {
            "difficulty": "easy",
            "size": 5,
            "seed": kw["seed"],
            "success": False,
            "output_path": str(tmp_path / "missing.ipuz"),
            "runtime_seconds": 0,
            "content_shadow_path": None,
            "content_shadow_status": study.DISABLED,
        },
    )
    result = CliRunner().invoke(
        cli.main,
        [
            "generate-pilot-batch",
            "--output-root",
            str(tmp_path / "batch"),
            "--batch-id",
            "not-observed",
            "--buckets",
            "easy/5",
            "--count",
            "1",
            "--max-workers",
            "1",
            "--no-refresh-dictionaries",
            "--no-intra-batch-dedup",
            "--no-unlimited-answer-novelty",
            "--no-exclude-recent-answers",
            "--no-exclude-scheduled-sixty",
            "--no-exclude-existing-boards",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Jev shadow" not in result.output
    assert not directory.exists()


@pytest.mark.parametrize(
    "key,enabled,paused,expected",
    [
        (False, "auto", False, False),
        (True, "auto", False, True),
        (True, True, False, True),
        (True, False, False, False),
        (True, "auto", True, False),
    ],
)
def test_pipeline_automatically_wires_observer(
    directory, tmp_path, monkeypatch, key, enabled, paused, expected
):
    from crossword_generator.config import load_config
    from crossword_generator.pipeline import create_pipeline
    from crossword_generator.steps.clue_grading_step import ClueWithGradingStep

    if key:
        monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    else:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    if paused:
        study.update_study(decision="retire", notes="Testing paused collection")
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.easy.yaml")
    cfg.output.directory = str(tmp_path / "generation")
    cfg.grading.clue.content_shadow.enabled = enabled
    cfg.llm.provider = "claude"
    monkeypatch.setattr("crossword_generator.pipeline.ClaudeProvider", MagicMock())
    pipeline, _ = create_pipeline(cfg)
    step = next(s for s in pipeline.steps if isinstance(s, ClueWithGradingStep))
    assert (step._content_shadow is not None) is expected


def test_pilot_norm_automatically_enters_queue(directory, tmp_path):
    original = (
        Path(__file__).resolve().parents[1]
        / "evals/classification/content-policy-v2/shadow-pilot-2026-09-22"
    )
    copied = tmp_path / "pilot"
    shutil.copytree(original, copied)
    result = study.update_study(copied)
    assert result["final_clues"] == 40
    with (directory / "review.csv").open() as f:
        rows = list(csv.DictReader(f))
    norm = next(r for r in rows if r["answer"] == "NORM")
    assert "borderline" in norm["review_reason"]
    assert norm["editor_verdict"] == ""
    assert "Agent triage" in norm["editor_notes"]


def test_large_batch_checkpoints_without_serializing_live_objects(tmp_path):
    from crossword_generator.cli import _checkpoint_batch_progress

    results = [{"seed": 1, "success": True, "_sweep": {"step": object()}}]
    _checkpoint_batch_progress(tmp_path, "large-batch", results, 500)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["completed"] == 1 and progress["planned"] == 500
    assert "duplicate sweep pending" in progress["phase"]
    assert progress["results"] == [{"seed": 1, "success": True}]
    assert "_sweep" in results[0]
    _checkpoint_batch_progress(tmp_path, "large-batch", results, 500, phase="complete")
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "complete"
