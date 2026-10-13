import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

from crossword_generator.classification_eval.core import digest
from crossword_generator.config import Config
from crossword_generator.content_shadow import (
    ContentShadow,
    load_protocol,
    report_batch,
)
from crossword_generator.models import (
    ClueEntry,
    ClueGrade,
    ClueGradeReport,
    FillResult,
    PuzzleEnvelope,
)
from crossword_generator.steps.clue_grading_step import ClueWithGradingStep


class Adapter:
    def __init__(self, fail=False, model="jev-1.13.0"):
        self.calls, self.fail, self.model = [], fail, model

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("SECRET credential in simulated provider error")
        return {
            "probabilities": dict.fromkeys(questions, 0.99),
            "resolved_model": self.model,
            "usage": {"input_tokens": 10},
        }


def envelope():
    return PuzzleEnvelope(
        fill=FillResult(grid=[["A"]], filler_used="test"),
        clues=[
            ClueEntry(
                number=1,
                direction="across",
                answer="BOURBON",
                clue="Whiskey associated with Kentucky",
            )
        ],
        clue_grade_report=ClueGradeReport(
            overall_score=96,
            clue_count=1,
            passing=True,
            clue_grades=[
                ClueGrade(
                    number=1,
                    direction="across",
                    answer="BOURBON",
                    score=96,
                    fairness=24,
                    accuracy=24,
                    craft=24,
                    freshness=24,
                )
            ],
        ),
    )


def test_protocol_is_exactly_the_evaluated_suites_and_thresholds():
    p = load_protocol()
    base = (
        Path(__file__).resolve().parents[1] / "evals/classification/content-policy-v2"
    )
    assert p["selection"] == json.loads((base / "selection.json").read_text())
    for name, suite in p["suites"].items():
        assert digest(suite) == p["selection"]["candidates"][name]["suite_sha256"]
        assert digest(p["model"]) == p["selection"]["candidates"][name]["model_sha256"]
    assert Config().grading.clue.content_shadow.enabled is False


def test_observation_caches_requests_and_does_not_mutate(tmp_path):
    api, e = Adapter(), envelope()
    before = e.model_dump()
    shadow = ContentShadow(tmp_path / "content-shadow.json", adapter=api)
    shadow.observe(e, phase="generated", quality_flags=set())
    shadow.observe(e, phase="detector-sweep", quality_flags=set(), findings={})
    shadow.observe(e, phase="final", quality_flags=set())
    assert e.model_dump() == before
    assert len(api.calls) == 2
    assert {tuple(s) for s, _ in api.calls} == {("answer", "clue"), ("clue",)}
    summary = report_batch(tmp_path)
    assert summary["unique_clue_revisions"] == summary["final_clues"] == 1
    assert summary["additional_to_grader"] == 1
    assert summary["additional_to_detector_sweep"] == 1
    changed = e.model_copy(deep=True)
    changed.clues[0].clue = "French royal house"
    shadow.observe(changed, phase="post-duplicate-sweep", quality_flags=set())
    assert len(api.calls) == 4
    assert report_batch(tmp_path)["final_clues"] == 1


def test_missing_predictions_stay_unknown_and_errors_are_redacted(tmp_path):
    shadow = ContentShadow(tmp_path / "content-shadow.json", adapter=Adapter(fail=True))
    shadow.observe(envelope(), phase="final", quality_flags=set())
    assert "SECRET" not in shadow.path.read_text()
    entry = shadow.data["observations"][0]["entries"][0]
    assert entry["flagged"] is None
    summary = report_batch(tmp_path)
    assert summary["request_errors"] == 2
    assert summary["incomplete_revisions"] == 1
    assert summary["additional_to_grader"] == 0


def test_model_drift_rejected_and_missing_grade_not_assumed_clean(tmp_path):
    shadow = ContentShadow(
        tmp_path / "content-shadow.json", adapter=Adapter(model="jev-new")
    )
    e = envelope().model_copy(update={"clue_grade_report": None})
    shadow.observe(e, phase="final", quality_flags=set())
    assert report_batch(tmp_path)["request_errors"] == 2
    assert shadow.data["observations"][0]["entries"][0]["grader_would_repair"] is None


def test_editor_notes_survive_report_regeneration(tmp_path):
    shadow = ContentShadow(tmp_path / "content-shadow.json", adapter=Adapter())
    shadow.observe(envelope(), phase="final", quality_flags=set())
    report_batch(tmp_path)
    path = tmp_path / "content-shadow-review.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    rows[0]["editor_verdict"] = "confirmed"
    rows[0]["editor_notes"] = "Reviewed by editor"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_batch(tmp_path)
    assert "Reviewed by editor" in path.read_text()


def test_report_uses_manifest_cohort_and_reports_missing_artifact(tmp_path):
    stale = ContentShadow(tmp_path / "stale/content-shadow.json", adapter=Adapter())
    stale.observe(envelope(), phase="final", quality_flags=set())
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "output_root": "original/batch",
                "results": [
                    {"content_shadow_path": "original/batch/new/content-shadow.json"}
                ],
            }
        )
    )
    summary = report_batch(tmp_path)
    assert summary["artifacts"] == 0
    assert summary["expected_artifacts"] == 1
    assert len(summary["missing_artifacts"]) == 1
    assert summary["unique_clue_revisions"] == 0


def test_shadow_flags_never_enter_repairs_or_change_generation(tmp_path, caplog):
    e = envelope()
    results = []
    observers = [
        None,
        ContentShadow(tmp_path / "content-shadow.json", adapter=Adapter()),
        MagicMock(),
    ]
    observers[-1].observe.side_effect = RuntimeError("SECRET")
    for observer in observers:
        grader = MagicMock()
        grader.grade.return_value = e.clue_grade_report
        step = ClueWithGradingStep(MagicMock(), grader, content_shadow=observer)
        step._clue_step.run = lambda _: e.model_copy(deep=True)
        step._repair_entries = MagicMock(
            side_effect=AssertionError("No repairs expected")
        )
        results.append(step.run(e.model_copy(update={"clues": []})).model_dump())
        step._repair_entries.assert_not_called()
    assert results[0] == results[1] == results[2]
    assert "SECRET" not in caplog.text


def test_post_batch_duplicate_repair_updates_final_shadow_revision(tmp_path):
    from crossword_generator.clue_history import ClueHistoryIndex

    e = envelope()
    history = ClueHistoryIndex()
    history.add_clues(e.clues)
    shadow = ContentShadow(tmp_path / "content-shadow.json", adapter=Adapter())
    shadow.observe(e, phase="final", quality_flags=set())
    original_ids = shadow.data["final_entry_ids"][:]
    updated = e.model_copy(deep=True)
    updated.clues[0].clue = "French royal house"
    step = ClueWithGradingStep(
        MagicMock(), MagicMock(), content_shadow=shadow, clue_history=history
    )
    step._run_clue_repair_once = MagicMock(return_value=updated)
    result = step.repair_external_duplicates(e, history.find_duplicates(e.clues))
    assert result is updated
    assert shadow.data["observations"][-1]["phase"] == "post-duplicate-sweep"
    assert shadow.data["final_entry_ids"] != original_ids
    assert report_batch(tmp_path)["final_clues"] == 1
