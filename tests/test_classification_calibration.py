import copy
import json

import pytest

from crossword_generator.classification_eval.calibration import (
    evaluate_thresholds,
    fit_thresholds,
)
from crossword_generator.classification_eval.core import run_experiment

SUITE = {
    "id": "policy-test",
    "state_fields": ["answer", "clue"],
    "questions": {
        "q": {
            "required_fields": ["clue"],
            "question": {"type": "noul", "instructions": "Q?"},
        }
    },
}
MODEL = {"provider": "fake", "model": "fake"}


class Adapter:
    def evaluate(self, state, questions):
        return {"probabilities": {"q": float(state["clue"])}, "usage": {}}


def rows(split):
    return [
        {
            "id": f"{split}-{i}",
            "state": {"answer": str(i), "clue": p},
            "labels": {"q": label},
            "metadata": {"split": split, "group": f"{split}-{i}"},
        }
        for i, (p, label) in enumerate([("0.8", True), ("0.6", True), ("0.4", False)])
    ]


def test_fit_uses_only_dev_and_frozen_threshold_on_test(tmp_path):
    dev, test = tmp_path / "dev", tmp_path / "test"
    run_experiment(rows("dev"), SUITE, MODEL, dev, adapter=Adapter())
    run_experiment(rows("test"), SUITE, MODEL, test, adapter=Adapter())
    selection = fit_thresholds({"context": dev})
    assert selection["questions"]["q"]["threshold"] == 0.6
    result = evaluate_thresholds(selection, {"context": test})
    assert result["questions"]["q"]["metrics"]["recall"] == 1
    with pytest.raises(ValueError, match="only dev"):
        fit_thresholds({"context": test})
    changed = copy.deepcopy(selection)
    changed["development_groups"].append("test-0")
    with pytest.raises(ValueError, match="overlap"):
        evaluate_thresholds(changed, {"context": test})
    changed = copy.deepcopy(selection)
    changed["candidates"]["context"]["suite_sha256"] = "different"
    with pytest.raises(ValueError, match="configuration"):
        evaluate_thresholds(changed, {"context": test})


def test_incomplete_predictions_cannot_calibrate(tmp_path):
    run_experiment(rows("dev"), SUITE, MODEL, tmp_path, adapter=Adapter())
    path = tmp_path / "results.jsonl"
    results = [json.loads(s) for s in path.read_text().splitlines()]
    results[0]["error"] = "failed"
    path.write_text("\n".join(json.dumps(r) for r in results))
    with pytest.raises(ValueError, match="complete"):
        fit_thresholds({"context": tmp_path})


def test_fixture_groups_and_state_ablation():
    from pathlib import Path

    base = (
        Path(__file__).resolve().parents[1] / "evals/classification/content-policy-v2"
    )
    dev, test = [json.loads((base / f"{s}.json").read_text()) for s in ("dev", "test")]
    assert len(dev) + len(test) == 200
    assert not (
        {r["metadata"]["group"] for r in dev} & {r["metadata"]["group"] for r in test}
    )
    context, clue = [
        json.loads((base / f"suite-{s}.json").read_text())
        for s in ("context", "clue-only")
    ]
    assert context["state_fields"] == ["answer", "clue"]
    assert clue["state_fields"] == ["clue"]
    assert {k: v["question"] for k, v in context["questions"].items()} == {
        k: v["question"] for k, v in clue["questions"].items()
    }
