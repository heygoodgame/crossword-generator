"""Checks that evaluation cannot silently leak labels or inflate success metrics."""

from __future__ import annotations

import copy
import json

import pytest

from crossword_generator.classification_eval.core import (
    import_legacy,
    prepare,
    run_experiment,
)
from crossword_generator.classification_eval.import_logs import extract_grading_record
from crossword_generator.classification_eval.providers import validate_probabilities
from crossword_generator.classification_eval.report import (
    archive,
    binary_metrics,
    compare,
    rescore,
    summarize,
)

SUITE = {
    "id": "test-v1",
    "state_fields": ["answer", "clue", "hint"],
    "questions": {
        "bad": {
            "required_fields": ["answer", "clue"],
            "question": {"type": "noul", "instructions": "Bad clue?"},
        },
        "hint": {
            "required_fields": ["hint"],
            "question": {"type": "noul", "instructions": "Useful hint?"},
        },
    },
}
MODEL = {"provider": "test", "model": "test-1"}


def corpus():
    return [
        {
            "id": "a",
            "state": {"answer": "CAT", "clue": "Pet"},
            "labels": {"bad": True},
            "metadata": {"label_source": "test", "reference": {"feedback": "secret"}},
        },
        {"id": "b", "state": {"answer": "DOG", "clue": "Pet"}, "labels": {}},
    ]


class FakeAdapter:
    def __init__(self, value=0.8, fail=False):
        self.value, self.fail, self.calls = value, fail, []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("secret-api-key must not appear in saved errors")
        return {
            "probabilities": {q: self.value for q in questions},
            "resolved_model": "test-1",
            "usage": {"input_tokens": 10},
        }


def test_state_allowlist_and_optional_questions():
    row = corpus()[0]
    row["state"]["feedback"] = "answer label"
    state, questions = prepare(row, SUITE)
    assert state == {"answer": "CAT", "clue": "Pet"}
    assert set(questions) == {"bad"}
    row["state"]["hint"] = "Meowing pet"
    assert set(prepare(row, SUITE)[1]) == {"bad", "hint"}


@pytest.mark.parametrize(
    "value", [None, True, "0.5", -0.1, 1.1, float("nan"), float("inf")]
)
def test_invalid_probabilities_rejected(value):
    with pytest.raises(ValueError):
        validate_probabilities({"q": value}, {"q": {}})


def test_missing_answer_rejected():
    with pytest.raises(ValueError):
        validate_probabilities({}, {"q": {}})


def test_resume_and_changed_input_refusal(tmp_path):
    adapter = FakeAdapter()
    run_experiment(corpus(), SUITE, MODEL, tmp_path, adapter=adapter, repeats=2)
    assert len(adapter.calls) == 4
    run_experiment(corpus(), SUITE, MODEL, tmp_path, adapter=adapter, repeats=2)
    assert len(adapter.calls) == 4
    changed = copy.deepcopy(SUITE)
    changed["questions"]["bad"]["question"]["instructions"] = "Different question"
    with pytest.raises(ValueError, match="Inputs"):
        run_experiment(corpus(), changed, MODEL, tmp_path, adapter=adapter, repeats=2)
    assert not (tmp_path / ".running").exists()


def test_error_redaction_and_failed_requests_can_resume(tmp_path):
    run_experiment(corpus(), SUITE, MODEL, tmp_path, adapter=FakeAdapter(fail=True))
    assert "secret-api-key" not in (tmp_path / "results.jsonl").read_text()
    assert summarize(tmp_path)["errors"] == 2
    adapter = FakeAdapter()
    run_experiment(corpus(), SUITE, MODEL, tmp_path, adapter=adapter)
    assert len(adapter.calls) == 2
    summary = summarize(tmp_path)
    assert summary["errors"] == 0
    assert summary["successful_requests"] == 2


def test_unknown_labels_and_repeats_not_counted_as_negatives(tmp_path):
    run_experiment(corpus(), SUITE, MODEL, tmp_path, adapter=FakeAdapter(), repeats=3)
    summary = summarize(tmp_path)
    assert summary["successful_requests"] == 6
    metrics = summary["questions"]["bad"]["metrics"]
    assert metrics["n"] == 1
    assert metrics["negative"] == 0
    assert metrics["fpr"] is None
    assert metrics["auc"] is None
    assert summary["questions"]["hint"]["scored_cases"] == 0


def test_metric_ties_and_zero_denominators():
    m = binary_metrics([(0.5, True), (0.5, False)], 0.5)
    assert m["auc"] == 0.5
    assert m["tp"] == 1 and m["fp"] == 1
    assert m["brier"] == 0.25
    assert binary_metrics([], 0.5)["accuracy"] is None
    assert binary_metrics([(0.9, True), (0.1, False)], 0.5)["auc"] == 1


def test_archive_preserves_attempts_metrics_and_comparison(tmp_path):
    source, frozen = tmp_path / "live", tmp_path / "frozen"
    run_experiment(corpus(), SUITE, MODEL, source, adapter=FakeAdapter(fail=True))
    run_experiment(corpus(), SUITE, MODEL, source, adapter=FakeAdapter())
    expected = summarize(source)
    assert archive(source, frozen) == expected
    assert len(json.loads((frozen / "results.json").read_text())) == 4
    result = compare(source, frozen, tmp_path / "comparison.json")
    assert result["decision_changes"] == 0
    assert result["paired_predictions"] == 2
    with pytest.raises(ValueError, match="Archived"):
        run_experiment(corpus(), SUITE, MODEL, frozen, adapter=FakeAdapter())
    with pytest.raises(FileExistsError):
        archive(source, frozen)


def test_paired_comparison_and_suite_guard(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    run_experiment(corpus(), SUITE, MODEL, left, adapter=FakeAdapter(0.8))
    run_experiment(corpus(), SUITE, MODEL, right, adapter=FakeAdapter(0.2))
    result = compare(left, right, tmp_path / "comparison.json")
    assert result["paired_predictions"] == result["decision_changes"] == 2
    manifest = json.loads((right / "manifest.json").read_text())
    manifest["suite"]["id"] = "changed"
    (right / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identical"):
        compare(left, right, tmp_path / "comparison.json")


def test_legacy_import_preserves_bad_labels_only_as_references(tmp_path):
    source, output = tmp_path / "old.json", tmp_path / "new.json"
    source.write_text(
        json.dumps(
            [
                {
                    "answer": "RENEW",
                    "clue": "Keep a subscription going",
                    "opus_defect": True,
                    "jev": {"leaks_answer": 0.9},
                }
            ]
        )
    )
    rows = import_legacy(source, output)
    assert rows[0]["labels"] == {}
    assert rows[0]["metadata"]["reference"]["opus_defect"] is True
    assert "opus_defect" not in rows[0]["state"]


def test_log_import_handles_quotes_and_preserves_source():
    record = {
        "context": {"step": "clue_grading", "difficulty": "easy"},
        "request": {
            "prompt": '- 3-DOWN: Answer=RYAN, Clue=""Saving Private ___"" '
            "(crossing words: CAT, DOG)\n"
        },
        "response": {
            "text": '```json\n[{"number":3,"direction":"down",'
            '"accuracy":24,"feedback":"fine"}]\n```'
        },
        "provider": "claude",
        "model": "historical-model",
    }
    rows = extract_grading_record(record, "source:3")
    assert rows[0]["state"]["clue"] == '"Saving Private ___"'
    assert rows[0]["state"]["crossing_answers"] == "CAT, DOG"
    assert rows[0]["labels"] == {}
    assert rows[0]["metadata"]["references"][0]["source"] == "source:3"
    record["response"]["text"] = "[]"
    with pytest.raises(ValueError, match="omitted"):
        extract_grading_record(record, "source:3")


def test_rescore_reuses_predictions_and_rejects_changed_state(tmp_path):
    source = tmp_path / "run"
    run_experiment(corpus(), SUITE, MODEL, source, adapter=FakeAdapter())
    reviewed = corpus()[:1]
    reviewed[0]["labels"]["bad"] = False
    labels = tmp_path / "reviewed.json"
    labels.write_text(json.dumps(reviewed))
    summary = rescore(source, labels, tmp_path / "review")
    assert summary["questions"]["bad"]["metrics"]["fp"] == 1
    assert summary["cases"] == 1
    assert json.loads((source / "corpus.json").read_text())[0]["labels"]["bad"]
    reviewed[0]["state"]["clue"] = "changed input"
    labels.write_text(json.dumps(reviewed))
    with pytest.raises(ValueError, match="unchanged"):
        rescore(source, labels, tmp_path / "bad-review")


def test_typesafe_adapter_request_and_native_response(monkeypatch):
    from crossword_generator.classification_eval.providers import TypeSafeAdapter

    monkeypatch.setenv("TEST_TYPESAFE_KEY", "test-only-key")
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "resolved",
                    "usage": {"input_tokens": 5},
                    "answers": {"bad": {"type": "noul", "noul": 0.9}},
                }
            ).encode()

    def post(request, timeout):
        sent.append(json.loads(request.data))
        assert request.get_header("Authorization") == "Bearer test-only-key"
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", post)
    adapter = TypeSafeAdapter(
        {"model": "requested", "api_key_env": "TEST_TYPESAFE_KEY"}
    )
    state, questions = prepare(corpus()[0], SUITE)
    result = adapter.evaluate(state, questions)
    assert sent == [{"model": "requested", "state": state, "questions": questions}]
    assert "test-only-key" not in json.dumps(result)
    assert result["resolved_model"] == "resolved"
    assert result["probabilities"] == {"bad": 0.9}
