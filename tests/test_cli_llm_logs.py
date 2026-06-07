"""Tests for LLM log browsing and replay helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from crossword_generator.cli import main
from crossword_generator.llm.base import LLMCallResponse, LLMProvider
from crossword_generator.llm.log_browser import (
    LLMLogFilters,
    LLMLogRecord,
    LLMReplayRequest,
    filter_llm_log_records,
    load_llm_logs,
    replay_llm_record,
)


class MockReplayProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "claude"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> str:
        return self.generate_with_details(prompt, system=system, **kwargs).text

    def generate_with_details(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> LLMCallResponse:
        model = str(kwargs.get("model", "mock-model"))
        temperature = float(kwargs.get("temperature", 0.7))
        return LLMCallResponse(
            text=f"variant response for {prompt}",
            provider="claude",
            model=model,
            request={
                "prompt": prompt,
                "system": system,
                "kwargs": dict(kwargs),
                "temperature": temperature,
            },
            response={"text": f"variant response for {prompt}"},
            usage={"input_tokens": 11, "output_tokens": 7},
            cost={"estimated_cost_usd": 0.02, "currency": "USD"},
        )

    def is_available(self) -> bool:
        return True


def test_load_llm_logs_from_single_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "sample.llm.jsonl"
    _write_jsonl(log_path, [_sample_record(request_id="abc")])

    records, problems = load_llm_logs(log_path, project_root=tmp_path)

    assert problems == []
    assert len(records) == 1
    record = records[0]
    assert record.request_id == "abc"
    assert record.step == "clue_generation"
    assert record.difficulty == "easy"
    assert record.size == 5
    assert record.seed == 123
    assert record.prompt_text == "baseline prompt"
    assert record.system_text == "baseline system"
    assert record.response_text == "baseline response"


def test_load_llm_logs_from_manifest_and_reports_missing(
    tmp_path: Path,
) -> None:
    existing_log = tmp_path / "logs" / "one.llm.jsonl"
    missing_log = tmp_path / "logs" / "missing.llm.jsonl"
    _write_jsonl(existing_log, [_sample_record(request_id="one")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "results": [
                {"llm_log_path": str(existing_log)},
                {"llm_log_path": str(missing_log)},
            ]
        }),
        encoding="utf-8",
    )

    records, problems = load_llm_logs(manifest, project_root=tmp_path)

    assert [record.request_id for record in records] == ["one"]
    assert len(problems) == 1
    assert "missing log" in problems[0].message


def test_load_llm_logs_from_directory_and_malformed_line(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "logs" / "sample.llm.jsonl"
    log_path.parent.mkdir()
    log_path.write_text(
        json.dumps(_sample_record(request_id="ok")) + "\nnot json\n",
        encoding="utf-8",
    )

    records, problems = load_llm_logs(log_path.parent, project_root=tmp_path)

    assert [record.request_id for record in records] == ["ok"]
    assert len(problems) == 1
    assert problems[0].line_number == 2


def test_filter_llm_log_records() -> None:
    records = [
        _record_from_raw(_sample_record(request_id="one")),
        _record_from_raw(
            _sample_record(
                request_id="two",
                step="clue_grading",
                model="claude-haiku",
                seed=456,
                error={"type": "ValueError", "message": "bad response"},
            )
        ),
    ]

    assert [
        record.request_id
        for record in filter_llm_log_records(
            records,
            LLMLogFilters(step="clue_grading", errors_only=True, text="bad"),
        )
    ] == ["two"]
    assert filter_llm_log_records(records, LLMLogFilters(seed=999)) == []


def test_replay_llm_record_writes_artifacts(tmp_path: Path) -> None:
    record = _record_from_raw(_sample_record(request_id="replay-me"))

    artifact = replay_llm_record(
        record,
        LLMReplayRequest(
            model="claude-opus-test",
            temperature=0.2,
            prompt="variant prompt",
            system="variant system",
        ),
        experiment_root=tmp_path / "experiments",
        project_root=tmp_path,
        provider_factory=lambda _provider: MockReplayProvider(),
    )

    assert artifact.baseline_path.exists()
    assert artifact.variant_path.exists()
    assert artifact.comparison_path.exists()
    baseline = json.loads(artifact.baseline_path.read_text())
    variant = json.loads(artifact.variant_path.read_text())
    comparison = artifact.comparison_path.read_text()
    assert baseline["request_id"] == "replay-me"
    assert variant["model"] == "claude-opus-test"
    assert variant["request"]["prompt"] == "variant prompt"
    assert "Baseline model" in comparison
    assert "variant prompt" in comparison
    assert "baseline response" in comparison


def test_llm_logs_browse_cli_launches_with_resolved_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "sample.llm.jsonl"
    _write_jsonl(log_path, [_sample_record()])
    launched: dict[str, object] = {}

    def fake_launch(records, problems, *, initial_filters, experiment_root):
        launched["records"] = records
        launched["problems"] = problems
        launched["filters"] = initial_filters
        launched["experiment_root"] = experiment_root

    monkeypatch.setattr("crossword_generator.cli._launch_llm_log_browser", fake_launch)

    result = CliRunner().invoke(
        main,
        [
            "llm-logs",
            "browse",
            str(log_path),
            "--step",
            "clue_generation",
            "--experiment-root",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert len(launched["records"]) == 1
    assert launched["filters"].step == "clue_generation"
    assert launched["experiment_root"] == tmp_path / "out"


def test_llm_logs_browse_cli_rejects_invalid_target(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["llm-logs", "browse", str(tmp_path / "missing")],
    )

    assert result.exit_code == 1
    assert "Target does not exist" in result.stderr
    assert "No LLM log records found" in result.stderr


def _sample_record(
    *,
    request_id: str = "req-1",
    step: str = "clue_generation",
    model: str = "claude-sonnet",
    seed: int = 123,
    error: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "request_id": request_id,
        "started_at": "2026-06-07T00:00:00+00:00",
        "finished_at": "2026-06-07T00:00:01+00:00",
        "duration_seconds": 1.0,
        "context": {
            "step": step,
            "difficulty": "easy",
            "grid_size": 5,
            "seed": seed,
        },
        "provider": "claude",
        "model": model,
        "request": {
            "prompt": "baseline prompt",
            "system_text": "baseline system",
            "temperature": 0.7,
        },
        "response": {"text": "baseline response"},
        "usage": {"input_tokens": 5, "output_tokens": 3},
        "cost": {"estimated_cost_usd": 0.01, "currency": "USD"},
        "error": error,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record_from_raw(raw: dict[str, object]) -> LLMLogRecord:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        path = root / "sample.llm.jsonl"
        path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        records, _problems = load_llm_logs(path, project_root=root)
        return records[0]
