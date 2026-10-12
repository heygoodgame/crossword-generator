"""Regression coverage for long batches and unpublished continuations."""

import json
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from crossword_generator import cli
from crossword_generator.config import ClaudeConfig
from crossword_generator.exporters.ipuz_exporter import IpuzExporter
from crossword_generator.llm.claude_provider import ClaudeProvider
from crossword_generator.models import FillResult, PuzzleEnvelope
from crossword_generator.pipeline import Pipeline
from tests.test_cli_batch import _StubExporter, _sweep_envelope, _sweep_result


@pytest.mark.parametrize("fails", [False, True])
def test_retained_pipelines_release_sockets_and_can_repair(
    tmp_path, monkeypatch, fails
):
    clients = []

    class Client:
        def __init__(self, **kwargs):
            self.ends = socket.socketpair()
            self.messages = SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    content=[{"type": "text", "text": "ok"}]
                )
            )
            clients.append(self)

        def close(self):
            for end in self.ends:
                end.close()

    monkeypatch.setattr("anthropic.Anthropic", Client)
    retained = []
    for _ in range(100):
        provider = ClaudeProvider(ClaudeConfig())
        step = MagicMock()
        step.name = "test"

        def run(envelope):
            assert provider.generate("generate") == "ok"
            if fails:
                raise RuntimeError("pipeline failed")
            return envelope

        step.run.side_effect = run
        pipeline = Pipeline([step], [], tmp_path, providers=[provider, provider])
        if fails:
            with pytest.raises(RuntimeError, match="pipeline failed"):
                pipeline.run(PuzzleEnvelope())
        else:
            pipeline.run(PuzzleEnvelope())
        retained.append(pipeline)
        assert all(end.fileno() == -1 for c in clients for end in c.ends)
    # The sweep can use the retained provider, which opens a fresh client.
    first = retained[0].providers[0]
    assert first.generate("repair") == "ok"
    first.close()
    first.close()
    assert len(clients) == 101
    assert all(end.fileno() == -1 for c in clients for end in c.ends)


def test_unlimited_continuation_keeps_counts_clues_and_fill_pool(tmp_path, monkeypatch):
    puzzle = tmp_path / "prior.ipuz"
    envelope = _sweep_envelope("Feline pet")
    envelope.fill = FillResult(
        grid=[list("CAT"), list("###"), list("###")], filler_used="test"
    )
    IpuzExporter().export_to_file(envelope, puzzle)
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "difficulty": "easy",
                        "size": 5,
                        "seed": 1,
                        "success": True,
                        "output_path": str(puzzle),
                    }
                ]
            }
        )
    )
    usage = cli._AnswerUsageCounter()
    usage.add(["CAT"], records=1)
    monkeypatch.setattr(
        cli,
        "_load_unlimited_answer_usage_by_bucket",
        lambda *a, **kw: {("easy", 5): usage},
    )
    monkeypatch.setattr(cli, "_load_existing_clue_history", lambda *a, **kw: 0)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs["excluded_fill_words"] == set()
        assert kwargs["answer_usage_counts"]["CAT"] == len(calls) + 1
        assert "Feline pet" in kwargs["clue_history"].clues_for_answer("CAT")
        return {
            "difficulty": "easy",
            "size": 5,
            "seed": kwargs["seed"],
            "success": True,
            "output_path": str(puzzle),
            "runtime_seconds": 0,
            "clue_score": 90,
        }

    monkeypatch.setattr(cli, "_run_batch_item", generate)
    result = CliRunner().invoke(
        cli.main,
        [
            "generate-pilot-batch",
            "--output-root",
            str(tmp_path / "batch"),
            "--batch-id",
            "continuation",
            "--buckets",
            "easy/5",
            "--count",
            "2",
            "--seed-start",
            "10",
            "--max-workers",
            "1",
            "--no-refresh-dictionaries",
            "--no-intra-batch-dedup",
            "--no-exclude-recent-answers",
            "--no-exclude-scheduled-sixty",
            "--no-exclude-existing-boards",
            "--prior-batch-manifest",
            str(prior),
            "--prior-batch-manifest",
            str(prior),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    manifest = json.loads((tmp_path / "batch/manifest.json").read_text())
    assert manifest["prior_batch_manifests"]["puzzles_seeded"] == 1


def test_progress_checkpoint_survives_sweep_crash_and_resumes(tmp_path, monkeypatch):
    template = tmp_path / "template.ipuz"
    envelope = _sweep_envelope("Feline pet")
    envelope.fill = FillResult(
        grid=[list("CAT"), list("###"), list("###")], filler_used="test"
    )
    IpuzExporter().export_to_file(envelope, template)
    monkeypatch.setattr(
        cli,
        "_load_unlimited_answer_usage_by_bucket",
        lambda *a, **kw: {("easy", 5): cli._AnswerUsageCounter()},
    )
    monkeypatch.setattr(cli, "_load_existing_clue_history", lambda *a, **kw: 0)
    completed_before_each_puzzle = []

    def generate(**kwargs):
        progress = kwargs["output_root"] / "progress.json"
        completed_before_each_puzzle.append(
            json.loads(progress.read_text())["completed"] if progress.exists() else 0
        )
        output = tmp_path / f"seed-{kwargs['seed']}.ipuz"
        output.write_text(template.read_text())
        return {
            "difficulty": "easy",
            "size": 5,
            "seed": kwargs["seed"],
            "success": True,
            "output_path": str(output),
            "runtime_seconds": 0,
            "clue_score": 90,
            # Not JSON-serializable: the checkpoint must leave it out.
            "_sweep": {"envelope": object()},
        }

    monkeypatch.setattr(cli, "_run_batch_item", generate)

    def run_batch(name, *extra):
        return CliRunner().invoke(
            cli.main,
            [
                "generate-pilot-batch",
                "--output-root",
                str(tmp_path / name),
                "--batch-id",
                name,
                "--buckets",
                "easy/5",
                "--max-workers",
                "1",
                "--no-refresh-dictionaries",
                "--no-intra-batch-dedup",
                "--no-exclude-recent-answers",
                "--no-exclude-scheduled-sixty",
                "--no-exclude-existing-boards",
                *extra,
            ],
        )

    with monkeypatch.context() as crash:
        crash.setattr(
            cli,
            "_run_duplicate_grid_sweep",
            MagicMock(side_effect=RuntimeError("sweep died")),
        )
        crashed = run_batch("crashed", "--count", "2", "--seed-start", "10")
    assert isinstance(crashed.exception, RuntimeError)
    assert completed_before_each_puzzle == [0, 1]
    assert not (tmp_path / "crashed/manifest.json").exists()
    progress_path = tmp_path / "crashed/progress.json"
    progress = json.loads(progress_path.read_text())
    assert progress["phase"] == "generating; duplicate sweep pending"
    assert (progress["planned"], progress["completed"]) == (2, 2)
    assert [r["seed"] for r in progress["results"]] == [10, 11]
    assert all("_sweep" not in r for r in progress["results"])

    resumed = run_batch(
        "resumed",
        "--count",
        "1",
        "--seed-start",
        "12",
        "--prior-batch-manifest",
        str(progress_path),
    )
    assert resumed.exit_code == 0, resumed.output
    manifest = json.loads((tmp_path / "resumed/manifest.json").read_text())
    assert manifest["prior_batch_manifests"]["puzzles_seeded"] == 2
    finished = json.loads((tmp_path / "resumed/progress.json").read_text())
    assert (finished["phase"], finished["completed"]) == ("complete", 1)


def test_duplicate_sweep_closes_reopened_clients_after_repair_failure(tmp_path):
    from crossword_generator.clue_history import ClueHistoryIndex

    provider = MagicMock()
    step = MagicMock()
    step.repair_external_duplicates.side_effect = RuntimeError("repair failed")
    results = [
        _sweep_result(i, _sweep_envelope("Feline pet"), step, tmp_path / f"{i}.ipuz")
        for i in (1, 2)
    ]
    results[1]["_sweep"]["providers"] = [provider]
    stats = cli._run_duplicate_sweep(
        results, ClueHistoryIndex(), exporter=_StubExporter()
    )
    assert stats["unresolved"] == 1
    provider.close.assert_called_once()
