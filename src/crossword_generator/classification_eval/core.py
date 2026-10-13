"""Corpus import, immutable experiment manifests, and resumable evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .providers import InvalidResponseError, build_adapter, validate_probabilities


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Corpus must be a JSON array or JSONL objects")
    return rows


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def import_legacy(path: Path, output: Path) -> list[dict]:
    rows = []
    for original in read_rows(path):
        state = {
            "answer": original["answer"],
            "clue": original["clue"],
            "puzzle_difficulty": original.get("difficulty"),
            "puzzle_type": original.get("puzzle_type"),
            "crossing_answers": original.get("crossing", ""),
        }
        rows.append(
            {
                "id": digest(state)[:20],
                "state": state,
                "labels": {},
                "metadata": {
                    "source": str(path),
                    "source_sha256": digest(original),
                    "label_source": "historical-model-unverified",
                    "reference": original,
                },
            }
        )
    # Repeated instances stay together and do not inflate per-clue metrics.
    unique = {}
    for row in rows:
        if row["id"] in unique:
            if (
                row["metadata"]["reference"]
                != unique[row["id"]]["metadata"]["reference"]
            ):
                raise ValueError(
                    "Conflicting duplicate historical rows; resolve explicitly"
                )
        unique[row["id"]] = row
    result = list(unique.values())
    write_json(output, result)
    return result


def validate_inputs(rows: list[dict], suite: dict, model: dict) -> None:
    if not rows:
        raise ValueError("Empty corpus")
    if (
        not suite.get("id")
        or not suite.get("questions")
        or not suite.get("state_fields")
    ):
        raise ValueError(
            "Suite requires id, questions, and an explicit state_fields allowlist"
        )
    if not model.get("model") or not model.get("provider"):
        raise ValueError("Model config requires provider and model")
    if any(key in model for key in ("api_key", "token", "authorization")):
        raise ValueError(
            "Credentials belong in environment variables, never model configs"
        )
    ids = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not isinstance(row.get("state"), dict):
            raise ValueError("Every corpus row requires a string id and state object")
        if row["id"] in ids:
            raise ValueError(f"Duplicate corpus ID {row['id']}")
        ids.add(row["id"])
        for label in row.get("labels", {}).values():
            if label is not None and not isinstance(label, bool):
                raise ValueError("Labels must be boolean or null (unknown)")
    for spec in suite["questions"].values():
        if spec["question"].get("type") != "noul":
            raise ValueError(
                "This evaluator supports binary probability questions only"
            )
        if not set(spec.get("required_fields", [])).issubset(suite["state_fields"]):
            raise ValueError("Question requires a field outside the state allowlist")


def prepare(row: dict, suite: dict) -> tuple[dict, dict]:
    state = {
        key: row["state"][key] for key in suite["state_fields"] if key in row["state"]
    }
    questions = {
        key: spec["question"]
        for key, spec in suite["questions"].items()
        if all(
            state.get(field) is not None and state.get(field) != ""
            for field in spec.get("required_fields", [])
        )
    }
    return state, questions


def run_experiment(
    rows: list[dict],
    suite: dict,
    model: dict,
    output: Path,
    *,
    workers: int = 4,
    repeats: int = 1,
    adapter=None,
) -> list[dict]:
    validate_inputs(rows, suite, model)
    if workers < 1 or repeats < 1:
        raise ValueError("workers and repeats must be positive")
    # Frozen inputs include labels and metadata, but only prepare() output reaches API.
    implementation = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    config = {
        "schema_version": 1,
        "corpus_sha256": digest(rows),
        "suite": suite,
        "model": model,
        "repeats": repeats,
        "implementation": implementation,
    }
    run_hash = digest(config)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two processes appending to the same run.
    lock = output / ".running"
    with lock.open("x"):
        pass
    try:
        if (output / "results.json").exists():
            raise ValueError("Archived runs are read-only; choose a new run directory")
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["run_hash"] != run_hash:
                raise ValueError(
                    "Inputs or evaluator changed; use a new output directory"
                )
        else:
            if (output / "results.jsonl").exists():
                raise ValueError(
                    "Results without a manifest; use a new output directory"
                )
            write_json(
                manifest_path,
                config
                | {"run_hash": run_hash, "created_at": datetime.now(UTC).isoformat()},
            )
            write_json(output / "corpus.json", rows)
        result_path = output / "results.jsonl"
        previous = read_rows(result_path) if result_path.exists() else []
        done = {(r["case_id"], r["repeat"]) for r in previous if not r.get("error")}
        jobs = [
            (row, repeat)
            for row in rows
            for repeat in range(repeats)
            if (row["id"], repeat) not in done
        ]
        if not jobs:
            return previous
        adapter = adapter or build_adapter(model)

        def evaluate(row, repeat):
            state, questions = prepare(row, suite)
            record = {
                "case_id": row["id"],
                "repeat": repeat,
                "request": {"state": state, "questions": questions},
                "started_at": datetime.now(UTC).isoformat(),
            }
            started = time.monotonic()
            if not questions:
                return record | {
                    "probabilities": {},
                    "skipped": True,
                    "latency_s": 0,
                    "error": None,
                }
            try:
                response = adapter.evaluate(state, questions)
                response["probabilities"] = validate_probabilities(
                    response["probabilities"], questions
                )
                record.update(response)
                record["error"] = None
            except Exception as exc:
                if isinstance(exc, InvalidResponseError):
                    record.update(exc.result)
                # Provider exceptions can embed request headers; never log their text.
                record["error"] = type(exc).__name__
                if isinstance(exc, RuntimeError) and str(exc).startswith(
                    "TypeSafe HTTP "
                ):
                    record["error"] = str(exc)
            record["latency_s"] = round(time.monotonic() - started, 4)
            return record

        with result_path.open("a") as handle, ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(evaluate, row, repeat) for row, repeat in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                previous.append(record)
                if index % 25 == 0 or index == len(jobs):
                    errors = sum(bool(r.get("error")) for r in previous)
                    print(
                        f"{index}/{len(jobs)} requests completed; "
                        f"{errors} recorded errors",
                        flush=True,
                    )
        return previous
    finally:
        lock.unlink()
