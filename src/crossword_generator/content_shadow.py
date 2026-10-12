"""Observational Jev content checks; no scores, repairs, or upload decisions.

Report a batch with: python -m crossword_generator.content_shadow BATCH_DIRECTORY
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

from crossword_generator.classification_eval.core import digest, prepare, write_json
from crossword_generator.classification_eval.providers import TypeSafeAdapter


def load_protocol() -> dict:
    return json.loads(
        files("crossword_generator.resources")
        .joinpath("content_policy_v2.json")
        .read_text()
    )


class ContentShadow:
    def __init__(self, path: Path, *, workers=4, timeout=10, adapter=None):
        self.path, self.workers = path, workers
        self.protocol = load_protocol()
        self._adapter, self._init_error = adapter, None
        self._timeout = timeout
        self.data = {
            "schema_version": 1,
            "mode": "shadow",
            "started_at": datetime.now(UTC).isoformat(),
            "protocol": self.protocol,
            "protocol_sha256": digest(self.protocol),
            "requests": {},
            "observations": [],
            "final_entry_ids": [],
        }

    def _evaluate(self, job):
        key, variant, state, questions = job
        started = time.monotonic()
        record = {
            "variant": variant,
            "state": state,
            "questions": questions,
            "started_at": datetime.now(UTC).isoformat(),
        }
        try:
            if self._init_error:
                record["error"] = self._init_error
            else:
                response = self._adapter.evaluate(state, questions)
                record.update(response)
                if response.get("resolved_model") != self.protocol["model"]["model"]:
                    record["error"] = "UnexpectedResolvedModel"
                else:
                    record["error"] = None
        except Exception as exc:
            # Exception messages may contain credentials/headers. Store class only.
            record["error"] = type(exc).__name__
        record["latency_s"] = round(time.monotonic() - started, 4)
        return key, record

    def observe(self, envelope, *, phase, quality_flags, findings=None):
        """Serialize observations without modifying the envelope or findings."""
        if self._adapter is None and self._init_error is None:
            try:
                self._adapter = TypeSafeAdapter(
                    self.protocol["model"] | {"timeout": self._timeout}
                )
            except Exception as exc:
                self._init_error = type(exc).__name__
        jobs, request_keys = {}, {}
        for clue in envelope.clues:
            state = {"answer": clue.answer, "clue": clue.clue}
            for variant, suite in self.protocol["suites"].items():
                prepared, questions = prepare({"state": state}, suite)
                key = digest([variant, prepared, questions])
                request_keys[(clue.number, clue.direction, variant)] = key
                if key not in self.data["requests"]:
                    jobs[key] = (key, variant, prepared, questions)
        if jobs:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for key, record in pool.map(self._evaluate, jobs.values()):
                    self.data["requests"][key] = record
        grades = {
            (g.number, g.direction): g.model_dump()
            for g in (
                envelope.clue_grade_report.clue_grades
                if envelope.clue_grade_report
                else []
            )
        }
        entries = []
        for clue in envelope.clues:
            position = (clue.number, clue.direction)
            state = {"answer": clue.answer, "clue": clue.clue}
            probabilities, flags = {}, []
            for question, choice in self.protocol["selection"]["questions"].items():
                key = request_keys[(*position, choice["variant"])]
                record = self.data["requests"][key]
                p = None if record.get("error") else record["probabilities"][question]
                probabilities[question] = p
                if p is not None and p >= choice["threshold"]:
                    flags.append(question)
            complete = all(p is not None for p in probabilities.values())
            entry_id = digest([position, state])[:24]
            entries.append(
                {
                    "id": entry_id,
                    "number": clue.number,
                    "direction": clue.direction,
                    "state": state,
                    "probabilities": probabilities,
                    "complete": complete,
                    "flagged_questions": flags,
                    "flagged": bool(flags) if complete or flags else None,
                    "grader_grade": grades.get(position),
                    "grader_would_repair": (
                        position in quality_flags if position in grades else None
                    ),
                    "existing_findings": findings.get(position, [])
                    if findings is not None
                    else None,
                    "request_ids": {
                        v: request_keys[(*position, v)] for v in self.protocol["suites"]
                    },
                }
            )
        self.data["observations"].append(
            {
                "phase": phase,
                "observed_at": datetime.now(UTC).isoformat(),
                "puzzle": {
                    "seed": envelope.metadata.get("seed"),
                    "difficulty": envelope.difficulty.value,
                    "grid_size": envelope.grid_size,
                },
                "entries": entries,
            }
        )
        if phase in {"final", "post-duplicate-sweep"}:
            self.data["final_entry_ids"] = [r["id"] for r in entries]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        write_json(temporary, self.data)
        temporary.replace(self.path)


def report_batch(root: Path, *, artifact_paths: list[Path] | None = None) -> dict:
    review, requests, incomplete_artifacts = [], [], 0
    paths = (
        artifact_paths
        if artifact_paths is not None
        else sorted(root.rglob("content-shadow.json"))
    )
    unsuccessful = set()
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and artifact_paths is None:
        manifest = json.loads(manifest_path.read_text())
        original_root = Path(manifest.get("output_root", str(root)))
        paths = []
        for item in manifest.get("results", []):
            if item.get("content_shadow_path"):
                path = Path(item["content_shadow_path"])
                try:
                    path = root / path.relative_to(original_root)
                except ValueError:
                    pass
                paths.append(path)
                if item.get("success") is False:
                    unsuccessful.add(path)
    missing, unreadable = [], []
    loaded_count = 0
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            unreadable.append(str(path))
            continue
        loaded_count += 1
        requests.extend(data["requests"].values())
        latest, detector_observations = {}, {}
        for event in data["observations"]:
            for entry in event["entries"]:
                latest[entry["id"]] = entry | {
                    "phase": event["phase"],
                    "puzzle": event["puzzle"],
                }
                if entry["existing_findings"] is not None:
                    detector_observations[entry["id"]] = entry["existing_findings"]
        final_ids = set(data["final_entry_ids"]) if path not in unsuccessful else set()
        if not final_ids:
            incomplete_artifacts += 1
        for entry_id, row in latest.items():
            findings = detector_observations.get(entry_id)
            review.append(
                {
                    "observation_id": digest(
                        [data["protocol_sha256"], data["started_at"], entry_id]
                    ),
                    "case_id": digest([data["protocol_sha256"], row["state"]]),
                    "protocol_sha256": data["protocol_sha256"],
                    "artifact": str(path),
                    "entry_id": entry_id,
                    "seed": row["puzzle"]["seed"],
                    "difficulty": row["puzzle"]["difficulty"],
                    "size": row["puzzle"]["grid_size"],
                    "number": row["number"],
                    "direction": row["direction"],
                    **row["state"],
                    "final": entry_id in final_ids,
                    "phase": row["phase"],
                    "complete": row["complete"],
                    "flagged": row["flagged"],
                    "flagged_questions": ",".join(row["flagged_questions"]),
                    **row["probabilities"],
                    "grader_would_repair": row["grader_would_repair"],
                    "additional_to_grader": row["flagged"] is True
                    and row["grader_would_repair"] is False,
                    "last_detector_findings": json.dumps(findings)
                    if findings is not None
                    else "",
                    "additional_to_detector_sweep": (
                        bool(row["flagged"]) and not findings
                        if findings is not None
                        else None
                    ),
                    "grade": json.dumps(row["grader_grade"]),
                    "editor_verdict": "",
                    "editor_notes": "",
                }
            )
    usage = {}
    for r in requests:
        for k, v in r.get("usage", {}).items():
            usage[k] = usage.get(k, 0) + v
    summary = {
        "mode": "shadow",
        "artifacts": loaded_count,
        "expected_artifacts": len(paths),
        "missing_artifacts": missing,
        "unreadable_artifacts": unreadable,
        "unfinished_artifacts": incomplete_artifacts,
        "unique_clue_revisions": len(review),
        "requests": len(requests),
        "request_errors": sum(bool(r.get("error")) for r in requests),
        "incomplete_revisions": sum(not r["complete"] for r in review),
        "flagged_revisions": sum(r["flagged"] is True for r in review),
        "additional_to_grader": sum(r["additional_to_grader"] for r in review),
        "additional_to_detector_sweep": sum(
            r["additional_to_detector_sweep"] is True for r in review
        ),
        "final_clues": sum(r["final"] for r in review),
        "complete_final_clues": sum(r["final"] and r["complete"] for r in review),
        "final_flagged_clues": sum(r["final"] and r["flagged"] is True for r in review),
        "usage": usage,
        "note": "Repeated observations use cached predictions. Revisions are not "
        "independent labeled samples. Marginal flags are review candidates, not proven "
        "violations; detector comparisons use the latest sweep of the same "
        "clue revision.",
    }
    write_json(root / "content-shadow-summary.json", summary)
    # Do not overwrite a file in which an editor has entered decisions.
    review_path = root / "content-shadow-review.csv"
    if review_path.exists():
        with review_path.open() as f:
            previous = list(csv.DictReader(f))
        decisions = {(r["artifact"], r["entry_id"]): r for r in previous}
        stable_decisions = {
            r["observation_id"]: r for r in previous if r.get("observation_id")
        }
        # Older reports have no observation ID. Preserve notes after an archive
        # moves only when its puzzle/revision identity has a unique match.
        legacy = {}

        def legacy_key(row):
            return tuple(
                str(row.get(k, ""))
                for k in ("seed", "size", "difficulty", "entry_id", "answer", "clue")
            )

        for old in previous:
            legacy.setdefault(legacy_key(old), []).append(old)
        for row in review:
            old = stable_decisions.get(row["observation_id"]) or decisions.get(
                (row["artifact"], row["entry_id"])
            )
            matches = legacy.get(legacy_key(row), [])
            if old is None:
                old = matches[0] if len(matches) == 1 else {}
            for key in ("editor_verdict", "editor_notes"):
                row[key] = old.get(key, "")
    if review:
        review.sort(
            key=lambda r: (not r["additional_to_grader"], not r["final"], r["answer"])
        )
        with review_path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(report_batch(args.directory), indent=2))


if __name__ == "__main__":
    main()
