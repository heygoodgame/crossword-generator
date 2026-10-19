"""Cumulative, bounded review workflow for the Jev shadow experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from crossword_generator.classification_eval.core import digest
from crossword_generator.config import find_project_root
from crossword_generator.content_shadow import load_protocol, report_batch

TARGET = 500
VERDICTS = {"violation", "acceptable", "ambiguous"}
# collection_status() reason for a run that never asked for observation.
DISABLED = "disabled"


def study_directory() -> Path:
    override = os.environ.get("CROSSWORD_SHADOW_STUDY_DIR")
    return (
        Path(override)
        if override
        else (find_project_root() / "output/content-shadow/content-policy-v2")
    )


def _count(db, protocol):
    return db.execute(
        "SELECT COUNT(DISTINCT case_id) FROM observations "
        "WHERE protocol=? AND final=1 AND complete=1",
        (protocol,),
    ).fetchone()[0]


def status(directory: Path | None = None) -> dict:
    directory = directory or study_directory()
    db_path = directory / "study.sqlite3"
    count, decision, checkpoint = 0, None, None
    if db_path.exists():
        db = sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=10
        )
        try:
            count = _count(db, digest(load_protocol()))
            meta = dict(db.execute("SELECT key, value FROM meta"))
            decision = meta.get("decision")
            checkpoint = meta.get("checkpoint_reached_at")
        finally:
            db.close()
    phase = decision or (
        "review_due" if checkpoint or count >= TARGET else "collecting"
    )
    return {
        "phase": phase,
        "final_clues": count,
        "target": TARGET,
        "queue_path": str(directory / "review.csv"),
        "status_path": str(directory / "status.json"),
    }


def collection_status(enabled="auto") -> tuple[bool, str]:
    if enabled is False:
        return False, DISABLED
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        return False, "no TYPESAFE_API_KEY"
    try:
        phase = status()["phase"]
    except (OSError, sqlite3.Error, ValueError) as exc:
        return False, f"tracking unavailable ({type(exc).__name__})"
    if phase != "collecting":
        return False, f"checkpoint paused: {phase}"
    return True, "collecting"


def _true(value):
    return value is True or value == "True"


def _sample(case_id):
    # Stable pseudo-random 5% sample, independent of scores and file ordering.
    return int(digest(["shadow-review-sample-v1", case_id])[:8], 16) / 2**32 < 0.05


def _connect(directory):
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "study.sqlite3", timeout=30)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL, protocol TEXT NOT NULL,
            final INTEGER NOT NULL, complete INTEGER NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews (
            case_id TEXT PRIMARY KEY, verdict TEXT NOT NULL, notes TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions (
            at TEXT NOT NULL, decision TEXT NOT NULL, notes TEXT NOT NULL);
    """)
    db.execute("BEGIN IMMEDIATE")
    return db


def _atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def update_study(
    batch: Path | None = None,
    directory: Path | None = None,
    *,
    decision: str | None = None,
    notes: str = "",
    artifact_paths: list[Path] | None = None,
) -> dict:
    """Ingest a batch idempotently; SQLite serializes concurrent batch updates."""
    if decision is not None and (
        decision not in {"revise", "retire", "ready-for-integration"}
        or not notes.strip()
    ):
        raise ValueError("A supported decision and nonempty review notes are required")
    directory = directory or study_directory()
    batch_summary, incoming = None, []
    if batch is not None:
        batch_summary = report_batch(batch, artifact_paths=artifact_paths)
        if batch_summary["unique_clue_revisions"]:
            with (batch / "content-shadow-review.csv").open() as f:
                incoming = list(csv.DictReader(f))
    protocol = load_protocol()
    fingerprint = digest(protocol)
    db = _connect(directory)
    try:
        queue_path = directory / "review.csv"
        if queue_path.exists():
            with queue_path.open() as f:
                for row in csv.DictReader(f):
                    db.execute(
                        "INSERT OR REPLACE INTO reviews VALUES (?,?,?)",
                        (row["case_id"], row["editor_verdict"], row["editor_notes"]),
                    )
        for row in incoming:
            db.execute(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?)",
                (
                    row["observation_id"],
                    row["case_id"],
                    row["protocol_sha256"],
                    int(_true(row["final"])),
                    int(_true(row["complete"])),
                    json.dumps(row),
                ),
            )
            # Carry initial per-batch editorial/triage notes into the central queue.
            if row["editor_notes"] or row["editor_verdict"]:
                db.execute(
                    "INSERT OR IGNORE INTO reviews VALUES (?,?,?)",
                    (row["case_id"], row["editor_verdict"], row["editor_notes"]),
                )
        if decision:
            db.execute("INSERT OR REPLACE INTO meta VALUES ('decision',?)", (decision,))
            db.execute(
                "INSERT INTO decisions VALUES (?,?,?)",
                (datetime.now(UTC).isoformat(), decision, notes),
            )
        count = _count(db, fingerprint)
        if count >= TARGET:
            db.execute(
                "INSERT OR IGNORE INTO meta VALUES ('checkpoint_reached_at',?)",
                (datetime.now(UTC).isoformat(),),
            )
        meta = dict(db.execute("SELECT key,value FROM meta"))
        phase = meta.get("decision") or (
            "review_due" if "checkpoint_reached_at" in meta else "collecting"
        )
        groups = {}
        for (payload,) in db.execute(
            "SELECT payload FROM observations WHERE protocol=?", (fingerprint,)
        ):
            row = json.loads(payload)
            groups.setdefault(row["case_id"], []).append(row)
        reviews = {k: (v, n) for k, v, n in db.execute("SELECT * FROM reviews")}
        queue = []
        for case_id, rows in groups.items():

            def rank(row):
                return (
                    _true(row["additional_to_grader"]),
                    _true(row["flagged"]),
                    max(
                        float(row.get(q) or 0)
                        for q in protocol["selection"]["questions"]
                    ),
                )

            representative = max(rows, key=rank)
            reasons = set()
            if any(_true(r["flagged"]) for r in rows):
                reasons.add(
                    "additional_flag"
                    if any(_true(r["additional_to_grader"]) for r in rows)
                    else "overlapping_flag"
                )
            if any(
                any(
                    0.5 <= float(r.get(q) or 0) < v["threshold"]
                    for q, v in protocol["selection"]["questions"].items()
                )
                for r in rows
            ):
                reasons.add("borderline")
            final_complete = any(
                _true(r["final"]) and _true(r["complete"]) for r in rows
            )
            if not reasons and final_complete and _sample(case_id):
                reasons.add("random_unflagged")
            verdict, note = reviews.get(case_id, ("", ""))
            if not reasons and (verdict or note):
                reasons.add("previous_review")
            if not reasons:
                continue
            queue.append(
                {
                    "case_id": case_id,
                    "review_reason": ",".join(sorted(reasons)),
                    "answer": representative["answer"],
                    "clue": representative["clue"],
                    "final_observed": final_complete,
                    "observations": len(rows),
                    **{
                        q: representative[q] for q in protocol["selection"]["questions"]
                    },
                    "grader_would_repair": representative["grader_would_repair"],
                    "grade": representative["grade"],
                    "artifact": representative["artifact"],
                    "editor_verdict": verdict,
                    "editor_notes": note,
                }
            )
        queue.sort(
            key=lambda r: (
                "additional_flag" not in r["review_reason"],
                "borderline" not in r["review_reason"],
                r["case_id"],
            )
        )
        pending = sum(
            r["editor_verdict"].strip().lower() not in VERDICTS for r in queue
        )
        summary = {
            "experiment": protocol["id"],
            "protocol_sha256": fingerprint,
            "phase": phase,
            "final_clues": count,
            "target": TARGET,
            "review_cases": len(queue),
            "awaiting_review": pending,
            "observed_cases": len(groups),
            "queue_path": str(queue_path),
            "last_batch": str(batch) if batch else None,
            "last_batch_summary": batch_summary,
            "updated_at": datetime.now(UTC).isoformat(),
            "note": "Counts distinct complete final answer/clue pairs. "
            "At the checkpoint automatic collection pauses before new puzzles. "
            "Review flags, borderline cases, and sampled negatives; then "
            "explicitly revise, retire, or mark ready for integration. "
            "No decision automatically enables repairs.",
        }
        import io

        output = io.StringIO()
        fields = (
            list(queue[0]) if queue else ["case_id", "editor_verdict", "editor_notes"]
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
        _atomic_text(queue_path, output.getvalue())
        _atomic_text(directory / "status.json", json.dumps(summary, indent=2) + "\n")
        _atomic_text(
            directory / "README.md",
            (
                f"# Jev shadow experiment\n\n{count}/{TARGET} distinct final clues; "
                f"{pending} awaiting review. Status: **{phase}**.\n\n"
                "Edit [review.csv](review.csv), then run "
                "`uv run python -m "
                "crossword_generator.content_shadow_study status`.\n\n"
                "At 500: review the queue and record a decision. Collection is paused "
                "until a deliberately revised experiment; no automatic promotion.\n"
            ),
        )
        db.commit()
        return summary
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def progress_line(summary):
    suffix = (
        " · CHECKPOINT: review and decide" if summary["phase"] != "collecting" else ""
    )
    return (
        f"Jev shadow: {summary['final_clues']}/{summary['target']} distinct final "
        f"clues · {summary['awaiting_review']} awaiting review{suffix}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("batch", type=Path)
    decide = commands.add_parser("decide")
    decide.add_argument(
        "decision", choices=["revise", "retire", "ready-for-integration"]
    )
    decide.add_argument("--notes", required=True)
    args = parser.parse_args()
    summary = update_study(
        getattr(args, "batch", None),
        args.study_dir,
        decision=getattr(args, "decision", None),
        notes=getattr(args, "notes", ""),
    )
    print(progress_line(summary))
    print(f"Review: {summary['queue_path']}")


if __name__ == "__main__":
    main()
