"""Recover clue states and historical references without turning prose into labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .core import digest, write_json

CLUE_LINE = re.compile(
    r'^- (\d+)-(ACROSS|DOWN): Answer=(.*?), Clue="(.*)"'
    r"(?: \(crossing words: (.*)\))?$",
    re.MULTILINE,
)


def extract_grading_record(record: dict, source: str) -> list[dict]:
    context = record.get("context", {})
    if context.get("step") != "clue_grading" or record.get("error"):
        return []
    request = record.get("request") or {}
    prompt = request.get("prompt")
    if not prompt:
        prompt = "\n".join(
            m.get("content", "")
            for m in request.get("messages", [])
            if m.get("role") == "user"
        )
    matches = list(CLUE_LINE.finditer(prompt))
    if not matches:
        raise ValueError("No supported clue lines in grading request")
    response = record.get("response") or {}
    text = response.get("text", "")
    start = text.find("[")
    if start < 0:
        raise ValueError("No JSON array in grading response")
    grades, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(grades, list):
        raise ValueError("Expected grading array")
    by_key = {(int(g["number"]), g["direction"].lower()): g for g in grades}
    if len(by_key) != len(grades):
        raise ValueError("Duplicate grading response keys")
    rows = []
    for match in matches:
        number, direction, answer, clue, crossing = match.groups()
        key = (int(number), direction.lower())
        if key not in by_key:
            raise ValueError("Grading response omitted an input clue")
        state = {
            "answer": answer,
            "clue": clue,
            "crossing_answers": crossing or "",
            "puzzle_difficulty": context.get("difficulty"),
            "puzzle_type": context.get("puzzle_type"),
        }
        rows.append(
            {
                "id": digest(state)[:20],
                "state": state,
                "labels": {},
                "metadata": {
                    "label_source": "historical-model-unverified",
                    "references": [
                        {
                            "source": source,
                            "request_id": record.get("request_id"),
                            "provider": record.get("provider"),
                            "model": record.get("model"),
                            "grade": by_key[key],
                        }
                    ],
                },
            }
        )
    return rows


def import_logs(root: Path, output: Path) -> dict:
    files = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    unique, problems = {}, []
    parsed = 0
    for path in files:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    rows = extract_grading_record(record, f"{path}:{line_number}")
                    parsed += bool(rows)
                    for row in rows:
                        if row["id"] in unique:
                            unique[row["id"]]["metadata"]["references"].extend(
                                row["metadata"]["references"]
                            )
                        else:
                            unique[row["id"]] = row
                except (ValueError, KeyError, TypeError, AttributeError):
                    problems.append(
                        {
                            "source": str(path),
                            "line": line_number,
                            "error": "unsupported or malformed record",
                        }
                    )
    write_json(output, sorted(unique.values(), key=lambda r: r["id"]))
    report = {
        "files": len(files),
        "grading_calls": parsed,
        "unique_clues": len(unique),
        "problems": problems,
        "note": "Scores and feedback are unverified references, not labels. "
        "Historical runtime repair decisions cannot be recovered from "
        "scores alone without that run's exact config and pipeline.",
    }
    write_json(output.with_suffix(".import-report.json"), report)
    return report
