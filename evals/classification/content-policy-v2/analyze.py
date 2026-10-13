"""Reproduce development selection or report frozen held-out policy results.

Usage: python analyze.py fit|report --runs-root output/classification-evals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossword_generator.classification_eval.calibration import (
    evaluate_thresholds,
    fit_thresholds,
)
from crossword_generator.classification_eval.core import digest, write_json
from crossword_generator.classification_eval.report import (
    binary_metrics,
    successful_results,
)

ROOT = Path(__file__).resolve().parent
KEYS = ["firearms_angle", "alcohol_angle", "drug_use_angle"]


def union_label(row):
    labels = [row["labels"][k] for k in KEYS]
    return (
        True
        if any(y is True for y in labels)
        else (None if any(y is None for y in labels) else False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "report"])
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    runs = args.runs_root
    dev = {
        "context": runs / "content-v2-jev-context-dev",
        "clue-only": runs / "content-v2-jev-clue-dev",
    }
    selection_path = ROOT / "selection.json"
    if args.command == "fit":
        selection = fit_thresholds(dev)
        if selection_path.exists():
            if json.loads(selection_path.read_text()) != selection:
                raise ValueError("Existing selection differs; version the experiment")
        else:
            write_json(selection_path, selection)
        print(selection_path)
        return
    selection = json.loads(selection_path.read_text())
    heldout = evaluate_thresholds(
        selection,
        {
            "context": runs / "content-v2-jev-context-test",
            "clue-only": runs / "content-v2-jev-clue-test",
        },
    )
    write_json(ROOT / "test-results.json", heldout)
    _, rows, baseline_results = successful_results(runs / "content-v2-claude")
    baseline = {r["case_id"]: r for r in baseline_results if not r.get("error")}
    _, _, grader_results = successful_results(runs / "content-v2-current-grader")
    grader = {r["case_id"]: r for r in grader_results if not r.get("error")}
    if len(baseline) != len(rows) or len(grader) != len(rows):
        raise ValueError("Wait for complete baseline and grader runs")
    test = [r for r in rows if r["metadata"]["split"] == "test"]
    baseline_metrics = {
        k: binary_metrics(
            [
                (baseline[r["id"]]["probabilities"][k], r["labels"][k])
                for r in test
                if r["labels"][k] is not None
            ],
            0.5,
        )
        for k in KEYS
    }
    disagreements = [
        {
            "case_id": r["id"],
            "state": r["state"],
            "question": k,
            "split": r["metadata"]["split"],
            "label": r["labels"][k],
            "claude_probability": baseline[r["id"]]["probabilities"][k],
        }
        for r in rows
        for k in KEYS
        if r["labels"][k] is not None
        and (baseline[r["id"]]["probabilities"][k] >= 0.5) != r["labels"][k]
    ]
    selected_flags = {}
    for r in heldout["review"]:
        selected_flags[r["case_id"]] = (
            selected_flags.get(r["case_id"], False) or r["flagged"]
        )
    comparison = []
    for row in test:
        g = grader[row["id"]]["grade"]
        comparison.append(
            {
                "case_id": row["id"],
                "state": row["state"],
                "label_any_policy_violation": union_label(row),
                "jev_flag": selected_flags[row["id"]],
                "grader_fairness_flag": g["fairness"] < 15,
                "grader": g,
            }
        )
    known_positive = [r for r in comparison if r["label_any_policy_violation"] is True]
    known_negative = [r for r in comparison if r["label_any_policy_violation"] is False]
    overlap = {
        "positive_cases": len(known_positive),
        "grader_fairness_flagged_positive": sum(
            r["grader_fairness_flag"] for r in known_positive
        ),
        "jev_additional_positive_flags": [
            r for r in known_positive if r["jev_flag"] and not r["grader_fairness_flag"]
        ],
        "combined_flagged_positive": sum(
            r["jev_flag"] or r["grader_fairness_flag"] for r in known_positive
        ),
        "grader_fairness_flagged_content_negative": sum(
            r["grader_fairness_flag"] for r in known_negative
        ),
        "caveat": "Raw fairness <15 also flags leaks and other defects. "
        "Content-negative does not mean the overall clue is clean. "
        "No crossings; Easy Mini rubric, one clue per call.",
    }
    write_json(
        ROOT / "baseline-comparison.json",
        {
            "selection_sha256": digest(selection),
            "claude_test_default_05": baseline_metrics,
            "blind_label_disagreements": disagreements,
            "grader_overlap": overlap,
            "test_rows": comparison,
        },
    )
    lines = [
        "# Content policy v2: measured results",
        "",
        "Thresholds and state were selected using only the 68 development cases.",
        "Test set: 132 cases; four ambiguous category labels excluded "
        "where applicable.",
        "",
        "| Check | State | Threshold | TP | FN | FP | TN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for k, v in heldout["questions"].items():
        m = v["metrics"]
        lines.append(
            f"| {k} | {v['variant']} | {v['threshold']} | "
            f"{m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} |"
        )
    m = heldout["any_policy_violation"]
    lines += [
        "",
        f"Combined: {m['tp']}/{m['positive']} violations caught; "
        f"{m['fp']}/{m['negative']} content-clean cases flagged.",
        "",
        "## Missed violations",
        "",
    ]
    for r in heldout["review"]:
        if r["label"] is True and not r["flagged"]:
            lines.append(
                f"- {r['state']['answer']} / {r['state']['clue']}: "
                f"{r['question']} = {r['probability']} (threshold {r['threshold']})."
            )
    lines += [
        "",
        "## Existing grader overlap",
        "",
        f"Raw fairness <15 flagged {overlap['grader_fairness_flagged_positive']}"
        f"/{len(known_positive)} content-positive test cases. Jev added "
        f"{len(overlap['jev_additional_positive_flags'])} positive flags.",
        f"Combined: {overlap['combined_flagged_positive']}/{len(known_positive)} "
        "content-positive test cases flagged.",
        "",
        f"It also flagged {overlap['grader_fairness_flagged_content_negative']}"
        f"/{len(known_negative)} content-negative cases for fairness. "
        "These may be valid flags for other defects; they are not automatically "
        "content false positives.",
        "",
        overlap["caveat"],
        "",
        "## Blind Claude classifier baseline at 0.5",
        "",
        "Claude's probabilities were not threshold-tuned; this is a reference, "
        "not an equal-operating-point ranking.",
        "",
        "| Check | TP | FN | FP | TN |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, m in baseline_metrics.items():
        lines.append(f"| {k} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} |")
    lines += [
        "",
        f"Blind Claude review disagreed with {len(disagreements)} of the "
        "596 known category labels across the full corpus. "
        "Original labels were retained, "
        "not revised to match Claude or Jev. See baseline-comparison.json.",
        "",
        "Labels are provisional agent judgments. Cases are deliberately enriched with "
        "boundary examples and are not a production-prevalence sample. "
        "Historical cases "
        "were explored previously. No production decision policy was changed.",
    ]
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "results.md")


if __name__ == "__main__":
    main()
