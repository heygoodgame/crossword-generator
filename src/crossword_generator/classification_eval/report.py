"""Per-question metrics, paired comparisons, and reviewable disagreement exports."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .core import digest, read_rows, validate_inputs, write_json


def binary_metrics(pairs: list[tuple[float, bool]], threshold: float) -> dict:
    tp = sum(p >= threshold and y for p, y in pairs)
    fp = sum(p >= threshold and not y for p, y in pairs)
    fn = sum(p < threshold and y for p, y in pairs)
    tn = sum(p < threshold and not y for p, y in pairs)
    positive = [p for p, y in pairs if y]
    negative = [p for p, y in pairs if not y]

    def ratio(a, b):
        return a / b if b else None

    auc = None
    if positive and negative:
        # Rank-sum AUC, with half credit for ties.
        ordered = sorted(pairs)
        rank_sum, i = 0.0, 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            rank_sum += (i + 1 + j) / 2 * sum(y for _, y in ordered[i:j])
            i = j
        auc = (rank_sum - len(positive) * (len(positive) + 1) / 2) / (
            len(positive) * len(negative)
        )
    return {
        "n": len(pairs),
        "positive": len(positive),
        "negative": len(negative),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": ratio(tp + tn, len(pairs)),
        "recall": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
        "precision": ratio(tp, tp + fp),
        "auc": auc,
        "brier": statistics.mean((p - int(y)) ** 2 for p, y in pairs)
        if pairs
        else None,
    }


def successful_results(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    corpus = json.loads((directory / "corpus.json").read_text())
    records = read_results(directory)
    # Attempts are append-only. The last attempt for each case/repeat wins.
    latest = {(r["case_id"], r["repeat"]): r for r in records}
    return manifest, corpus, list(latest.values())


def read_results(directory: Path) -> list[dict]:
    path = directory / "results.jsonl"
    return read_rows(path if path.exists() else directory / "results.json")


def archive(source: Path, output: Path) -> dict:
    """Freeze a completed run in Git-friendly JSON without repeated requests."""
    if (source / ".running").exists():
        raise ValueError("Cannot archive a running experiment")
    manifest, corpus, _ = successful_results(source)
    attempts = read_results(source)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    write_json(output / "corpus.json", corpus)
    write_json(
        output / "results.json",
        [{k: v for k, v in row.items() if k != "request"} for row in attempts],
    )
    write_json(
        output / "archive.json",
        {
            "source": str(source),
            "original_records_sha256": digest(attempts),
            "note": "All attempts retained; repeated request objects omitted. "
            "Reconstruct each with core.prepare(corpus row, manifest suite).",
        },
    )
    return summarize(output)


def summarize(directory: Path, threshold: float = 0.5) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    manifest, corpus, results = successful_results(directory)
    cases = {row["id"]: row for row in corpus}
    successful = [r for r in results if not r.get("error") and not r.get("skipped")]
    # Repeats characterize stability. Metrics use their mean once per case.
    probabilities = defaultdict(list)
    for result in successful:
        for question, p in result["probabilities"].items():
            probabilities[(result["case_id"], question)].append(p)
    questions, review = {}, []
    for question in manifest["suite"]["questions"]:
        values, pairs = [], []
        grouped_labels = defaultdict(list)
        stability = []
        for (case_id, key), samples in probabilities.items():
            if key != question:
                continue
            p = statistics.mean(samples)
            values.append(p)
            row = cases[case_id]
            label = row.get("labels", {}).get(key)
            source = row.get("metadata", {}).get("label_source", "unspecified")
            if label is not None:
                pairs.append((p, label))
                grouped_labels[source].append((p, label))
            if len(samples) > 1:
                stability.append(max(samples) - min(samples))
            review.append(
                {
                    "case_id": case_id,
                    "question": key,
                    "answer": row["state"].get("answer"),
                    "clue": row["state"].get("clue"),
                    "hint": row["state"].get("hint"),
                    "previous_clue": row["state"].get("previous_clue"),
                    "probability": p,
                    "predicted": p >= threshold,
                    "label": label,
                    "label_source": source,
                    "disagreement": label is not None and (p >= threshold) != label,
                    "repeats": len(samples),
                    "range": max(samples) - min(samples),
                    "historical_feedback": row.get("metadata", {})
                    .get("reference", {})
                    .get("feedback"),
                }
            )
        questions[question] = {
            "scored_cases": len(values),
            "flagged": sum(p >= threshold for p in values),
            "uncertain_02_08": sum(0.2 < p < 0.8 for p in values),
            "metrics": binary_metrics(pairs, threshold),
            "metrics_by_label_source": {
                k: binary_metrics(v, threshold) for k, v in grouped_labels.items()
            },
            "threshold_sweep_exploratory": {
                str(t / 10): binary_metrics(pairs, t / 10) for t in range(1, 10)
            },
            "mean_repeat_range": statistics.mean(stability) if stability else None,
        }
    usage = Counter()
    attempts = read_results(directory)
    for result in attempts:
        usage.update(result.get("usage", {}))
    rates = manifest["model"].get("prices_usd_per_million")
    cost = sum(usage.get(k, 0) * v / 1e6 for k, v in rates.items()) if rates else None
    provider_costs = [
        r.get("cost", {}).get("estimated_cost_usd")
        for r in attempts
        if r.get("cost", {}).get("estimated_cost_usd") is not None
    ]
    if rates is None and len(provider_costs) == len(attempts):
        cost = sum(provider_costs)
    latency = sorted(r["latency_s"] for r in successful)
    summary = {
        "run_hash": manifest["run_hash"],
        "suite": manifest["suite"]["id"],
        "requested_model": manifest["model"]["model"],
        "resolved_models": sorted(
            {r.get("resolved_model") or "unknown" for r in successful}
        ),
        "cases": len(corpus),
        "expected_requests": len(corpus) * manifest["repeats"],
        "completed_requests": len(results),
        "attempts": len(attempts),
        "attempts_with_usage": sum(bool(r.get("usage")) for r in attempts),
        "successful_requests": len(successful),
        "errors": sum(bool(r.get("error")) for r in results),
        "skipped_requests": sum(bool(r.get("skipped")) for r in results),
        "usage": dict(usage),
        "estimated_cost_usd": cost,
        "latency_p50_s": statistics.median(latency) if latency else None,
        "latency_p95_s": latency[min(len(latency) - 1, int(len(latency) * 0.95))]
        if latency
        else None,
        "threshold": threshold,
        "questions": questions,
        "label_sources": dict(
            Counter(
                r.get("metadata", {}).get("label_source", "unspecified") for r in corpus
            )
        ),
    }
    write_json(directory / "summary.json", summary)
    review.sort(key=lambda r: (not r["disagreement"], -r["probability"], r["case_id"]))
    if review:
        with (directory / "review.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    lines = [
        f"# {summary['suite']} — {summary['requested_model']}",
        "",
        f"Cases: {len(corpus)}; successful requests: {len(successful)}; "
        f"errors: {summary['errors']}; threshold: {threshold}.",
        "",
        "Labels are source-specific references, not automatically ground truth. "
        "Historical feedback never enters model state. Unlabeled cases contribute "
        "only to flag counts; a missing label is never treated as clean. "
        "Threshold sweeps are exploratory, not held-out validation. "
        "Repeat metrics use one mean probability per case.",
        "",
        f"Label sources: `{summary['label_sources']}`",
        "",
        f"Estimated cost: {cost}; latency p50/p95: "
        f"{summary['latency_p50_s']}/{summary['latency_p95_s']} seconds.",
        "",
        "| Question | Scored | Flagged | Labels (+/-) | "
        "Accuracy | Recall | FPR | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    for key, info in questions.items():
        m = info["metrics"]
        lines.append(
            f"| {key} | {info['scored_cases']} | {info['flagged']} | "
            f"{m['positive']}/{m['negative']} | {fmt(m['accuracy'])} | "
            f"{fmt(m['recall'])} | {fmt(m['fpr'])} | {fmt(m['auc'])} |"
        )
    lines += ["", "See `review.csv` for all predictions and label disagreements."]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def compare(left: Path, right: Path, output: Path, threshold: float = 0.5) -> dict:
    lm, lc, lr = successful_results(left)
    rm, rc, rr = successful_results(right)
    if digest(lc) != digest(rc) or digest(lm["suite"]) != digest(rm["suite"]):
        raise ValueError("Paired model comparisons require identical corpus and suite")

    def average(records):
        p = defaultdict(list)
        for r in records:
            if not r.get("error"):
                for q, v in r.get("probabilities", {}).items():
                    p[(r["case_id"], q)].append(v)
        return {k: statistics.mean(v) for k, v in p.items()}

    a, b = average(lr), average(rr)
    shared = sorted(a.keys() & b.keys())
    changes = [
        {
            "case_id": key[0],
            "question": key[1],
            "left": a[key],
            "right": b[key],
            "delta": b[key] - a[key],
            "decision_changed": (a[key] >= threshold) != (b[key] >= threshold),
        }
        for key in shared
    ]
    result = {
        "left_run": lm["run_hash"],
        "right_run": rm["run_hash"],
        "threshold": threshold,
        "paired_predictions": len(shared),
        "left_only": len(a.keys() - b.keys()),
        "right_only": len(b.keys() - a.keys()),
        "decision_changes": sum(r["decision_changed"] for r in changes),
        "predictions": sorted(changes, key=lambda r: -abs(r["delta"])),
    }
    write_json(output, result)
    return result


def rescore(source: Path, labels: Path, output: Path) -> dict:
    """Review new labels without another inference call; preserve run provenance."""
    manifest, original, results = successful_results(source)
    reviewed = read_rows(labels)
    validate_inputs(reviewed, manifest["suite"], manifest["model"])
    by_id = {row["id"]: row for row in original}
    for row in reviewed:
        if row["id"] not in by_id or row["state"] != by_id[row["id"]]["state"]:
            raise ValueError("Reviewed labels must refer to unchanged original states")
    if output.exists():
        raise ValueError("Rescore output must be a new directory")
    ids = {row["id"] for row in reviewed}
    derived = manifest | {
        "corpus_sha256": digest(reviewed),
        "derived_from": {"run_hash": manifest["run_hash"], "directory": str(source)},
        "inference_reused": True,
    }
    derived["run_hash"] = digest([manifest["run_hash"], reviewed])
    write_json(output / "manifest.json", derived)
    write_json(output / "corpus.json", reviewed)
    with (output / "results.jsonl").open("w") as handle:
        for result in results:
            if result["case_id"] in ids:
                handle.write(json.dumps(result) + "\n")
    return summarize(output)
