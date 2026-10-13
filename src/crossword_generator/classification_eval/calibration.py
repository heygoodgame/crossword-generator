"""Fit decision thresholds on development runs, then apply them unchanged."""

from __future__ import annotations

import statistics
from pathlib import Path

from .core import digest
from .report import binary_metrics, successful_results


def load_predictions(path: Path, split: str):
    manifest, rows, results = successful_results(path)
    if any(r.get("metadata", {}).get("split") != split for r in rows):
        raise ValueError(f"Expected only {split} cases")
    probabilities = {}
    for r in results:
        if r.get("error") or r.get("skipped"):
            continue
        for key, p in r["probabilities"].items():
            probabilities.setdefault((r["case_id"], key), []).append(p)
    means = {k: statistics.mean(v) for k, v in probabilities.items()}
    expected = {(r["id"], q) for r in rows for q in manifest["suite"]["questions"]}
    if set(means) != expected:
        raise ValueError("Calibration requires complete predictions")
    return manifest, rows, means


def fit_thresholds(runs: dict[str, Path], max_fpr: float = 0.02) -> dict:
    if not runs or not 0 <= max_fpr <= 1:
        raise ValueError("Provide candidate runs and a valid false-positive limit")
    choices, fingerprint, groups = {}, None, None
    provenance = {}
    for variant, path in runs.items():
        manifest, rows, predictions = load_predictions(path, "dev")
        if fingerprint is not None and fingerprint != digest(rows):
            raise ValueError("Candidate development corpora must be identical")
        fingerprint = digest(rows)
        groups = sorted({r["metadata"]["group"] for r in rows})
        provenance[variant] = {
            "run_hash": manifest["run_hash"],
            "suite_sha256": digest(manifest["suite"]),
            "model_sha256": digest(manifest["model"]),
        }
        for key in manifest["suite"]["questions"]:
            pairs = [
                (predictions[r["id"], key], r["labels"][key])
                for r in rows
                if r.get("labels", {}).get(key) is not None
            ]
            if not any(y for _, y in pairs) or not any(not y for _, y in pairs):
                raise ValueError("Each calibrated question needs both label classes")
            # 1.01 explicitly represents a disabled gate, even for p=1 outputs.
            for threshold in [i / 100 for i in range(1, 102)]:
                metrics = binary_metrics(pairs, threshold)
                if metrics["fpr"] > max_fpr:
                    continue
                rank = (metrics["recall"], -metrics["fpr"], threshold)
                current = choices.get(key)
                if current is None or rank > tuple(current["selection_rank"]):
                    choices[key] = {
                        "variant": variant,
                        "threshold": threshold,
                        "disabled": threshold > 1,
                        "development_metrics": metrics,
                        "selection_rank": list(rank),
                    }
    return {
        "protocol": "Maximize dev recall at FPR <= limit; break ties by lower "
        "FPR then higher threshold. Exact ties keep the first variant.",
        "max_development_fpr": max_fpr,
        "development_corpus_sha256": fingerprint,
        "development_groups": groups,
        "candidates": provenance,
        "questions": choices,
    }


def evaluate_thresholds(selection: dict, runs: dict[str, Path]) -> dict:
    loaded, fingerprint = {}, None
    for variant, path in runs.items():
        manifest, rows, predictions = load_predictions(path, "test")
        if fingerprint is not None and fingerprint != digest(rows):
            raise ValueError("Candidate test corpora must be identical")
        fingerprint = digest(rows)
        if set(selection["development_groups"]) & {
            r["metadata"]["group"] for r in rows
        }:
            raise ValueError("Development and test groups overlap")
        config = selection["candidates"][variant]
        if config["suite_sha256"] != digest(manifest["suite"]) or config[
            "model_sha256"
        ] != digest(manifest["model"]):
            raise ValueError("Test model/suite differs from fitted configuration")
        loaded[variant] = (rows, predictions)
    questions, review, union = {}, [], []
    for key, choice in selection["questions"].items():
        rows, predictions = loaded[choice["variant"]]
        pairs, origins = [], {}
        for row in rows:
            p = predictions[row["id"], key]
            y = row.get("labels", {}).get(key)
            flagged = p >= choice["threshold"]
            review.append(
                {
                    "case_id": row["id"],
                    "state": row["state"],
                    "question": key,
                    "probability": p,
                    "label": y,
                    "flagged": flagged,
                    "variant": choice["variant"],
                    "threshold": choice["threshold"],
                }
            )
            if y is not None:
                pairs.append((p, y))
                origins.setdefault(row["metadata"].get("origin", "unknown"), []).append(
                    (p, y)
                )
        questions[key] = {
            "variant": choice["variant"],
            "threshold": choice["threshold"],
            "metrics": binary_metrics(pairs, choice["threshold"]),
            "by_origin": {
                k: binary_metrics(v, choice["threshold"]) for k, v in origins.items()
            },
        }
    by_case = {}
    for r in review:
        by_case.setdefault(r["case_id"], []).append(r)
    for values in by_case.values():
        labels = [v["label"] for v in values]
        # A known violation establishes union-positive even if another label is null.
        label = (
            True
            if any(y is True for y in labels)
            else (None if any(y is None for y in labels) else False)
        )
        if label is not None:
            union.append((float(any(v["flagged"] for v in values)), label))
    return {
        "selection_sha256": digest(selection),
        "test_corpus_sha256": fingerprint,
        "questions": questions,
        "any_policy_violation": binary_metrics(union, 0.5),
        "review": review,
    }
