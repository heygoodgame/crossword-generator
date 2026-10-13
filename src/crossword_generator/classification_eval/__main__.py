"""Run with uv run python -m crossword_generator.classification_eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import digest, import_legacy, read_rows, run_experiment, validate_inputs
from .import_logs import import_logs
from .report import archive, compare, rescore, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("import-legacy", "import-logs"):
        cmd = commands.add_parser(name)
        cmd.add_argument("source", type=Path)
        cmd.add_argument("output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--seed", default="classification-v1")
    run.add_argument("--dry-run", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    report.add_argument("--threshold", type=float, default=0.5)
    frozen = commands.add_parser("archive")
    frozen.add_argument("source", type=Path)
    frozen.add_argument("output", type=Path)
    paired = commands.add_parser("compare")
    paired.add_argument("left", type=Path)
    paired.add_argument("right", type=Path)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--threshold", type=float, default=0.5)
    reviewed = commands.add_parser("rescore")
    reviewed.add_argument("source", type=Path)
    reviewed.add_argument("--labels", type=Path, required=True)
    reviewed.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "import-legacy":
        print(f"Imported {len(import_legacy(args.source, args.output))} clues")
    elif args.command == "import-logs":
        info = import_logs(args.source, args.output)
        print(
            json.dumps({k: len(v) if k == "problems" else v for k, v in info.items()})
        )
    elif args.command == "report":
        summarize(args.directory, args.threshold)
        print(args.directory / "report.md")
    elif args.command == "archive":
        archive(args.source, args.output)
        print(args.output / "report.md")
    elif args.command == "compare":
        result = compare(args.left, args.right, args.output, args.threshold)
        print(
            f"Paired {result['paired_predictions']} predictions; "
            f"{result['decision_changes']} changed decisions"
        )
    elif args.command == "rescore":
        rescore(args.source, args.labels, args.output)
        print(args.output / "report.md")
    else:
        rows = read_rows(args.corpus)
        if args.limit < 0:
            parser.error("--limit cannot be negative")
        # Hash order provides reproducible sampling without relying on file ordering.
        rows = sorted(rows, key=lambda r: digest([args.seed, r["id"]]))
        if args.limit:
            rows = rows[: args.limit]
        suite, model = (
            json.loads(path.read_text()) for path in (args.suite, args.model)
        )
        validate_inputs(rows, suite, model)
        if args.dry_run:
            print(
                f"Validated {len(rows)} cases, {len(suite['questions'])} questions, "
                f"{args.repeats} repeats; no API calls"
            )
            return
        results = run_experiment(
            rows, suite, model, args.output, workers=args.workers, repeats=args.repeats
        )
        summarize(args.output)
        print(args.output / "report.md")
        latest = {(r["case_id"], r["repeat"]): r for r in results}
        if any(r.get("error") for r in latest.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
