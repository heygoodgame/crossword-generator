#!/usr/bin/env python3
"""Scan generated puzzles for the pre-PR#20 "dropped fact-check fix" bug.

Failure mode: the clue fact-checker proposed/kept a corrected clue for an entry
(recorded in metadata.clue_fact_check), but the final clues[] (and the published
.ipuz) still carry the original, un-repaired clue. Under the old order-dependent
repair logic the computed fix could be dropped before reaching the envelope.

Detection signature, per entry (matched by number+direction+answer):
    clue in metadata.clue_fact_check  !=  clue in clues[]

Each such mismatch is a clue whose fact-checked version never made it into the
final puzzle.
"""

import argparse
import glob
import json
import os
import sys

# Shipped clues scoring below this, where the fact-check had a different clue on
# hand, are treated as high-signal dropped fixes (NAT scored 56).
LOW_SCORE = 70


def key(entry):
    return (
        entry.get("number"),
        (entry.get("direction") or "").lower(),
        (entry.get("answer") or "").upper(),
    )


def published_clue(batch_dir, difficulty, grid, seed_dir):
    """Best-effort: return {(num,dir,answer): clue} from the published .ipuz, if present."""
    seed = os.path.basename(seed_dir)
    ipuz = os.path.join(batch_dir, difficulty, grid, f"{seed}.ipuz")
    if not os.path.exists(ipuz):
        return None
    try:
        d = json.load(open(ipuz))
    except Exception:
        return None
    out = {}
    clues = d.get("clues", {})
    for sec_key, direction in (("Across", "across"), ("across", "across"), ("Down", "down"), ("down", "down")):
        for item in clues.get(sec_key, []) or []:
            if isinstance(item, list) and len(item) >= 2:
                out[(item[0], direction)] = item[1]
    return out


def scan_file(path):
    """Return list of dropped-fix findings for one intermediate JSON, or []."""
    try:
        d = json.load(open(path))
    except Exception:
        return []
    fc_list = (d.get("metadata") or {}).get("clue_fact_check")
    clues = d.get("clues")
    if not fc_list or not clues:
        return []

    final = {key(c): c.get("clue") for c in clues}
    scores = {key(c): c.get("quality_score") for c in clues}
    findings = []
    for fc in fc_list:
        k = key(fc)
        fc_clue = fc.get("clue")
        final_clue = final.get(k)
        if final_clue is None or fc_clue is None:
            continue
        status = (fc.get("status") or "").lower()
        score = scores.get(k)

        # Tier "confirmed": fact-check judged this exact clue not-safe
        # (incorrect/uncertain) AND that same flagged clue still ships. A clue
        # the checker explicitly called wrong made it into the puzzle.
        if status != "safe" and fc_clue == final_clue:
            tier = "confirmed"
        # Tier "low-quality": NAT shape -- fact-check's clue differs from what
        # ships AND the shipped clue scored poorly (< threshold). The checker
        # had a better clue on hand that never reached clues[]. High-signal.
        elif fc_clue != final_clue and score is not None and score < LOW_SCORE:
            tier = "low-quality"
        # Tier "suspected": fact-check's clue differs but the shipped clue
        # scored fine -> mostly legit later edits. Reported for completeness.
        elif fc_clue != final_clue:
            tier = "suspected"
        else:
            continue

        findings.append(
            {
                "tier": tier,
                "number": k[0],
                "direction": k[1],
                "answer": k[2],
                "final_clue": final_clue,
                "factcheck_clue": fc_clue,
                "factcheck_status": fc.get("status"),
                "quality_score": score,
            }
        )
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="output/batches/unlimited-*",
        help="batch directory glob (default: unlimited-* only)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--no-suspected", action="store_true", help="hide the noisy suspected tier")
    args = ap.parse_args()

    intermediates = []
    for batch_dir in sorted(glob.glob(args.glob)):
        intermediates += glob.glob(
            os.path.join(batch_dir, "**", "intermediate_clue-generation-with-grading_*.json"),
            recursive=True,
        )

    flagged = []
    for path in sorted(intermediates):
        findings = scan_file(path)
        if not findings:
            continue
        # path: output/batches/<batch>/<difficulty>/<grid>/intermediates/<seed>/<file>
        parts = path.split(os.sep)
        try:
            i = parts.index("intermediates")
            batch_dir = os.sep.join(parts[:i - 2])
            difficulty, grid = parts[i - 2], parts[i - 1]
            seed_dir = os.sep.join(parts[: i + 2])
        except ValueError:
            batch_dir = difficulty = grid = seed_dir = None

        # cross-check against the actually-published .ipuz when we can
        pub = published_clue(batch_dir, difficulty, grid, seed_dir) if batch_dir else None
        for f in findings:
            if pub is not None:
                f["in_published_ipuz"] = pub.get((f["number"], f["direction"])) == f["final_clue"]
            else:
                f["in_published_ipuz"] = None

        flagged.append(
            {
                "batch": os.path.basename(batch_dir) if batch_dir else None,
                "difficulty": difficulty,
                "grid": grid,
                "seed": os.path.basename(seed_dir) if seed_dir else None,
                "path": path,
                "findings": findings,
            }
        )

    if args.json:
        print(json.dumps(flagged, indent=2, ensure_ascii=False))
        return

    if not flagged:
        print("No dropped fact-check fixes found.")
        return

    def tier_count(t):
        return sum(1 for p in flagged for f in p["findings"] if f["tier"] == t)

    def puzzles_in(t):
        return len([p for p in flagged if any(f["tier"] == t for f in p["findings"])])

    print("=" * 70)
    print(f"CONFIRMED:   {tier_count('confirmed'):>4} clue(s) fact-check called incorrect/uncertain that")
    print(f"             still ship -- across {puzzles_in('confirmed')} puzzle(s).")
    print(f"LOW-QUALITY: {tier_count('low-quality'):>4} clue(s) (NAT-shape) where a better fact-check clue")
    print(f"             was dropped and a clue scoring <{LOW_SCORE} shipped -- {puzzles_in('low-quality')} puzzle(s).")
    print(f"SUSPECTED:   {tier_count('suspected'):>4} clue(s) where fact-check differs but shipped clue")
    print(f"             scored fine (mostly legit later edits).")
    print("=" * 70)

    def render(findings):
        for f in findings:
            pub = f["in_published_ipuz"]
            mark = "SHIPPED" if pub else ("ipuz-clean" if pub is False else "no-ipuz")
            qs = f.get("quality_score")
            qs_s = f" q={int(qs)}" if isinstance(qs, (int, float)) else ""
            print(f"  {f['number']}{f['direction'][0].upper()} {f['answer']:<8} [{mark}] status={f['factcheck_status']}{qs_s}")
            print(f"     ships:      {f['final_clue']!r}")
            print(f"     factcheck:  {f['factcheck_clue']!r}")

    tiers = ("confirmed", "low-quality") if args.no_suspected else ("confirmed", "low-quality", "suspected")
    for label in tiers:
        puzzles = [p for p in flagged if any(f["tier"] == label for f in p["findings"])]
        if not puzzles:
            continue
        print(f"\n\n########## {label.upper()} ##########\n")
        for p in puzzles:
            fs = [f for f in p["findings"] if f["tier"] == label]
            print(f"## {p['batch']}  [{p['difficulty']}/{p['grid']}  {p['seed']}]")
            render(fs)
            print()


if __name__ == "__main__":
    sys.exit(main())
