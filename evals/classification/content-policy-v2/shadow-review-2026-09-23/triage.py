"""Save the agent's proposals without converting them into human labels."""

import csv
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from crossword_generator.content_shadow_study import update_study

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
QUEUE = ROOT / "output/content-shadow/content-policy-v2/review.csv"
original = QUEUE.read_text()
rows = list(csv.DictReader(io.StringIO(original)))
assert len(rows) == 29

# These are editorial judgments, not additional Jev results or human approval.
notes = {
    "VODKA": "The clue directly identifies an alcoholic beverage. Jev's 0.91 alcohol flag is useful additional coverage: the recorded grader fairness was 23/25 and detector findings were empty. Hold this candidate for refill/replacement; forcing a neutral sense of VODKA would be contrived.",
    "NORM": "Barstool regular evokes a drinking setting, but the frozen alcohol question does not explicitly define venue-only references. Propose ambiguous for the classifier label and rewrite the clue through the everyday meaning, e.g. 'The usual standard'.",
    "DRE": "A rapper/producer name is not itself a reference to recreational drug use. Do not import associations with the artist into the label. Acceptable for all three content questions.",
    "TAT": "A tattoo abbreviation; no firearm, alcohol, or recreational-drug angle.",
    "SITS": "Ordinary sitting action; no targeted content angle.",
    "SHE": "Ordinary pronoun reference; no targeted content angle.",
    "ALLOW": "Ordinary permission sense; no targeted content angle.",
    "WARDS": "Hospital-room sense. Ordinary medical context is outside the recreational-drug question.",
    "DYE": "Changing a shirt's color; no targeted content angle.",
    "SON": "Ordinary family relationship; no targeted content angle.",
    "EBB": "Tidal movement; no targeted content angle.",
    "ETHAN": "Actor identification using a film credit. The clue does not reference guns or drug use; associations with a work's subject matter alone should not create a violation under this frozen rubric.",
    "ANKLE": "A neutral anatomical definition, without an anatomy joke or targeted content angle.",
    "MISS": "Longing for someone, not a missed shot or weapon sense.",
    "SLEEP": "Ordinary sleep/rest sense; no intoxication or drug implication in the wording.",
    "ETA": "Content-negative, but a separate literal answer leak: zETA and thETA both contain ETA contiguously. The current deterministic detector returned no finding. Propose 'Seventh Greek letter' as a replacement, subject to the normal duplicate/quality checks.",
    "NASTY": "Defines a negative adjective without describing prohibited content or directing abuse at anyone. Acceptable for these three checks.",
    "MRS": "Conventional title abbreviation; no targeted content angle.",
    "LAPSE": "An ordinary memory slip; no stated substance-use or recovery sense.",
    "SCENT": "A flower's smell; no targeted content angle.",
    "DUEL": "Showdown at dawn evokes a violent duel, but the wording does not resolve firearms versus another contest/weapon. Propose ambiguous for the narrow firearms label. Rewrite under the broader generator instruction to avoid evoking violence, e.g. 'One-on-one contest'. Do not count this as a confirmed Jev firearms miss.",
    "HULL": "Ship structure; no targeted content angle.",
    "NICE": "A familiar saying, not a firearm, alcohol, or recreational-drug reference. Tone/craft can be considered separately.",
    "WOE": "Ordinary sadness vocabulary; no targeted content angle.",
    "OLDIE": "An older song; no targeted content angle.",
    "LES": "Content-negative, but a separate literal answer leak: MisérabLES contains LES contiguously. The current deterministic detector returned no finding. Propose a French-article clue, e.g. 'The, in French (plural)', subject to normal duplicate and quality checks.",
    "AUDIO": "Recording/sound terminology; no targeted content angle.",
    "HOW": "An ordinary spelling question; no targeted content angle.",
}
created = datetime.now(UTC).isoformat()
proposals = []
for row in rows:
    answer = row["answer"]
    verdict = "violation" if answer == "VODKA" else (
        "ambiguous" if answer in {"NORM", "DUEL"} else "acceptable"
    )
    action = "hold_for_refill_or_replacement" if answer == "VODKA" else (
        "rewrite" if answer in {"NORM", "DUEL", "ETA", "LES"} else "no_content_change"
    )
    labels = {"firearms_angle": False, "alcohol_angle": False, "drug_use_angle": False}
    if answer == "VODKA":
        labels["alcohol_angle"] = True
    if answer == "NORM":
        labels["alcohol_angle"] = None
    if answer == "DUEL":
        labels["firearms_angle"] = None
    rationale = notes[answer]
    if answer == "DRE" and row["final_observed"] == "False":
        rationale += " This intermediate wording was already replaced because of an exact duplicate; the final wording is 'Music mogul Dr. ___'. No restoration is proposed."
    proposal = {
        "source_row": dict(row), "proposed_content_verdict": verdict,
        "proposed_category_labels": labels, "proposed_action": action,
        "outside_content_scope": "literal_answer_leak" if answer in {"ETA", "LES"} else None,
        "rationale": rationale,
    }
    proposals.append(proposal)
    marker = "Agent triage proposal 2026-09-23 (not human-approved):"
    if marker not in row["editor_notes"]:
        row["editor_notes"] = (row["editor_notes"] + "\n" if row["editor_notes"] else "") + (
            f"{marker} content={verdict}; action={action}. {rationale}"
        )

snapshot = {
    "created_at": created, "author": "Codex agent", "human_approved": False,
    "protocol": "content-policy-v2", "scope": "Three frozen content questions; separate editorial/leak observations do not change those labels.",
    "source_queue": str(QUEUE), "reviewed_rows": len(rows),
    "proposed_content_verdict_counts": dict(Counter(p["proposed_content_verdict"] for p in proposals)),
    "proposed_action_counts": dict(Counter(p["proposed_action"] for p in proposals)),
    "proposals": proposals,
}
(HERE / "proposals.json").write_text(json.dumps(snapshot, indent=2) + "\n")

# Keep the shared queue's editor decisions intact. Abort if it changed mid-review.
assert QUEUE.read_text() == original, "Queue changed; reload before writing notes"
output = io.StringIO()
writer = csv.DictWriter(output, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
temporary = QUEUE.with_suffix(".triage.tmp")
temporary.write_text(output.getvalue())
temporary.replace(QUEUE)
status = update_study()
after = {r["case_id"]: r for r in csv.DictReader(QUEUE.open())}
assert all(after[p["source_row"]["case_id"]]["editor_verdict"] == p["source_row"]["editor_verdict"] for p in proposals)
assert status["phase"] == "review_due"
print(json.dumps({k: snapshot[k] for k in ("reviewed_rows", "proposed_content_verdict_counts", "proposed_action_counts")}))

def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")

lines = [
    "# Jev shadow review proposals — September 23, 2026", "",
    "Agent triage of all 29 queued cases. These proposals are **not human-approved labels**; editor verdicts remain unchanged and the experiment remains paused.", "",
    "Proposed content labels: **1 violation, 2 ambiguous, 26 acceptable**. Proposed editorial actions: **1 hold/refill, 4 rewrites, 24 no content changes** (including an already-discarded intermediate DRE clue).", "",
    "The 29 rows comprise one actual threshold flag, two borderline rows, and 26 sampled unflagged rows. There are 28 final-observed rows and one discarded revision. They are neither 29 defects nor 29 independent puzzle samples.", "",
    "## Decisions worth discussing", "",
    "- **VODKA:** alcohol violation; hold the candidate for refill/replacement. Jev 0.91 exceeded 0.82, while the existing grader gave fairness 23/25 and the detector sweep returned no findings. This is one useful additional catch, subject to editor confirmation.",
    "- **NORM:** venue/character association is a policy boundary. Propose ambiguous, but use the harmless everyday sense rather than the barstool clue.",
    "- **DUEL:** recommend a neutral contest clue under the broader no-violence guidance. Firearms specifically are ambiguous; do not call 0.49 a confirmed firearms false negative.",
    "- **ETA / LES:** acceptable on the three content axes, but their answers are literally present in clue words. Flag these separately for the mechanical leak workflow; do not make them positive content labels.",
    "- **DRE / ETHAN:** accept the neutral artist/film-credit wording. The elevated drug scores do not establish that these clues refer to drug use. DRE's discarded wording was an existing-clue duplicate, not a content defect.", "",
    "## All 29 proposals", "",
    "| Answer | Clue | Queue reason | Final observed | Content verdict | Action | Reason |",
    "|---|---|---|---|---|---|---|",
]
for p in proposals:
    r = p["source_row"]
    lines.append("| " + " | ".join(cell(v) for v in (r["answer"], r["clue"], r["review_reason"], r["final_observed"], p["proposed_content_verdict"], p["proposed_action"], p["rationale"])) + " |")
lines += ["", "## Evidence and limits", "",
    "Source snapshot and per-category proposals: [proposals.json](proposals.json). Exact probabilities, grader snapshots, case IDs, and source artifact paths are retained there. Proposed notes are also in the central review.csv; no puzzle, dictionary, threshold, editor verdict, or integration decision was changed.", "",
    "Policy: bundled content_policy_v2.json; broader generation guidance in src/crossword_generator/llm/prompts/clue_generation.py. The deterministic leak detector with hgg-hard.txt returned no finding for either ETA or LES; direct word inspection confirms the contiguous substrings.", "",
    "The queue is selected by scores plus a small random-negative sample, with mostly Easy production clues and the initial mixed pilot. Do not infer population precision/recall from one positive flag or tune thresholds to these cases. Proposed rewrites still need the normal duplicate/fact/quality checks before use.", "",
    "Suggested next experiment decision: revise/extend validation after editorial decisions, including the NORM venue boundary and DUEL's distinction between firearms and broader violence. This is a recommendation only; no study decision was recorded.",
]
(HERE / "README.md").write_text("\n".join(lines) + "\n")
