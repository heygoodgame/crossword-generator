# Jev content-policy shadow checks

Observation is off by default. Opt a batch in with `--jev-content-shadow`; it
then observes while `TYPESAFE_API_KEY` is available and the experiment is
collecting. Observing sends every answer and clue to TypeSafe. All existing batch
scope, scheduling-exclusion, dictionary-refresh, and upload rules still apply.
For other pipeline entry points, the equivalent config is:

```yaml
grading:
  clue:
    content_shadow:
      enabled: true
      max_workers: 4
      timeout_seconds: 10
```

`enabled` defaults to `false`; `true` and `auto` both opt in. Batch CLI flags
`--jev-content-shadow` and `--no-jev-content-shadow` override the config. No
setting bypasses the experiment's checkpoint pause.
Observation adds model calls and elapsed time. Missing keys,
request failures, unexpected model versions, and observation/logging exceptions
do not change the generator's repair decisions or block puzzle generation.
Failures are recorded as unknown results, or as missing artifacts if logging
itself fails; they are never counted as clean judgments.

## Status

Collection for `content-policy-v2` reached its checkpoint on September 23, 2026,
at 558 distinct final clues. The 29-case review queue was closed the same day
([record](../evals/classification/content-policy-v2/shadow-review-2026-09-23/README.md)).
No `revise`, `retire`, or `ready-for-integration` decision has been recorded, so
collection stays paused on the machine that holds that study.

## Keep the experiment moving

Every completed batch that opted in prints cumulative progress, batch coverage,
request errors, missing artifacts, extra flags, and the review path, including
when there are zero flags or collection was skipped (no key, or the checkpoint
pause). A run that did not opt in prints nothing and leaves the study files
alone. Each completed observed puzzle also updates the
central study, so large batches reach the checkpoint during generation.
`progress.json` records completed puzzle results; its phase explicitly indicates
whether the final duplicate repair sweep is still pending. `manifest.json` is the
authoritative report after the full run. Standalone generation reports progress too.

The shared local study lives under `output/content-shadow/content-policy-v2/`:
`study.sqlite3` is the cumulative record, `status.json` is the current summary,
and **`review.csv` is the single editorial queue**. These runtime files are ignored
by Git; raw batches and the checked-in pilot remain reproducible source evidence.
Keep the study directory when moving this workflow between machines: the
checkpoint count lives only there, so a clone without it starts again from zero.

```bash
# Refresh progress and preserve any edits to the central review CSV.
uv run python -m crossword_generator.content_shadow_study status

# Import an existing batch without making model calls; repeated imports are safe.
uv run python -m crossword_generator.content_shadow_study ingest output/batches/BATCH
```

The checkpoint counts **500 distinct, completely observed final answer/clue pairs**
under the frozen protocol. Copied archives, reruns of identical pairs, intermediate
revisions, failed puzzles, and incomplete predictions do not inflate that count.
The first pilot contributes 40. Puzzles already running can carry the total beyond
500; the pause applies before new puzzles start and does not interrupt generation.
Final duplicate repairs can still update observations for puzzles already observed.

The queue includes all threshold flags, below-threshold scores of at least 0.5,
and a reproducible 5% sample of otherwise unflagged final clues. The 0.5 boundary
selects cases for review; it does not change classification thresholds. NORM's
0.56 alcohol score therefore enters the queue automatically. Edit `editor_verdict`
(`violation`, `acceptable`, or `ambiguous`) and `editor_notes` there. Updates retain
those edits. Blank verdicts remain pending; agent notes do not close a case.

At the checkpoint, inspect additional flags for precision, the sampled unflagged
clues for misses, ambiguity, request failures, and added cost/latency. A small
negative sample cannot establish high recall. Choose narrowly scoped integration
only with sufficient reviewed evidence; revise the questions/thresholds and
re-evaluate on held-out labels when errors dominate, or retire the experiment if
it offers little additional value. Record the conclusion explicitly, for example:

```bash
uv run python -m crossword_generator.content_shadow_study decide revise \
  --notes "Describe reviewed evidence and the next experiment here."
```

Other decisions are `retire` and `ready-for-integration`. All keep collection
paused, and none enables repairs. A new experiment needs a deliberate versioned
protocol and its own study directory; do not clear the existing checkpoint to
continue indefinitely. `CROSSWORD_SHADOW_STUDY_DIR` can select an isolated directory
for tests or a deliberately revised experiment. `AGENTS.md` asks agents to report
this status whenever a run observes.

## Frozen policy

The bundled `resources/content_policy_v2.json` preserves the exact model, questions,
state variants, selection evidence, and thresholds from the
[200-case evaluation](../evals/classification/content-policy-v2/README.md):

| Question | State | Flag at or above |
|---|---|---:|
| Firearms angle | Answer + clue | 0.78 |
| Alcohol angle | Clue only | 0.82 |
| Recreational/illicit drugs | Answer + clue | 0.95 |

Both state variants still ask the exact three-question suite used in evaluation;
dropping unselected questions would change the experiment. Jev is pinned to
`jev-1.13.0`; unexpected resolved versions are recorded as errors. To evaluate a
new version, run the comparison framework and version the protocol explicitly.

These are three narrow checks on main clues. Hints, puzzle titles, and broader
editorial policies are outside this observer's scope.

## What is observed

The observer sees whole-puzzle grader results in the actual generator, newly
repaired clues, each convergence detector sweep, final clues (including rollbacks),
and post-batch duplicate-repair replacements. Each exact state/question combination
is called only once per puzzle and then cached, including failures. A changed clue
gets a new prediction. This preserves one observation per revision without paying
again on every sweep.

The observer reads existing findings and writes sidecar files. It does not modify
clues, grades, repair prompts, findings, envelope errors, or upload gates. Tests
exercise strong positive predictions, network/log errors, changed clues, duplicate
repair, unknown results, and preservation of reviewer notes.

## Artifacts and editorial review

- Each puzzle's intermediate directory contains `content-shadow.json`: frozen
  protocol, raw responses, usage, request state, scores, existing detector findings,
  and revision history. No request credentials are saved.
- Batch manifest items link to their observation files via `content_shadow_path`.
- `content-shadow-summary.json` reports coverage, failures, final-clue counts,
  flagged revisions, and potential flags beyond existing grading/detector checks.
- `content-shadow-review.csv` includes every revision, probabilities, original
  grading feedback, final/discarded status, and blank `editor_verdict`/`editor_notes`
  fields. Fill these with decisions such as `violation`, `acceptable`, or
  `ambiguous`. Notes explicitly marked “Agent triage” are not human verdicts.

`additional_to_grader` means Jev flagged a revision whose existing grade would
not trigger the configured grade-based repair rules. `additional_to_detector_sweep`
also checks the most recent complete detector sweep for that exact revision.
It is unknown if that revision never reached a detector sweep. Neither means a
confirmed policy violation; editor decisions are required to measure precision.
Repeated revisions are not independent labeled samples.

Reports are generated at the end of a batch and can be regenerated offline:

```bash
uv run python -m crossword_generator.content_shadow output/batches/BATCH
```

Existing editor verdicts and notes are preserved. With a batch manifest present,
only its shadow files are included; missing or unreadable files are reported.
During an unfinished run without a manifest, reporting scans available files and
marks puzzles without a final observation as unfinished.

## First live pilot: September 22, 2026

[Preserved pilot](../evals/classification/content-policy-v2/shadow-pilot-2026-09-22/README.md):
two Easy and two Hard 5x5 puzzles, generated with normal Claude clue generation,
whole-puzzle grading, fact checking, repair, naming, and hints. Local dictionaries
and no admin clue-history/schedule refresh were used for this unscheduled experiment.
Nothing was uploaded or scheduled.

- Four successful puzzles, no final pipeline errors.
- 40 final clues, 49 unique clue revisions, 98 successful Jev requests.
- No Jev threshold flags, unknown predictions, or missing observations.
- One agent-triaged editorial boundary: NORM / `Barstool regular on "Cheers"`.
  Alcohol score 0.56, below the frozen 0.82 threshold; grader fairness 21/25 and
  no detector findings. Its verdict remains open in the CSV. Thresholds were not
  changed after observing this case.

This pilot exercises the integration and review workflow. A larger, editorially
reviewed sample is needed to measure recall and marginal benefit in real batches.
