# AGENTS.md

## Repo-Local Skills

For generator batch work, dictionary updates, fill-quality rules, and data-store
uploads, load the repo-local Codex skill:

- `.codex/skills/crossword-generator/SKILL.md`
- `.codex/skills/crossword-generator/references/generator-workflow.md`

Claude has the same guidance mirrored at:

- `.claude/skills/crossword-generator/SKILL.md`
- `.claude/skills/crossword-generator/references/generator-workflow.md`

## Batch Ratio Default

When asked for a generated batch across Mini Crossword and Midi Crossword
without explicit size counts, default to a rough 5:2:7 ratio for 5x5, 7x7, and
9x9 puzzles. Midi Crossword always uses 9x9; Mini Crossword dailies are five
5x5 puzzles and two 7x7 puzzles per week.

## Jev Content-Shadow Experiment

Jev content-policy observation is opt-in and off by default. A batch observes
only with `--jev-content-shadow` (or `grading.clue.content_shadow.enabled`) and
`TYPESAFE_API_KEY` set; it sends every answer and clue to TypeSafe. It is
shadow-only: predictions must not change grading, repairs, or upload decisions.

When a run observes, its summary prints progress, coverage, errors, extra flags,
and the review CSV path. Include that status in the generation handoff, including
skipped collection or zero flags.

The study lives in `output/content-shadow/content-policy-v2/`, which Git ignores.
Check it with:

```bash
uv run python -m crossword_generator.content_shadow_study status
```

Review `review.csv` there: additional and overlapping flags, borderline scores,
and a reproducible 5% unflagged sample. Preserve editor notes; agent triage and
historical grader decisions are not human labels. Use `violation`, `acceptable`,
or `ambiguous` for editor verdicts.

At 500 successfully observed distinct final answer/clue pairs, collection pauses
before new puzzles start, even for a run that opts in. The `content-policy-v2`
study reached that checkpoint on 2026-09-23 and its `revise`, `retire`, or
`ready-for-integration` decision is still open. Do not reset the checkpoint or
delete the study to keep collecting. No decision automatically enables repair
gating. See [the experiment workflow](docs/jev-content-shadow.md) for commands
and criteria.
