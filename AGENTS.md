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
