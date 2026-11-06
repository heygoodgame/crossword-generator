# xw-runs

A Claude Code mod that adds a **Crossword runs** pane for starting and watching
generator runs. Type `/xw` in a Claude Code session opened in this repo.

## What it runs

| Button | Command |
| --- | --- |
| 1 Check open days | `fill-open-days --through <date> --plan-only` |
| 2 Fill (dry run) | `fill-open-days --through <date>`, with gates and an upload dry run |
| 3 Fill + upload | the same with `--upload`, after a second press to confirm |
| 4 Generate unlimited batch | `generate-pilot-batch` with the unlimited-pool flags (no dedup or date exclusions, usage penalty 2.0, 12 novelty candidates, 6 workers); starter also excludes the NotFamilyFriendly and safety words down to 3 letters |

Games, tracks, difficulty, size and count are set in the pane. Uploading and
promoting an unlimited batch stay manual.

## How runs behave

- Each run starts detached under `caffeinate -i -s`, so it survives closing
  Claude Code and the Mac sleeping. Logs go to `output/batches/<batch-id>.xw.log`.
- Progress comes from the batch's `progress.json`. The run list is kept across
  sessions.
- Only one generating run at a time: separate generator processes can't see
  each other's grids, so parallel runs can ship duplicate puzzles.
- **Stop** sends SIGTERM to the run's process group. **Review with Claude** asks
  Claude to check the finished run (gates, passing fills, pending drafts) and
  never uploads without asking.

The generator needs `HEYGG_CROSSWORD_GENERATOR_TOKEN` (or an admin token) in
the environment Claude Code was started from.

## Developing

`claude plugin validate .claude/skills/xw-runs` checks the manifest and hooks
module. Edits reload in a running session.
