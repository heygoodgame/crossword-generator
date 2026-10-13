# Frozen evaluation runs

See [findings](../findings-2026-09-18.md) for interpretations and label limitations.
Every directory includes a frozen corpus, model/question manifest, raw responses,
summary, report, and review CSV. `results.json` preserves all attempts; request
objects can be reconstructed from the manifest suite and corresponding corpus row.

| Run | Purpose |
|---|---|
| [jev-controls-v1](jev-controls-v1/report.md) | 68 controls × three repeats |
| [jev-historical-v1](jev-historical-v1/report.md) | 500 historical states, original verdicts withheld |
| [jev-historical-reviewed-v1](jev-historical-reviewed-v1/report.md) | Same predictions rescored on 85 reviewed states |
| [claude-controls-v1](claude-controls-v1/report.md) | Generative-model baseline, 67/68 valid |
| [claude-isolated-retry](claude-isolated-retry/report.md) | Separate retry of the invalid baseline case |
| [jev-controls-clue-only-v1](jev-controls-clue-only-v1/report.md) | Four-question clue-only variant, 68 controls |
| [jev-historical-clue-only-v1](jev-historical-clue-only-v1/report.md) | Clue-only variant, 500 historical states |
| [jev-historical-clue-only-reviewed-v1](jev-historical-clue-only-reviewed-v1/report.md) | Clue-only predictions rescored on reviewed states |

[Jev/Claude paired comparison](jev-vs-claude-controls.json) uses identical controls
and questions. Different repeat counts are averaged, not counted as extra labels.
The state ablation uses a different suite and is not a paired model comparison.

To compare a future model with the original Jev run, use the frozen corpus and suite
from the baseline (or the matching original fixtures) and a new model configuration.
Then run `compare` against `evals/classification/results/jev-controls-v1`.
Archives accept `report`, `compare`, and `rescore`; inference requires a new directory.
