# Roadmap

## Phase 0 — Project Foundation

- [x] Project scaffolding (README, CLAUDE.md, directory structure)
- [x] Data models (`PuzzleEnvelope`, `GridSpec`, `FilledGrid`)
- [x] Configuration loading (`config.yaml` → Pydantic settings)
- [x] Dictionary module (load, filter by score, lookup)
- [x] Unit tests for dictionary and config modules

## Phase 1 — Grid Fill Pipeline

- [x] go-crossword Docker wrapper (pull image, invoke, parse output)
- [x] go-crossword compact output parser
- [x] `GridSpec` catalog for mini (5x5, 7x7) and midi (9x9–11x11)
- [x] Clue numbering utility (standard American crossword numbering)
- [x] `.puz` exporter via puzpy
- [x] `.ipuz` exporter via ipuz + crossword
- [x] `FillStep` pipeline step
- [x] Pipeline orchestration with intermediate saves
- [x] CLI wiring (`--size`, `--seed`, `--verbose`)
- [x] End-to-end: config → grid fill → export

## Phase 2 — Fill Quality Grading

- [x] Rule-based fill scorer against Jeff Chen word list
- [x] Scoring criteria: minimum word score, obscurity penalties, duplicate detection
- [x] Retry logic: if fill scores below threshold, re-run filler with different seed
- [x] 2-letter word handling for midi grids (relaxed threshold or supplemental list)

## Phase 3 — Clue Generation

- [x] Ollama LLM provider implementation
- [x] Claude LLM provider implementation (optional, via `anthropic` package)
- [x] Clue generation prompt design and templates
- [x] Structured output parsing (JSON array → ClueEntry list with retry)
- [x] Batch clue generation with context (crossing words, theme)

## Phase 4 — Clue Quality Grading

- [ ] LLM-based clue evaluation (accuracy, fairness, misdirection)
- [ ] Scoring rubric aligned with `docs/clue-quality.md`
- [ ] Clue regeneration for entries flagged as low quality
- [ ] Human-override annotations (accept/reject/edit)

## Phase 5 — Theme Generation (midi)

- [ ] LLM theme concept generation (topic, wordplay type)
- [ ] Seed entry selection from dictionary matching theme
- [ ] Revealer clue/entry generation
- [ ] Theme constraint propagation to grid filler

## Phase 6 — Full Pipeline Integration

- [ ] End-to-end orchestration (theme → fill → grade → clue → grade → export)
- [ ] Retry loops with configurable max attempts per step
- [ ] Batch generation (N puzzles per run)
- [ ] Pipeline resumption from intermediate envelopes on disk

## Phase 7 — Alternative Fillers

- N/A genxword integration (not suitable — freeform word placer, not grid autofill)
- [x] Native Python CSP solver (constraint satisfaction with backtracking + forward checking)
- [x] Filler evaluation framework (`evaluate` CLI command, markdown report)
- [x] Filler-specific dictionary preprocessing

### go-crossword Enhancements (fork in `tools/go-crossword/`)

- [x] `-format json` flag: structured JSON output instead of text rendering
- [ ] `-skip-clues` flag: N/A — go-crossword has no clue generation (purely a grid filler)
- [x] `-dictionary <path>` flag: use custom word list (Jeff Chen) instead of embedded dictionary
- [ ] `-grid-template <path>` flag: accept pre-built grid with black-square pattern

### Phase 7a — Evaluation-Driven Improvements

Findings from running the evaluation framework (100 seeds × 5/7/9/10 grids):

- [x] Preprocessed Jeff Chen dictionary for go-crossword: export a plain word-per-line file with all words below score 50 removed (~24K words vs 280K). go-crossword's algorithm can't handle the full 280K-word Jeff Chen list (90%+ timeout rate on 5x5), but a pre-filtered subset closer to its embedded dict size should fill fast while improving quality scores.
- [x] CSP solver: improve algorithm for 7x7+ grids. Added score-based value ordering, degree heuristic for MRV ties, and AC-3 arc consistency propagation.
- [x] CSP solver: configurable timeout per grid size. Added `timeout_by_size` config option (e.g., `{5: 30, 7: 120, 9: 300}`).
- [x] `GoCrosswordFiller.is_available()`: check that the Docker image actually exists locally, not just that Docker is running.
- [x] Evaluation framework: add early abort option — skip remaining seeds for a filler×size combo after N consecutive failures (e.g., 5 timeouts in a row) to avoid wasting hours on known-broken configurations.

## Phase 8 — Polish

- [ ] CI pipeline (GitHub Actions: test, lint, type check)
- [ ] Test coverage targets (≥80%)
- [ ] Human review workflow (web UI or CLI for reviewing generated puzzles)
- [ ] Puzzle preview rendering (terminal grid display, PDF export)
- [ ] Documentation site or improved README
