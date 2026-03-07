# Roadmap

## Phase 0 — Project Foundation (current)

- [x] Project scaffolding (README, CLAUDE.md, directory structure)
- [ ] Data models (`PuzzleEnvelope`, `GridSpec`, `FilledGrid`)
- [x] Configuration loading (`config.yaml` → Pydantic settings)
- [x] Dictionary module (load, filter by score, lookup)
- [x] Unit tests for dictionary and config modules

## Phase 1 — Grid Fill Pipeline

- [ ] go-crossword Docker wrapper (pull image, invoke, parse output)
- [ ] `GridSpec` definitions for mini (5x5, 7x7) and midi (9x9–11x11) with black-square patterns
- [ ] `.puz` exporter via puzpy
- [ ] `.ipuz` exporter via ipuz + crossword
- [ ] End-to-end: config → grid fill → export

## Phase 2 — Fill Quality Grading

- [ ] Rule-based fill scorer against Jeff Chen word list
- [ ] Scoring criteria: minimum word score, obscurity penalties, duplicate detection
- [ ] Retry logic: if fill scores below threshold, re-run filler with different seed
- [ ] 2-letter word handling for midi grids (relaxed threshold or supplemental list)

## Phase 3 — Clue Generation

- [ ] Ollama LLM provider implementation
- [ ] Clue generation prompt design and templates
- [ ] Structured output parsing (clue text, difficulty level)
- [ ] Per-entry clue generation with context (crossing words, theme)

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

- [ ] genxword integration
- [ ] Native Python CSP solver (constraint satisfaction)
- [ ] Filler benchmarking (speed, fill quality, success rate)
- [ ] Filler-specific dictionary preprocessing

## Phase 8 — Polish

- [ ] CI pipeline (GitHub Actions: test, lint, type check)
- [ ] Test coverage targets (≥80%)
- [ ] Human review workflow (web UI or CLI for reviewing generated puzzles)
- [ ] Puzzle preview rendering (terminal grid display, PDF export)
- [ ] Documentation site or improved README
