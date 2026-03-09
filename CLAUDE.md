# CLAUDE.md

## Project Overview

Crossword Generator — an automated pipeline for generating mini (5x5, 7x7) and midi (9x9–11x11) crossword puzzles. Python 3.11+, managed with uv.

## Key Commands

```bash
make setup          # Install deps via uv
make test           # Run pytest
make lint           # Lint with ruff
make format         # Auto-format with ruff
make generate-mini  # Generate a mini crossword
make generate-midi  # Generate a midi crossword
make check-deps     # Verify Python, Docker, Ollama, uv are available
```

## Architecture

### Pipeline

Five atomic steps, each reading/writing a `PuzzleEnvelope` JSON document:

1. **Theme Generation** (midi only) — LLM generates theme concept, seed entries, revealer
2. **Grid Autofill** — pluggable filler populates the grid with words
3. **Fill Quality Grading** — scores fill against Jeff Chen word list, retries if below threshold
4. **Clue Generation** — LLM writes clues for each entry
5. **Clue Quality Grading** — LLM evaluates clue quality, regenerates flagged clues

Each step is stateless: reads a PuzzleEnvelope, adds its output, writes it back. Intermediates saved to disk for debugging and resumption.

### Pluggable Backends (all behind abstract interfaces)

- **Fillers**: `fillers/base.py` — `GridFiller` ABC. Implementations: go-crossword (Docker), genxword, future built-in CSP
- **LLM Providers**: `llm/base.py` — `LLMProvider` ABC. Implementation: Ollama
- **Exporters**: `exporters/base.py` — `Exporter` ABC. Formats: .puz (puzpy), .ipuz

### Data Flow

`PuzzleEnvelope` (Pydantic model in `models.py`) is the JSON contract between all steps. Config selects concrete implementations via `config.yaml`.

## Source Layout

```
src/crossword_generator/
├── cli.py          # Click CLI entrypoint
├── pipeline.py     # Pipeline orchestration
├── config.py       # YAML config loading
├── models.py       # PuzzleEnvelope and data models
├── dictionary.py   # Jeff Chen word list loader
├── steps/          # Pipeline step interface + implementations
├── fillers/        # Grid filler interface + implementations
├── graders/        # Quality grading implementations
├── llm/            # LLM provider interface + implementations
│   └── prompts/    # Prompt templates
└── exporters/      # Export format interface + implementations
```

## Conventions

- Python 3.11+, type hints everywhere
- Data models: Pydantic for validated models, dataclasses for simple DTOs
- Linting/formatting: ruff
- Use `logging` module, never `print()` for operational output
- Tests in `tests/`, run with `pytest`
- Package manager: uv (not pip)

## Dictionary

- File: `dictionaries/HggCuratedCrosswordList.txt`
- Format: `word;score` per line (e.g., `ocean;60`)
- ~203K entries, scores 50–60
- Minimum acceptable score: 50 (configurable)
- Words are lowercase in the file
- No 2-letter words score 50+; midi grids allow 2-letter words — needs special handling

## Domain Knowledge

- [Construction Design Principles](docs/construction-design-principles.md) — grid rules, symmetry, word length constraints
- [Fill Quality](docs/fill-quality.md) — what makes good crossword fill
- [Clue Quality](docs/clue-quality.md) — what makes good crossword clues
- [Puzzle File Formats](docs/puzzle-file-formats.md) — .puz and .ipuz specs

## LLM Integration

- Provider: Ollama at `http://localhost:11434`
- Default model: `llama3`
- Prompt templates in `src/crossword_generator/llm/prompts/`
- Used for: theme generation, clue writing, clue quality evaluation

## External Tools

- **go-crossword**: Grid filler, run via Docker. Uses its own dictionary (not Jeff Chen). Fill grading step acts as quality gate.
- **puzpy**: .puz file I/O
- **ipuz + crossword**: .ipuz file I/O

## Key Constraint

Pipeline steps are stateless. They read a PuzzleEnvelope JSON from disk, process it, and write the updated envelope back. This enables retry loops, step-level debugging, and pipeline resumption from any point.
