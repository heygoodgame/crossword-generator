# Crossword Generator

An automated pipeline for generating mini (5x5, 7x7) and midi (9x9–11x11) crossword puzzles. Uses pluggable grid fillers, LLM-powered clue generation, and quality grading against the Jeff Chen scored word list to produce publication-ready `.puz` and `.ipuz` files.

## Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Theme      │    │   Grid       │    │  Fill        │    │   Clue       │    │  Clue        │
│  Generation   │───▶│  Autofill    │───▶│  Quality     │───▶│  Generation  │───▶│  Quality     │
│  (midi only)  │    │              │    │  Grading     │    │              │    │  Grading     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

Each step is independently runnable and improvable. A `PuzzleEnvelope` JSON document flows through the pipeline, accumulating data at each stage. Intermediates are saved to disk for debugging and pipeline resumption.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.ai/) (local LLM for clue generation)

### Setup

```bash
make setup          # Install uv, create venv, install deps
make setup-ollama   # Install and configure Ollama with default model
```

### Generate a Puzzle

```bash
make generate-mini  # Generate a 5x5 mini crossword
make generate-midi  # Generate a 9x9 midi crossword
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust settings:

```bash
cp config.example.yaml config.yaml
```

Key settings:
- `puzzle_type` / `grid_size` — mini (5x5, 7x7) or midi (9x9–11x11)
- `fill.provider` — grid filler backend (`csp`)
- `dictionary.min_word_score` — minimum Jeff Chen score threshold (default: 50)
- `llm.model` — Ollama model for clue/theme generation (default: `llama3`)
- `output.formats` — export formats (`.puz`, `.ipuz`)

## Grid Filler

The CSP (constraint satisfaction problem) filler is a native Python constraint solver with backtracking, forward checking, and AC-3 arc consistency. It uses the Jeff Chen scored word list directly and supports configurable quality tiers and per-size timeouts.

## Dictionary

Uses the HGG Curated Crossword List (~203K entries), derived from the [Jeff Chen / XWordInfo scored word list](https://www.xwordinfo.com/). Format: `word;score` (e.g., `OCEAN;60`). Words scoring 50+ are considered acceptable fill. The dictionary module loads, filters, and provides lookup by word and score.

## Output Formats

- **`.puz`** — AcrossLite binary format (via `puzpy`), widely supported by solving apps
- **`.ipuz`** — Open JSON-based format (via `ipuz` + `crossword` libraries)

## Development

```bash
make test        # Run tests
make lint        # Lint with ruff
make format      # Auto-format with ruff
make check-deps  # Check for Python, Docker, Ollama, uv
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the phased development plan.

## Design Documents

- [Historical classification model evaluations](evals/classification/README.md)
- [Jev content-policy shadow checks](docs/jev-content-shadow.md)
- [Construction Design Principles](docs/construction-design-principles.md)
- [Fill Quality](docs/fill-quality.md)
- [Clue Quality](docs/clue-quality.md)
- [Puzzle File Formats](docs/puzzle-file-formats.md)
