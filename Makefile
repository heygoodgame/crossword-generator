.PHONY: setup install test test-all lint format check-deps setup-ollama generate-mini generate-midi generate-themes clean

setup: install

install:
	uv sync

test:
	uv run pytest

test-all:
	uv run pytest -m ""

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

check-deps:
	@echo "Checking dependencies..."
	@printf "Python: " && python3 --version 2>/dev/null || echo "NOT FOUND"
	@printf "uv: " && uv --version 2>/dev/null || echo "NOT FOUND"
	@printf "Ollama: " && ollama --version 2>/dev/null || echo "NOT FOUND"

setup-ollama:
	bash scripts/setup_ollama.sh

generate-mini:
	uv run crossword-generator generate --type mini

generate-midi:
	uv run crossword-generator generate --type midi

generate-themes:
	uv run crossword-generator generate-themes --count 10 --size 9

clean:
	rm -rf .pytest_cache htmlcov .coverage .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f output/*.puz output/*.ipuz output/*.json
