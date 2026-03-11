.PHONY: setup install test lint format check-deps setup-ollama generate-mini generate-midi generate-themes build-go-crossword clean

setup: install

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

check-deps:
	@echo "Checking dependencies..."
	@printf "Python: " && python3 --version 2>/dev/null || echo "NOT FOUND"
	@printf "uv: " && uv --version 2>/dev/null || echo "NOT FOUND"
	@printf "Docker: " && docker --version 2>/dev/null || echo "NOT FOUND"
	@printf "Ollama: " && ollama --version 2>/dev/null || echo "NOT FOUND"

setup-ollama:
	bash scripts/setup_ollama.sh

build-go-crossword:
	docker build -t crossword-generator/go-crossword-cli:latest \
		-f tools/go-crossword/cli/Dockerfile tools/go-crossword/

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
