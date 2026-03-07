#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL="llama3"

echo "=== Ollama Setup ==="

# Check if Ollama is installed
if ! command -v ollama &>/dev/null; then
    echo "Ollama is not installed."
    echo "Install it from: https://ollama.ai/download"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi
echo "Ollama is installed: $(ollama --version)"

# Check if Ollama is running
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "Ollama is not running. Starting it..."
    ollama serve &>/dev/null &
    sleep 2
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        echo "Failed to start Ollama. Please start it manually: ollama serve"
        exit 1
    fi
fi
echo "Ollama is running."

# Pull the default model
echo "Pulling model: ${DEFAULT_MODEL}..."
ollama pull "${DEFAULT_MODEL}"

# Verify with a test prompt
echo "Verifying model with a test prompt..."
RESPONSE=$(ollama run "${DEFAULT_MODEL}" "Say 'hello' and nothing else." 2>&1)
if [ $? -eq 0 ]; then
    echo "Model is working. Response: ${RESPONSE}"
else
    echo "Model verification failed."
    exit 1
fi

echo "=== Ollama setup complete ==="
