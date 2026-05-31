#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Checking prerequisites"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

echo "[2/4] Installing Python dependencies"
uv sync

echo "[3/4] Generating sample data and schema registry"
uv run generate_data.py

if command -v ollama >/dev/null 2>&1; then
  echo "[4/4] Checking Ollama model"
  if ! ollama list 2>/dev/null | grep -q "qwen2.5"; then
    echo "Pulling qwen2.5 for local agent use"
    if ! ollama pull qwen2.5; then
      echo "Warning: could not pull qwen2.5 automatically. Start Ollama and run 'ollama pull qwen2.5' manually."
    fi
  fi
else
  echo "Warning: Ollama is not installed or not on PATH. Install it before running the agent."
fi

echo "Setup complete. Start the agent with: uv run main.py"