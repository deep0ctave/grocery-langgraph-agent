#!/usr/bin/env sh

set -eu

MODEL_NAME="${MODEL_NAME:-qwen2.5}"
MODEL_NAME_WITHOUT_TAG=$(printf '%s' "$MODEL_NAME" | cut -d: -f1)
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

log() {
  printf '%s\n' "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Error: '$1' is required but not installed."
    exit 1
  fi
}

install_uv_if_missing() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  log "uv not found. Installing uv with pip..."
  require_cmd python3

  if ! python3 -m pip install --user uv; then
    log "Error: failed to install uv."
    exit 1
  fi

  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    log "Error: uv installed but not found in PATH."
    log "Add \$HOME/.local/bin to your PATH and rerun this script."
    exit 1
  fi
}

setup_project() {
  cd "$SCRIPT_DIR"
  install_uv_if_missing
  require_cmd ollama

  log "Syncing Python dependencies..."
  uv sync

  log "Ensuring Ollama model '$MODEL_NAME' is available..."
  # Normalize installed model names by dropping optional tags (e.g., ":latest").
  normalized_model_names=$(ollama list 2>/dev/null | awk 'NR > 1 {print $1}' | cut -d: -f1)
  if ! printf '%s\n' "$normalized_model_names" | grep -Fxq "${MODEL_NAME_WITHOUT_TAG}"; then
    ollama pull "$MODEL_NAME"
  fi

  log "Generating sample data and schema..."
  uv run generate_data.py

  log ""
  log "Setup complete."
  log "Run the agent with: uv run main.py"
}

setup_project
