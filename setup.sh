#!/usr/bin/env sh

set -eu

MODEL_NAME="${MODEL_NAME:-qwen2.5}"
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
    log "Add \$HOME/.local/bin to your PATH and rerun ./setup.sh."
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
  if ! ollama list 2>/dev/null | grep -q "^${MODEL_NAME}"; then
    ollama pull "$MODEL_NAME"
  fi

  log "Generating sample data and schema..."
  uv run generate_data.py

  log ""
  log "Setup complete."
  log "Run the agent with: uv run main.py"
}

setup_project
