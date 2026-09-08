#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/gemma-3-1b-it}"
export TOKENIZERS_PARALLELISM=false
export TENSORBOARD_DIR="$PROJECT_DIR/outputs/tensorboard"

cd "$PROJECT_DIR"

# verl launches vLLM and synchronizes updated model weights after every GRPO step.
exec python -m verl.trainer.main_ppo \
  --config-path "$PROJECT_DIR/configs" \
  --config-name coding \
  "$@"
