#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  if (return 1 2>/dev/null); then
    return 1
  fi
  exit 1
fi

# This script is intended to be sourced:
#   source scripts/activate_env.sh
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

export PROJECT_ROOT
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -d "$CUDA_HOME/bin" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

echo "activated $VENV_PATH"
echo "HF_HOME=$HF_HOME"
echo "CUDA_HOME=$CUDA_HOME"
