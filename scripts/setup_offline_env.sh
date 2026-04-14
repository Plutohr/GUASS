#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-gauss-sd}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_ROOT/offline_packages}"
REQ_FILE="${REQ_FILE:-$PROJECT_ROOT/requirements.offline.txt}"
TEST_SCRIPT="${TEST_SCRIPT:-$SCRIPT_DIR/test_cuda_env.py}"
BUILD_CUDA_FIELD="${BUILD_CUDA_FIELD:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/8] Check uploaded files"
test -d "$OFFLINE_DIR" || { echo "missing directory: $OFFLINE_DIR"; exit 1; }
test -f "$REQ_FILE" || { echo "missing file: $REQ_FILE"; exit 1; }

echo "[2/8] Load conda"
if [ -f /public/software/miniconda3/etc/profile.d/conda.sh ]; then
  source /public/software/miniconda3/etc/profile.d/conda.sh
elif [ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "conda not found"
  exit 1
fi

echo "[3/8] Create conda environment"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Environment already exists: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

echo "[4/8] Activate environment"
conda activate "$ENV_NAME"
PYTHON_BIN="$(command -v python)"

echo "[5/8] Install PyTorch GPU wheels offline"
"$PYTHON_BIN" -m pip install \
  --no-index \
  --find-links="$OFFLINE_DIR" \
  torch==2.4.1 \
  torchvision==0.19.1

echo "[6/8] Install remaining project dependencies offline"
"$PYTHON_BIN" -m pip install \
  --no-index \
  --find-links="$OFFLINE_DIR" \
  -r "$REQ_FILE"

echo "[7/8] Verify installation"
"$PYTHON_BIN" "$TEST_SCRIPT"

echo "[8/8] Build cuda_field extension"
if [ "$BUILD_CUDA_FIELD" = "1" ]; then
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/build_cuda_field_extension.sh"
else
  echo "Skipping cuda_field extension build (BUILD_CUDA_FIELD=$BUILD_CUDA_FIELD)"
fi
