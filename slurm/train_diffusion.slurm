#!/bin/bash
#SBATCH --job-name=gaussian-dreambooth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export INCLUDE="${INCLUDE:-}"
export LIBRARY_PATH="${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/hpc_sd_deployment}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/dreambooth_sd.yaml}"
ENV_NAME="${ENV_NAME:-gauss-sd}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_CUDA_FIELD="${BUILD_CUDA_FIELD:-1}"
PREFERRED_CUDA_HOME="${PREFERRED_CUDA_HOME:-/public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4}"
PREFERRED_MATHLIBS_HOME="${PREFERRED_MATHLIBS_HOME:-/public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4}"
USE_GCC_TOOLSET_9="${USE_GCC_TOOLSET_9:-1}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [ "$USE_GCC_TOOLSET_9" = "1" ] && [ -f /opt/rh/gcc-toolset-9/enable ]; then
  source /opt/rh/gcc-toolset-9/enable
fi

if [ -f "${PREFERRED_CUDA_HOME}/include/cuda_runtime.h" ]; then
  export CUDA_HOME="$PREFERRED_CUDA_HOME"
elif [ -f /usr/local/cuda-12.4/include/cuda_runtime.h ]; then
  export CUDA_HOME="/usr/local/cuda-12.4"
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
  export CPLUS_INCLUDE_PATH="$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi

if [ -f "${PREFERRED_MATHLIBS_HOME}/include/cusparse.h" ]; then
  export MATHLIBS_HOME="$PREFERRED_MATHLIBS_HOME"
  export CPATH="$MATHLIBS_HOME/include${CPATH:+:$CPATH}"
  export CPLUS_INCLUDE_PATH="$MATHLIBS_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  if [ -d "$MATHLIBS_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$MATHLIBS_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$MATHLIBS_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
  elif [ -d "$MATHLIBS_HOME/lib" ]; then
    export LD_LIBRARY_PATH="$MATHLIBS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$MATHLIBS_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  fi
fi

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

conda activate "$ENV_NAME"
PYTHON_BIN="$(command -v python)"

cd "$PROJECT_ROOT"
if [ "$BUILD_CUDA_FIELD" = "1" ]; then
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  PREFERRED_CUDA_HOME="${CUDA_HOME:-$PREFERRED_CUDA_HOME}" \
  PREFERRED_MATHLIBS_HOME="${MATHLIBS_HOME:-$PREFERRED_MATHLIBS_HOME}" \
  USE_GCC_TOOLSET_9="$USE_GCC_TOOLSET_9" \
  bash scripts/build_cuda_field_extension.sh
fi

"$PYTHON_BIN" scripts/train_diffusion.py --config "$CONFIG_PATH"
