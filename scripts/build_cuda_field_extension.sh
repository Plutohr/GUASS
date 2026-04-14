#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_FIELD_DIR="${CUDA_FIELD_DIR:-$PROJECT_ROOT/gaussian_peft/cuda_field}"
PREFERRED_CUDA_HOME="${PREFERRED_CUDA_HOME:-}"
PREFERRED_MATHLIBS_HOME="${PREFERRED_MATHLIBS_HOME:-}"
USE_GCC_TOOLSET_9="${USE_GCC_TOOLSET_9:-1}"

if [ "$USE_GCC_TOOLSET_9" = "1" ] && [ -f /opt/rh/gcc-toolset-9/enable ]; then
  # NVIDIA HPC SDK / nvcc builds are often more stable with GCC 9 on the cluster.
  source /opt/rh/gcc-toolset-9/enable
fi

detect_cuda_home() {
  if [ -n "$PREFERRED_CUDA_HOME" ] && [ -f "${PREFERRED_CUDA_HOME}/include/cuda_runtime.h" ]; then
    printf '%s\n' "$PREFERRED_CUDA_HOME"
    return 0
  fi

  if [ -n "${CUDA_HOME:-}" ] && [ -f "${CUDA_HOME}/include/cuda_runtime.h" ]; then
    printf '%s\n' "$CUDA_HOME"
    return 0
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path candidate
    nvcc_path="$(command -v nvcc)"
    candidate="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
    if [ -f "$candidate/include/cuda_runtime.h" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  for candidate in \
    /public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda \
    /public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/11.8 \
    /usr/local/cuda-12.1 \
    /public/software/cuda \
    /public/software/cuda-12.4 \
    /public/software/cuda-12.1; do
    if [ -f "$candidate/include/cuda_runtime.h" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

detect_mathlibs_home() {
  if [ -n "$PREFERRED_MATHLIBS_HOME" ] && [ -f "${PREFERRED_MATHLIBS_HOME}/include/cusparse.h" ]; then
    printf '%s\n' "$PREFERRED_MATHLIBS_HOME"
    return 0
  fi

  for candidate in \
    /public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4 \
    /public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/11.8 \
    /usr/local/cuda-12.4/targets/x86_64-linux \
    /usr/local/cuda/targets/x86_64-linux; do
    if [ -f "$candidate/include/cusparse.h" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

CUDA_HOME_DETECTED="$(detect_cuda_home || true)"
if [ -z "$CUDA_HOME_DETECTED" ]; then
  echo "[cuda_field] unable to locate CUDA toolkit headers (cuda_runtime.h)"
  exit 1
fi

export CUDA_HOME="$CUDA_HOME_DETECTED"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MATHLIBS_HOME_DETECTED="$(detect_mathlibs_home || true)"
if [ -n "$MATHLIBS_HOME_DETECTED" ]; then
  export MATHLIBS_HOME="$MATHLIBS_HOME_DETECTED"
  export CPATH="$MATHLIBS_HOME/include${CPATH:+:$CPATH}"
  export CPLUS_INCLUDE_PATH="$MATHLIBS_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  if [ -d "$MATHLIBS_HOME/lib64" ]; then
    export LIBRARY_PATH="$MATHLIBS_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$MATHLIBS_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  elif [ -d "$MATHLIBS_HOME/lib" ]; then
    export LIBRARY_PATH="$MATHLIBS_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$MATHLIBS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

echo "[cuda_field] CUDA_HOME=$CUDA_HOME"
echo "[cuda_field] CUDACXX=$CUDACXX"
echo "[cuda_field] CC=${CC:-$(command -v gcc || true)}"
echo "[cuda_field] CXX=${CXX:-$(command -v g++ || true)}"
if [ -n "${MATHLIBS_HOME:-}" ]; then
  echo "[cuda_field] MATHLIBS_HOME=$MATHLIBS_HOME"
fi

echo "[cuda_field] Build extension"
cd "$CUDA_FIELD_DIR"
"$PYTHON_BIN" setup.py build_ext

echo "[cuda_field] Verify import"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from gaussian_peft.cuda_field.loader import load_extension

mod = load_extension(verbose=False)
print({"cuda_field_extension": getattr(mod, "__name__", "<unknown>")})
PY
