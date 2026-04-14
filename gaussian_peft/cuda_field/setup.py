from __future__ import annotations

import sys
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
ARTIFACTS_ROOT = ROOT.parent.parent / ".artifacts" / "cuda_field_reference"
BUILD_BASE = ARTIFACTS_ROOT / "build"
BUILD_LIB = ARTIFACTS_ROOT / "lib"
CUDA_HOME = os.environ.get("CUDA_HOME")
INCLUDE_DIRS = []
LIBRARY_DIRS = []

if CUDA_HOME:
    include_dir = Path(CUDA_HOME) / "include"
    lib64_dir = Path(CUDA_HOME) / "lib64"
    if (include_dir / "cuda_runtime.h").exists():
        INCLUDE_DIRS.append(str(include_dir))
    if lib64_dir.exists():
        LIBRARY_DIRS.append(str(lib64_dir))

if "--inplace" in sys.argv:
    raise SystemExit(
        "Do not use --inplace for cuda_field reference builds. "
        f"Use `python setup.py build_ext` and collect artifacts from {ARTIFACTS_ROOT}."
    )


setup(
    name="gaussian_peft_cuda_field_reference",
    ext_modules=[
        CUDAExtension(
            name="gaussian_peft_cuda_field_reference",
            sources=[
                str(CSRC / "ext.cpp"),
                str(CSRC / "field_impl.cu"),
                str(CSRC / "forward.cu"),
                str(CSRC / "backward.cu"),
            ],
            include_dirs=INCLUDE_DIRS,
            library_dirs=LIBRARY_DIRS,
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    options={
        "build": {"build_base": str(BUILD_BASE)},
        "build_ext": {"build_lib": str(BUILD_LIB)},
    },
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
