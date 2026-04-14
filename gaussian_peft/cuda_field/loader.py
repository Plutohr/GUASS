from __future__ import annotations

import importlib.util
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path


CUDA_FIELD_REFERENCE_EXTENSION_NAME = "gaussian_peft_cuda_field_reference"
CUDA_FIELD_EXTENSION_NAME = CUDA_FIELD_REFERENCE_EXTENSION_NAME

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_ARTIFACTS_ROOT = _ROOT.parent.parent / ".artifacts" / "cuda_field_reference"
_BUILD = _ARTIFACTS_ROOT / "torch_extensions"
_PREBUILT_LIB = _ARTIFACTS_ROOT / "lib"

CUDA_FIELD_SOURCES = (
    _CSRC / "ext.cpp",
    _CSRC / "field_impl.cu",
    _CSRC / "forward.cu",
    _CSRC / "backward.cu",
)


def get_source_paths() -> list[str]:
    return [str(path) for path in CUDA_FIELD_SOURCES]


def build_instructions() -> str:
    return (
        "CUDA field reference path. Use torch.utils.cpp_extension.load when "
        "ninja and CUDA toolkit are available, or use `python setup.py build_ext` "
        "from gaussian_peft/cuda_field. Build artifacts are kept under "
        f"{_ARTIFACTS_ROOT}. Sources: "
        + ", ".join(get_source_paths())
    )


def _extension_suffixes() -> tuple[str, ...]:
    suffixes: list[str] = []
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if isinstance(ext_suffix, str) and ext_suffix:
        suffixes.append(ext_suffix)
    return tuple(suffixes)


def _is_abi_compatible_prebuilt(module_path: Path) -> bool:
    name = module_path.name
    return any(name.endswith(suffix) for suffix in _extension_suffixes())


def _find_prebuilt_extension() -> Path | None:
    if not _PREBUILT_LIB.exists():
        return None
    candidates = sorted(_PREBUILT_LIB.glob(f"{CUDA_FIELD_EXTENSION_NAME}*.so"))
    if not candidates:
        return None
    compatible = [path for path in candidates if _is_abi_compatible_prebuilt(path)]
    if compatible:
        return compatible[-1]
    return None


def _load_prebuilt_extension(module_path: Path):
    if CUDA_FIELD_EXTENSION_NAME in sys.modules:
        return sys.modules[CUDA_FIELD_EXTENSION_NAME]
    spec = importlib.util.spec_from_file_location(CUDA_FIELD_EXTENSION_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[CUDA_FIELD_EXTENSION_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(CUDA_FIELD_EXTENSION_NAME, None)
        raise
    return module


def _build_extension(*, verbose: bool):
    from torch.utils.cpp_extension import load

    _BUILD.mkdir(parents=True, exist_ok=True)
    return load(
        name=CUDA_FIELD_EXTENSION_NAME,
        sources=get_source_paths(),
        verbose=verbose,
        build_directory=str(_BUILD),
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
    )


@lru_cache(maxsize=1)
def load_extension(*, verbose: bool = False):
    prebuilt = _find_prebuilt_extension()
    if prebuilt is not None:
        try:
            return _load_prebuilt_extension(prebuilt)
        except Exception:
            if verbose:
                print(
                    f"[cuda_field.loader] Failed to import prebuilt extension {prebuilt}; "
                    "falling back to local build.",
                    file=sys.stderr,
                )
    return _build_extension(verbose=verbose)
