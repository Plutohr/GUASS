from __future__ import annotations

from contextlib import nullcontext

import torch


def get_compute_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype name: {name!r}")
    return mapping[normalized]


def autocast_context(device_type: str, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()
