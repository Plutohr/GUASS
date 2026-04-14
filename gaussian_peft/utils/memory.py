from __future__ import annotations

import torch


def get_peak_memory_mb(device: torch.device | str) -> float | None:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return None
    index = target.index if target.index is not None else torch.cuda.current_device()
    return float(torch.cuda.max_memory_allocated(index) / (1024.0 * 1024.0))
