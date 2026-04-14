from __future__ import annotations

import torch
from torch import Tensor


def build_linear_coords(
    out_features: int,
    in_features: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if out_features <= 0 or in_features <= 0:
        raise ValueError("out_features and in_features must be positive")

    target_dtype = dtype or torch.float32
    x = torch.linspace(-1.0, 1.0, out_features, device=device, dtype=target_dtype)
    y = torch.linspace(-1.0, 1.0, in_features, device=device, dtype=target_dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


def build_linear_axes(
    out_features: int,
    in_features: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    if out_features <= 0 or in_features <= 0:
        raise ValueError("out_features and in_features must be positive")
    target_dtype = dtype or torch.float32
    row_coords = torch.linspace(-1.0, 1.0, out_features, device=device, dtype=target_dtype)
    col_coords = torch.linspace(-1.0, 1.0, in_features, device=device, dtype=target_dtype)
    return row_coords, col_coords


def normalize_coords(coords: Tensor, mode: str = "minus_one_to_one") -> Tensor:
    if mode != "minus_one_to_one":
        raise ValueError(f"Unsupported coord normalization mode: {mode!r}")
    return coords


def reshape_delta_to_weight(delta: Tensor, out_features: int, in_features: int) -> Tensor:
    expected = out_features * in_features
    if delta.numel() != expected:
        raise ValueError(
            f"delta has {delta.numel()} elements, expected {expected} "
            f"for shape [{out_features}, {in_features}]"
        )
    return delta.reshape(out_features, in_features)
