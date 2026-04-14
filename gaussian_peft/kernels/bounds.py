from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class TileBounds:
    row_min: Tensor
    row_max: Tensor
    col_min: Tensor
    col_max: Tensor
    tile_r0: Tensor
    tile_r1: Tensor
    tile_c0: Tensor
    tile_c1: Tensor
    tiles_touched: Tensor
    num_tile_rows: int
    num_tile_cols: int


def compute_gaussian_tile_bounds(
    mu: Tensor,
    chol: Tensor,
    out_features: int,
    in_features: int,
    tile_out: int,
    tile_in: int,
    sigma_multiplier: float,
) -> TileBounds:
    if out_features <= 0 or in_features <= 0:
        raise ValueError("out_features and in_features must be positive")
    if tile_out <= 0 or tile_in <= 0:
        raise ValueError("tile_out and tile_in must be positive")
    if sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be positive")

    sigma = chol.to(dtype=torch.float32) @ chol.to(dtype=torch.float32).transpose(-1, -2)
    std_row = torch.sqrt(torch.clamp(sigma[:, 0, 0], min=1e-8))
    std_col = torch.sqrt(torch.clamp(sigma[:, 1, 1], min=1e-8))

    row_center = (mu[:, 0].to(dtype=torch.float32) + 1.0) * 0.5 * (out_features - 1)
    col_center = (mu[:, 1].to(dtype=torch.float32) + 1.0) * 0.5 * (in_features - 1)
    row_radius = sigma_multiplier * std_row * 0.5 * max(out_features - 1, 1)
    col_radius = sigma_multiplier * std_col * 0.5 * max(in_features - 1, 1)

    row_min = torch.floor(row_center - row_radius).to(dtype=torch.int64).clamp_(0, out_features - 1)
    row_max = torch.ceil(row_center + row_radius).to(dtype=torch.int64).clamp_(0, out_features - 1)
    col_min = torch.floor(col_center - col_radius).to(dtype=torch.int64).clamp_(0, in_features - 1)
    col_max = torch.ceil(col_center + col_radius).to(dtype=torch.int64).clamp_(0, in_features - 1)

    num_tile_rows = (out_features + tile_out - 1) // tile_out
    num_tile_cols = (in_features + tile_in - 1) // tile_in

    tile_r0 = (row_min // tile_out).clamp_(0, num_tile_rows - 1)
    tile_r1 = (row_max // tile_out).clamp_(0, num_tile_rows - 1)
    tile_c0 = (col_min // tile_in).clamp_(0, num_tile_cols - 1)
    tile_c1 = (col_max // tile_in).clamp_(0, num_tile_cols - 1)
    tiles_touched = (tile_r1 - tile_r0 + 1) * (tile_c1 - tile_c0 + 1)

    return TileBounds(
        row_min=row_min,
        row_max=row_max,
        col_min=col_min,
        col_max=col_max,
        tile_r0=tile_r0,
        tile_r1=tile_r1,
        tile_c0=tile_c0,
        tile_c1=tile_c1,
        tiles_touched=tiles_touched,
        num_tile_rows=num_tile_rows,
        num_tile_cols=num_tile_cols,
    )
