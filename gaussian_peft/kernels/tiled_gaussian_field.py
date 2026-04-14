from __future__ import annotations

import torch
from torch import Tensor

from gaussian_peft.kernels.bounds import TileBounds, compute_gaussian_tile_bounds
from gaussian_peft.kernels.tile_accumulate import accumulate_tiled_gaussian_field
from gaussian_peft.kernels.tile_index import TileIndex, build_tile_index


def tiled_gaussian_field(
    row_coords: Tensor,
    col_coords: Tensor,
    mu: Tensor,
    chol: Tensor,
    amp: Tensor,
    execution_mode: str,
    tile_out: int,
    tile_in: int,
    sigma_multiplier: float,
    normalize: bool = False,
    compute_dtype: torch.dtype = torch.float32,
    clamp_quad: float | None = 80.0,
    return_metadata: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor | int]]:
    bounds = compute_gaussian_tile_bounds(
        mu=mu,
        chol=chol,
        out_features=int(row_coords.shape[0]),
        in_features=int(col_coords.shape[0]),
        tile_out=tile_out,
        tile_in=tile_in,
        sigma_multiplier=sigma_multiplier,
    )
    tile_index = build_tile_index(bounds)
    delta, backend = accumulate_tiled_gaussian_field(
        row_coords=row_coords,
        col_coords=col_coords,
        mu=mu,
        chol=chol,
        amp=amp,
        execution_mode=execution_mode,
        tile_index=tile_index,
        tile_out=tile_out,
        tile_in=tile_in,
        normalize=normalize,
        compute_dtype=compute_dtype,
        clamp_quad=clamp_quad,
    )
    if not return_metadata:
        return delta
    metadata = build_tiled_metadata(
        bounds,
        tile_index,
        backend=backend,
        execution_mode=execution_mode,
    )
    return delta, metadata


def build_tiled_metadata(
    bounds: TileBounds,
    tile_index: TileIndex,
    *,
    backend: str,
    execution_mode: str,
) -> dict[str, Tensor | int | str]:
    per_tile_counts = tile_index.tile_ptr[1:] - tile_index.tile_ptr[:-1]
    return {
        "execution_mode": execution_mode,
        "accumulate_backend": backend,
        "tiles_touched": bounds.tiles_touched,
        "tile_ptr": tile_index.tile_ptr,
        "gaussian_ids": tile_index.gaussian_ids,
        "per_tile_counts": per_tile_counts,
        "num_tile_rows": tile_index.num_tile_rows,
        "num_tile_cols": tile_index.num_tile_cols,
    }
