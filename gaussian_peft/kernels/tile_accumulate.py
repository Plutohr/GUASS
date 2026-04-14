from __future__ import annotations

import torch
from torch import Tensor

from gaussian_peft.kernels.gaussian_core import (
    compute_gaussian_basis,
    prepare_gaussian_inputs,
    validate_gaussian_inputs,
)
from gaussian_peft.kernels.tile_accumulate_triton import (
    triton_accumulate_tiled_gaussian_field,
    triton_tile_accumulate_is_available,
)
from gaussian_peft.kernels.tile_index import TileIndex
from gaussian_peft.kernels.tile_index import build_padded_tile_gaussian_index


def accumulate_tiled_gaussian_field(
    row_coords: Tensor,
    col_coords: Tensor,
    mu: Tensor,
    chol: Tensor,
    amp: Tensor,
    execution_mode: str,
    tile_index: TileIndex,
    tile_out: int,
    tile_in: int,
    normalize: bool,
    compute_dtype: torch.dtype,
    clamp_quad: float | None,
) -> tuple[Tensor, str]:
    """
    Reference-only tiled accumulation helpers.

    This module is kept only as a correctness baseline for dense/tiled
    comparisons. It is not the intended training or acceleration backend for
    the mainline project. The main tiled backend should evolve under
    `gaussian_peft.cuda_field`.
    """
    if row_coords.ndim != 1 or col_coords.ndim != 1:
        raise ValueError("row_coords and col_coords must be 1D")
    if tile_out <= 0 or tile_in <= 0:
        raise ValueError("tile_out and tile_in must be positive")

    out_features = int(row_coords.shape[0])
    in_features = int(col_coords.shape[0])
    output_dtype = row_coords.dtype
    dummy_coords = torch.zeros(1, 2, device=row_coords.device, dtype=row_coords.dtype)
    validate_gaussian_inputs(dummy_coords, mu, chol, amp)
    prepared = prepare_gaussian_inputs(
        coords=dummy_coords,
        mu=mu,
        chol=chol,
        amp=amp,
        compute_dtype=compute_dtype,
    )
    mu_compute = prepared["mu"]
    amp_compute = prepared["amp"]
    inv_cov = prepared["inv_cov"]
    det_cov = prepared["det_cov"]
    effective_dtype = prepared["effective_dtype"]
    row_values = row_coords.to(dtype=effective_dtype)
    col_values = col_coords.to(dtype=effective_dtype)

    if _should_use_triton(
        execution_mode=execution_mode,
        device=row_coords.device,
        effective_dtype=effective_dtype,
        tile_index=tile_index,
        mu=mu,
        chol=chol,
        amp=amp,
    ):
        tile_gaussian_ids, tile_counts = build_padded_tile_gaussian_index(tile_index)
        delta = triton_accumulate_tiled_gaussian_field(
            row_coords=row_values.contiguous(),
            col_coords=col_values.contiguous(),
            tile_gaussian_ids=tile_gaussian_ids,
            tile_counts=tile_counts,
            num_tile_cols=tile_index.num_tile_cols,
            mu_compute=mu_compute.contiguous(),
            amp_compute=amp_compute.contiguous(),
            inv_cov=inv_cov.contiguous(),
            det_cov=det_cov.contiguous(),
            tile_out=tile_out,
            tile_in=tile_in,
            normalize=normalize,
            clamp_quad=clamp_quad,
        )
        return delta.to(dtype=output_dtype).reshape(-1), "triton"

    # Reference-only Python tile loop. Keep this path for correctness checks
    # and staged comparisons, not as a default training backend.
    delta = torch.zeros(out_features, in_features, device=row_coords.device, dtype=output_dtype)
    num_tiles = tile_index.num_tiles
    for tile_id in range(num_tiles):
        start = int(tile_index.tile_ptr[tile_id].item())
        end = int(tile_index.tile_ptr[tile_id + 1].item())
        if start >= end:
            continue

        tile_row = tile_id // tile_index.num_tile_cols
        tile_col = tile_id % tile_index.num_tile_cols
        row_start = tile_row * tile_out
        row_end = min(row_start + tile_out, out_features)
        col_start = tile_col * tile_in
        col_end = min(col_start + tile_in, in_features)

        local_row_values = row_values[row_start:row_end]
        local_col_values = col_values[col_start:col_end]
        grid_row, grid_col = torch.meshgrid(local_row_values, local_col_values, indexing="ij")
        local_coords = torch.stack((grid_row, grid_col), dim=-1).reshape(-1, 2)

        gaussian_ids = tile_index.gaussian_ids[start:end]
        basis = compute_gaussian_basis(
            coords_compute=local_coords,
            mu_compute=mu_compute.index_select(0, gaussian_ids),
            inv_cov=inv_cov.index_select(0, gaussian_ids),
            normalize=normalize,
            det_cov=det_cov.index_select(0, gaussian_ids),
            compute_dtype=effective_dtype,
            clamp_quad=clamp_quad,
        )
        local_delta = (amp_compute.index_select(0, gaussian_ids).reshape(-1, 1) * basis).sum(dim=0)
        delta[row_start:row_end, col_start:col_end] = local_delta.reshape(
            row_end - row_start,
            col_end - col_start,
        ).to(dtype=output_dtype)

    return delta.reshape(-1), "pytorch"


def _should_use_triton(
    *,
    execution_mode: str,
    device: torch.device,
    effective_dtype: torch.dtype,
    tile_index: TileIndex,
    mu: Tensor,
    chol: Tensor,
    amp: Tensor,
) -> bool:
    # Triton is frozen as a non-mainline experiment. Keep this gate only for
    # explicit side-branch validation, not as part of the primary backend path.
    if execution_mode != "tiled_triton_forward":
        return False
    if device.type != "cuda":
        return False
    if not triton_tile_accumulate_is_available():
        return False
    if effective_dtype != torch.float32:
        return False
    if tile_index.gaussian_ids.numel() == 0:
        return False
    if mu.requires_grad or chol.requires_grad or amp.requires_grad:
        raise RuntimeError(
            "execution_mode='tiled_triton_forward' is forward-only. "
            "It is not part of the mainline backend path. Use the cuda_field "
            "backend effort for training-facing work."
        )
    counts = tile_index.tile_ptr[1:] - tile_index.tile_ptr[:-1]
    if counts.numel() == 0:
        return False
    max_count = int(counts.max().item())
    mean_count = float(counts.float().mean().item())
    total_pairs = int(tile_index.gaussian_ids.numel())
    padded_pairs = counts.numel() * max_count
    if max_count > 64:
        return False
    if mean_count <= 0.0:
        return False
    if max_count > max(16, int(mean_count * 4.0)):
        return False
    if total_pairs < 512:
        return False
    if padded_pairs > total_pairs * 4:
        return False
    return True
