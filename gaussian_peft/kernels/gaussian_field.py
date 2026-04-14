from __future__ import annotations

import math

import torch
from torch import Tensor

from gaussian_peft.kernels.chunking import chunked_sum
from gaussian_peft.kernels.gaussian_core import (
    compute_gaussian_basis,
    prepare_gaussian_inputs,
    resolve_effective_compute_dtype,
    validate_gaussian_inputs,
)
from gaussian_peft.kernels.tiled_gaussian_field import tiled_gaussian_field


def gaussian_field(
    coords: Tensor,
    mu: Tensor,
    chol: Tensor,
    amp: Tensor,
    chunk_size: int | None = None,
    normalize: bool = False,
    compute_dtype: torch.dtype = torch.float32,
    clamp_quad: float | None = 80.0,
    *,
    execution_mode: str = "tiled_reference_pytorch",
    row_coords: Tensor | None = None,
    col_coords: Tensor | None = None,
    tile_out: int = 32,
    tile_in: int = 32,
    sigma_multiplier: float = 3.0,
    return_metadata: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor | float | str]]:
    validate_gaussian_inputs(coords=coords, mu=mu, chol=chol, amp=amp)
    normalized_execution_mode = _normalize_execution_mode(execution_mode)
    if normalized_execution_mode == "cuda_field_stage2_validation":
        raise ValueError(
            "execution_mode='cuda_field_stage2_validation' must be routed through "
            "gaussian_peft.cuda_field.runtime from GaussianLinear, not through "
            "gaussian_peft.kernels.gaussian_field."
        )
    if normalized_execution_mode == "tiled_reference_pytorch":
        if row_coords is None or col_coords is None:
            raise ValueError("row_coords and col_coords are required for tiled execution modes")
        return tiled_gaussian_field(
            row_coords=row_coords,
            col_coords=col_coords,
            mu=mu,
            chol=chol,
            amp=amp,
            execution_mode=normalized_execution_mode,
            tile_out=tile_out,
            tile_in=tile_in,
            sigma_multiplier=sigma_multiplier,
            normalize=normalize,
            compute_dtype=compute_dtype,
            clamp_quad=clamp_quad,
            return_metadata=return_metadata,
        )
    if normalized_execution_mode != "dense_reference":
        raise ValueError(f"Unsupported execution_mode: {execution_mode!r}")
    if chunk_size is None:
        delta = gaussian_field_full(
            coords=coords,
            mu=mu,
            chol=chol,
            amp=amp,
            normalize=normalize,
            compute_dtype=compute_dtype,
            clamp_quad=clamp_quad,
        )
    else:
        delta = gaussian_field_chunked_by_coords(
        coords=coords,
        mu=mu,
        chol=chol,
        amp=amp,
        chunk_size=chunk_size,
        normalize=normalize,
        compute_dtype=compute_dtype,
        clamp_quad=clamp_quad,
        )
    if not return_metadata:
        return delta
    empty = torch.empty(0, device=coords.device, dtype=torch.int64)
    metadata: dict[str, Tensor | float | str] = {
        "execution_mode": "dense_reference",
        "tiles_touched": empty,
        "tile_ptr": empty,
        "gaussian_ids": empty,
        "per_tile_counts": empty,
        "num_tile_rows": 1.0,
        "num_tile_cols": 1.0,
    }
    return delta, metadata


def gaussian_field_full(
    coords: Tensor,
    mu: Tensor,
    chol: Tensor,
    amp: Tensor,
    normalize: bool,
    compute_dtype: torch.dtype,
    clamp_quad: float | None,
) -> Tensor:
    prepared = prepare_gaussian_inputs(
        coords=coords,
        mu=mu,
        chol=chol,
        amp=amp,
        compute_dtype=compute_dtype,
    )
    basis = compute_gaussian_basis(
        coords_compute=prepared["coords"],
        mu_compute=prepared["mu"],
        inv_cov=prepared["inv_cov"],
        normalize=normalize,
        det_cov=prepared["det_cov"],
        compute_dtype=prepared["effective_dtype"],
        clamp_quad=clamp_quad,
    )
    delta = (prepared["amp"].reshape(-1, 1) * basis).sum(dim=0)
    return delta.to(dtype=coords.dtype)


def gaussian_field_chunked_by_coords(
    coords: Tensor,
    mu: Tensor,
    chol: Tensor,
    amp: Tensor,
    chunk_size: int,
    normalize: bool,
    compute_dtype: torch.dtype,
    clamp_quad: float | None,
) -> Tensor:
    prepared = prepare_gaussian_inputs(
        coords=coords,
        mu=mu,
        chol=chol,
        amp=amp,
        compute_dtype=compute_dtype,
    )
    gaussian_chunk_size = _resolve_gaussian_chunk_size(
        num_gaussians=prepared["mu"].shape[0],
        coord_chunk_size=chunk_size,
    )

    def compute_chunk(coord_chunk: Tensor, _start: int, _end: int) -> Tensor:
        delta_chunk: Tensor | None = None
        for gaussian_start, gaussian_end in _iter_gaussian_chunks(
            prepared["mu"].shape[0],
            gaussian_chunk_size,
        ):
            basis = compute_gaussian_basis(
                coords_compute=coord_chunk,
                mu_compute=prepared["mu"][gaussian_start:gaussian_end],
                inv_cov=prepared["inv_cov"][gaussian_start:gaussian_end],
                normalize=normalize,
                det_cov=prepared["det_cov"][gaussian_start:gaussian_end],
                compute_dtype=prepared["effective_dtype"],
                clamp_quad=clamp_quad,
            )
            delta_part = (
                prepared["amp"][gaussian_start:gaussian_end].reshape(-1, 1) * basis
            ).sum(dim=0)
            delta_chunk = delta_part if delta_chunk is None else delta_chunk + delta_part
        if delta_chunk is None:
            raise ValueError("gaussian count must be positive")
        return delta_chunk.to(dtype=coords.dtype)

    return chunked_sum(prepared["coords"], chunk_size, compute_chunk)


def estimate_basis_elements(
    num_coords: int,
    num_gaussians: int,
    chunk_size: int | None = None,
) -> int:
    if num_coords < 0 or num_gaussians < 0:
        raise ValueError("num_coords and num_gaussians must be non-negative")
    if chunk_size is None:
        return num_coords * num_gaussians
    return min(num_coords, chunk_size) * num_gaussians


def _normalize_execution_mode(execution_mode: str) -> str:
    aliases = {
        "dense": "dense_reference",
        "tiled": "tiled_reference_pytorch",
        "tiled_pytorch": "tiled_reference_pytorch",
        "tiled_triton_forward": "tiled_reference_pytorch",
    }
    return aliases.get(execution_mode, execution_mode)


def _resolve_gaussian_chunk_size(
    num_gaussians: int,
    coord_chunk_size: int,
) -> int:
    if num_gaussians <= 0:
        raise ValueError("num_gaussians must be positive")
    target = min(coord_chunk_size, 64)
    return max(16, min(num_gaussians, target))


def _iter_gaussian_chunks(total: int, chunk_size: int):
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        yield start, end
        start = end
