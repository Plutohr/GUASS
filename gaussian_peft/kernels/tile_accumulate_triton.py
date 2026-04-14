from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on GPU hosts with Triton installed.
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None and tl is not None


def triton_tile_accumulate_is_available() -> bool:
    return TRITON_AVAILABLE


if TRITON_AVAILABLE:  # pragma: no branch
    @triton.jit
    def _tile_accumulate_kernel(
        row_coords_ptr,
        col_coords_ptr,
        mu_row_ptr,
        mu_col_ptr,
        amp_ptr,
        inv00_ptr,
        inv01_ptr,
        inv11_ptr,
        norm_ptr,
        tile_gaussian_ids_ptr,
        tile_counts_ptr,
        output_ptr,
        out_features,
        in_features,
        num_tile_cols,
        max_gaussians_per_tile,
        clamp_quad_value,
        stride_row,
        stride_col,
        stride_tile_gaussian,
        stride_output_row,
        stride_output_col,
        BLOCK_OUT: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_G: tl.constexpr,
        MAX_G: tl.constexpr,
        NORMALIZE: tl.constexpr,
        HAS_CLAMP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tile_row = pid // num_tile_cols
        tile_col = pid % num_tile_cols

        row_offsets = tile_row * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        col_offsets = tile_col * BLOCK_IN + tl.arange(0, BLOCK_IN)
        row_mask = row_offsets < out_features
        col_mask = col_offsets < in_features

        rows = tl.load(row_coords_ptr + row_offsets * stride_row, mask=row_mask, other=0.0)
        cols = tl.load(col_coords_ptr + col_offsets * stride_col, mask=col_mask, other=0.0)
        acc = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        count = tl.load(tile_counts_ptr + pid)

        for g_start in range(0, MAX_G, BLOCK_G):
            for lane in tl.static_range(0, BLOCK_G):
                gaussian_offset = g_start + lane
                active = gaussian_offset < count
                gaussian_id = tl.load(
                    tile_gaussian_ids_ptr + pid * stride_tile_gaussian + gaussian_offset,
                    mask=active,
                    other=0,
                )
                amp_val = tl.load(amp_ptr + gaussian_id, mask=active, other=0.0)
                mu_row = tl.load(mu_row_ptr + gaussian_id, mask=active, other=0.0)
                mu_col = tl.load(mu_col_ptr + gaussian_id, mask=active, other=0.0)
                inv00 = tl.load(inv00_ptr + gaussian_id, mask=active, other=0.0)
                inv01 = tl.load(inv01_ptr + gaussian_id, mask=active, other=0.0)
                inv11 = tl.load(inv11_ptr + gaussian_id, mask=active, other=0.0)
                dr = rows - mu_row
                dc = cols - mu_col
                quad = (
                    (dr[:, None] * dr[:, None]) * inv00
                    + (2.0 * dr[:, None] * dc[None, :]) * inv01
                    + (dc[None, :] * dc[None, :]) * inv11
                )
                if HAS_CLAMP:
                    quad = tl.maximum(quad, 0.0)
                    quad = tl.minimum(quad, clamp_quad_value)
                basis = tl.exp(-0.5 * quad)
                if NORMALIZE:
                    norm = tl.load(norm_ptr + gaussian_id, mask=active, other=0.0)
                    basis = basis * norm
                acc += amp_val * basis

        output_ptrs = output_ptr + row_offsets[:, None] * stride_output_row + col_offsets[None, :] * stride_output_col
        output_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(output_ptrs, acc, mask=output_mask)


def triton_accumulate_tiled_gaussian_field(
    *,
    row_coords: Tensor,
    col_coords: Tensor,
    tile_gaussian_ids: Tensor,
    tile_counts: Tensor,
    num_tile_cols: int,
    mu_compute: Tensor,
    amp_compute: Tensor,
    inv_cov: Tensor,
    det_cov: Tensor,
    tile_out: int,
    tile_in: int,
    normalize: bool,
    clamp_quad: float | None,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if row_coords.device.type != "cuda" or col_coords.device.type != "cuda":
        raise ValueError("Triton accumulation requires CUDA tensors")
    if row_coords.dtype != torch.float32 or col_coords.dtype != torch.float32:
        raise ValueError("Triton accumulation currently expects float32 coordinate axes")
    if tile_gaussian_ids.ndim != 2:
        raise ValueError("tile_gaussian_ids must have shape [num_tiles, max_gaussians_per_tile]")

    out_features = int(row_coords.shape[0])
    in_features = int(col_coords.shape[0])
    num_tiles = int(tile_gaussian_ids.shape[0])
    max_gaussians_per_tile = int(tile_gaussian_ids.shape[1]) if tile_gaussian_ids.ndim == 2 else 0
    if num_tiles == 0 or max_gaussians_per_tile == 0:
        return torch.zeros((out_features, in_features), device=row_coords.device, dtype=torch.float32)

    mu_row = mu_compute[:, 0].contiguous()
    mu_col = mu_compute[:, 1].contiguous()
    amp = amp_compute.reshape(-1).contiguous()
    inv00 = inv_cov[:, 0, 0].contiguous()
    inv01 = inv_cov[:, 0, 1].contiguous()
    inv11 = inv_cov[:, 1, 1].contiguous()
    tile_gaussian_ids_i32 = tile_gaussian_ids.to(dtype=torch.int32).contiguous()
    tile_counts_i32 = tile_counts.to(dtype=torch.int32).contiguous()

    if normalize:
        norm = 1.0 / (2.0 * torch.pi * torch.sqrt(det_cov).clamp_min(torch.finfo(mu_compute.dtype).eps))
        norm = norm.contiguous()
    else:
        norm = torch.empty(1, device=row_coords.device, dtype=torch.float32)

    output = torch.empty((out_features, in_features), device=row_coords.device, dtype=torch.float32)
    has_clamp = clamp_quad is not None
    clamp_quad_value = float(clamp_quad) if clamp_quad is not None else 0.0
    block_g = 8 if max_gaussians_per_tile >= 8 else 4 if max_gaussians_per_tile >= 4 else 1

    _tile_accumulate_kernel[(num_tiles,)](
        row_coords.contiguous(),
        col_coords.contiguous(),
        mu_row,
        mu_col,
        amp,
        inv00,
        inv01,
        inv11,
        norm,
        tile_gaussian_ids_i32,
        tile_counts_i32,
        output,
        out_features,
        in_features,
        num_tile_cols,
        max_gaussians_per_tile,
        clamp_quad_value,
        row_coords.stride(0),
        col_coords.stride(0),
        tile_gaussian_ids_i32.stride(0),
        output.stride(0),
        output.stride(1),
        BLOCK_OUT=tile_out,
        BLOCK_IN=tile_in,
        BLOCK_G=block_g,
        MAX_G=max_gaussians_per_tile,
        NORMALIZE=normalize,
        HAS_CLAMP=has_clamp,
    )
    return output
