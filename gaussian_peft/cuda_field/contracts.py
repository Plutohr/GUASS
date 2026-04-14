from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CudaFieldStage0Thresholds:
    default_sigma_multiplier: float = 3.0
    default_forward_mean_abs_error_max: float = 3e-3
    default_forward_max_abs_error_max: float = 2e-2
    default_backward_mean_abs_error_max: float = 2e-3
    default_backward_max_abs_error_max: float = 2e-2
    partial_grad_bytes_limit: int = 256 * 1024 * 1024
    occupancy_imbalance_ratio: float = 4.0
    occupancy_max_count_floor: int = 64
    tile_work_p95_over_p50_limit: float = 3.0


@dataclass(frozen=True, slots=True)
class CudaFieldTensorContract:
    row_coords: str = "shape=[out_features], dtype=float32, device=cuda, contiguous"
    col_coords: str = "shape=[in_features], dtype=float32, device=cuda, contiguous"
    mu: str = "shape=[K, 2], dtype=float32, device=cuda, contiguous"
    chol_raw: str = "shape=[K, 3], dtype=float32, device=cuda, contiguous"
    amp: str = "shape=[K, 1], dtype=float32, device=cuda, contiguous"
    tile_ptr: str = "shape=[num_tiles + 1], dtype=int32 or int64, device=cuda, contiguous"
    gaussian_ids_sorted: str = "shape=[num_pairs], dtype=int32 or int64, device=cuda, contiguous"
    delta: str = "shape=[out_features * in_features], dtype=float32, device=cuda, contiguous"


@dataclass(frozen=True, slots=True)
class CudaFieldSavedTensorSpec:
    """Minimum persistent tensor set for the first training-capable CUDA path."""

    # Forward input references kept through ctx.save_for_backward(...)
    mu: str = "mu"
    chol_raw: str = "chol_raw"
    amp: str = "amp"
    row_coords: str = "row_coords"
    col_coords: str = "col_coords"

    # Forward-produced persistent tensors
    tile_ptr: str = "tile_ptr"
    gaussian_ids_sorted: str = "gaussian_ids_sorted"


@dataclass(frozen=True, slots=True)
class CudaFieldNonDifferentiableArgs:
    tile_out: str = "tile_out"
    tile_in: str = "tile_in"
    sigma_multiplier: str = "sigma_multiplier"
    normalize: str = "normalize"
    clamp_quad: str = "clamp_quad"


STAGE0_THRESHOLDS = CudaFieldStage0Thresholds()
TENSOR_CONTRACT = CudaFieldTensorContract()
SAVED_TENSOR_SPEC = CudaFieldSavedTensorSpec()
NON_DIFFERENTIABLE_ARGS = CudaFieldNonDifferentiableArgs()


class GaussianFieldCudaFunctionStage0(torch.autograd.Function):
    """
    Stage 0 contract only.

    Forward input order is frozen here so the future extension API does not
    drift while forward kernels and backward kernels are implemented.
    """

    @staticmethod
    def forward(
        ctx,
        row_coords: torch.Tensor,
        col_coords: torch.Tensor,
        mu: torch.Tensor,
        chol_raw: torch.Tensor,
        amp: torch.Tensor,
        tile_out: int,
        tile_in: int,
        sigma_multiplier: float,
        normalize: bool,
        clamp_quad: float | None,
    ) -> torch.Tensor:
        """
        Tensor contract:

        - row_coords: [out_features], float32, CUDA, contiguous
        - col_coords: [in_features], float32, CUDA, contiguous
        - mu: [K, 2], float32, CUDA, contiguous
        - chol_raw: [K, 3], float32, CUDA, contiguous
        - amp: [K, 1], float32, CUDA, contiguous

        Planned signature:

        Inputs:
        - row_coords: non-differentiable
        - col_coords: non-differentiable
        - mu: differentiable
        - chol_raw: differentiable
        - amp: differentiable
        - tile_out: non-differentiable
        - tile_in: non-differentiable
        - sigma_multiplier: non-differentiable
        - normalize: non-differentiable
        - clamp_quad: non-differentiable

        Frozen extension return protocol:

        extension forward must return a tensor tuple:
        (
            delta,               # [out_features * in_features], float32, CUDA, contiguous
            tile_ptr,            # [num_tiles + 1], int32/int64, CUDA, contiguous
            gaussian_ids_sorted, # [num_pairs], int32/int64, CUDA, contiguous
        )

        Python forward must expose only:
        - return value: delta
        - ctx.save_for_backward(...): row_coords, col_coords, mu, chol_raw, amp, tile_ptr, gaussian_ids_sorted
        """
        raise NotImplementedError("Stage 0 signature skeleton only.")

    @staticmethod
    def backward(ctx, grad_delta: torch.Tensor):
        """
        Planned backward return order, one-to-one with forward inputs:

        (
            None,        # row_coords
            None,        # col_coords
            d_mu,
            d_chol_raw,
            d_amp,
            None,        # tile_out
            None,        # tile_in
            None,        # sigma_multiplier
            None,        # normalize
            None,        # clamp_quad
        )

        Planned internal gradient pipeline:

        grad_delta
          -> d_amp, d_mu, d_inv_cov, d_det_cov
          -> d_cov
          -> d_chol
          -> d_chol_raw

        Frozen extension backward input protocol:
        (
            grad_delta,
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_ptr,
            gaussian_ids_sorted,
            tile_out,
            tile_in,
            sigma_multiplier,
            normalize,
            clamp_quad,
        )

        Frozen extension backward output protocol:
        (
            d_mu,         # [K, 2], float32, CUDA, contiguous
            d_chol_raw,   # [K, 3], float32, CUDA, contiguous
            d_amp,        # [K, 1], float32, CUDA, contiguous
        )
        """
        raise NotImplementedError("Stage 0 signature skeleton only.")
