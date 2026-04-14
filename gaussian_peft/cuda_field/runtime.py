from __future__ import annotations

import warnings

import torch
from torch import Tensor

from gaussian_peft.cuda_field.loader import load_extension


class GaussianFieldCudaReferenceFunction(torch.autograd.Function):
    """
    Reference-only extension-backed autograd path.

    Forward runs through the CUDA extension reference implementation.
    Backward replays the saved active set and delegates derivative propagation
    to PyTorch autograd inside the extension. This path exists only for
    numerical alignment checks and must not be used as the training backend.
    """

    _backward_warning_emitted = False

    @staticmethod
    def forward(
        ctx,
        row_coords: Tensor,
        col_coords: Tensor,
        mu: Tensor,
        chol_raw: Tensor,
        amp: Tensor,
        tile_out: int,
        tile_in: int,
        sigma_multiplier: float,
        normalize: bool,
        clamp_quad: float | None,
    ) -> Tensor:
        ext = load_extension(verbose=False)
        delta, tile_ptr, gaussian_ids_sorted = ext.gaussian_field_forward(
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_out,
            tile_in,
            sigma_multiplier,
            normalize,
            clamp_quad,
        )
        ctx.save_for_backward(
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_ptr,
            gaussian_ids_sorted,
        )
        ctx.tile_out = tile_out
        ctx.tile_in = tile_in
        ctx.sigma_multiplier = sigma_multiplier
        ctx.normalize = normalize
        ctx.clamp_quad = clamp_quad
        return delta

    @staticmethod
    def backward(ctx, grad_delta: Tensor):
        if not GaussianFieldCudaReferenceFunction._backward_warning_emitted:
            warnings.warn(
                "gaussian_field_reference backward is a validation-only "
                "reference replay path. Use it only for numerical alignment checks, "
                "not for training or performance measurements.",
                stacklevel=2,
            )
            GaussianFieldCudaReferenceFunction._backward_warning_emitted = True
        ext = load_extension(verbose=False)
        row_coords, col_coords, mu, chol_raw, amp, tile_ptr, gaussian_ids_sorted = ctx.saved_tensors
        d_mu, d_chol_raw, d_amp = ext.gaussian_field_backward_reference(
            grad_delta,
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_ptr,
            gaussian_ids_sorted,
            ctx.tile_out,
            ctx.tile_in,
            ctx.sigma_multiplier,
            ctx.normalize,
            ctx.clamp_quad,
        )
        return (
            None,
            None,
            d_mu,
            d_chol_raw,
            d_amp,
            None,
            None,
            None,
            None,
            None,
        )


class GaussianFieldCudaTrainFunction(torch.autograd.Function):
    """
    Training backend autograd path.

    Forward reuses the CUDA extension forward path. Backward calls the
    extension's explicit CUDA backward path instead of replaying the graph
    through torch.autograd.grad. This is the active `cuda_field_train`
    backend. It is the main training path for the tiled CUDA implementation.
    Remaining work is limited to performance tuning and broader regression
    coverage, not correctness fallback.
    """

    _backward_warning_emitted = False

    @staticmethod
    def forward(
        ctx,
        row_coords: Tensor,
        col_coords: Tensor,
        mu: Tensor,
        chol_raw: Tensor,
        amp: Tensor,
        tile_out: int,
        tile_in: int,
        sigma_multiplier: float,
        normalize: bool,
        clamp_quad: float | None,
    ) -> Tensor:
        ext = load_extension(verbose=False)
        delta, tile_ptr, gaussian_ids_sorted = ext.gaussian_field_forward(
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_out,
            tile_in,
            sigma_multiplier,
            normalize,
            clamp_quad,
        )
        ctx.save_for_backward(
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_ptr,
            gaussian_ids_sorted,
        )
        ctx.tile_out = tile_out
        ctx.tile_in = tile_in
        ctx.sigma_multiplier = sigma_multiplier
        ctx.normalize = normalize
        ctx.clamp_quad = clamp_quad
        return delta

    @staticmethod
    def backward(ctx, grad_delta: Tensor):
        if not GaussianFieldCudaTrainFunction._backward_warning_emitted:
            warnings.warn(
                "gaussian_field_train uses the active custom CUDA backward path. "
                "This backend already runs pair-buffer partial gradients with "
                "grouped reduce-by-key. Treat current limits as performance-related "
                "rather than correctness fallback.",
                stacklevel=2,
            )
            GaussianFieldCudaTrainFunction._backward_warning_emitted = True
        ext = load_extension(verbose=False)
        row_coords, col_coords, mu, chol_raw, amp, tile_ptr, gaussian_ids_sorted = ctx.saved_tensors
        d_mu, d_chol_raw, d_amp = ext.gaussian_field_backward(
            grad_delta,
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_ptr,
            gaussian_ids_sorted,
            ctx.tile_out,
            ctx.tile_in,
            ctx.sigma_multiplier,
            ctx.normalize,
            ctx.clamp_quad,
        )
        return (
            None,
            None,
            d_mu,
            d_chol_raw,
            d_amp,
            None,
            None,
            None,
            None,
            None,
        )


def gaussian_field_forward_reference(
    *,
    row_coords: Tensor,
    col_coords: Tensor,
    mu: Tensor,
    chol_raw: Tensor,
    amp: Tensor,
    tile_out: int,
    tile_in: int,
    sigma_multiplier: float,
    normalize: bool = False,
    clamp_quad: float | None = 80.0,
    verbose_build: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    ext = load_extension(verbose=verbose_build)
    delta, tile_ptr, gaussian_ids_sorted = ext.gaussian_field_forward(
        row_coords,
        col_coords,
        mu,
        chol_raw,
        amp,
        tile_out,
        tile_in,
        sigma_multiplier,
        normalize,
        clamp_quad,
    )
    return delta, tile_ptr, gaussian_ids_sorted


def gaussian_field_reference(
    *,
    row_coords: Tensor,
    col_coords: Tensor,
    mu: Tensor,
    chol_raw: Tensor,
    amp: Tensor,
    tile_out: int,
    tile_in: int,
    sigma_multiplier: float,
    normalize: bool = False,
    clamp_quad: float | None = 80.0,
    allow_reference_backward: bool = False,
) -> Tensor:
    if (
        not allow_reference_backward
        and (mu.requires_grad or chol_raw.requires_grad or amp.requires_grad)
    ):
        raise RuntimeError(
            "gaussian_field_reference is validation-only. "
            "Set allow_reference_backward=True only for explicit gradient "
            "alignment checks; do not use this path as a training backend."
        )
    return GaussianFieldCudaReferenceFunction.apply(
        row_coords,
        col_coords,
        mu,
        chol_raw,
        amp,
        tile_out,
        tile_in,
        sigma_multiplier,
        normalize,
        clamp_quad,
    )


def gaussian_field_train(
    *,
    row_coords: Tensor,
    col_coords: Tensor,
    mu: Tensor,
    chol_raw: Tensor,
    amp: Tensor,
    tile_out: int,
    tile_in: int,
    sigma_multiplier: float,
    normalize: bool = False,
    clamp_quad: float | None = 80.0,
) -> Tensor:
    return GaussianFieldCudaTrainFunction.apply(
        row_coords,
        col_coords,
        mu,
        chol_raw,
        amp,
        tile_out,
        tile_in,
        sigma_multiplier,
        normalize,
        clamp_quad,
    )


def gaussian_field_cuda_field_stage2_validation(
    *,
    row_coords: Tensor,
    col_coords: Tensor,
    mu: Tensor,
    chol_raw: Tensor,
    amp: Tensor,
    tile_out: int,
    tile_in: int,
    sigma_multiplier: float,
    normalize: bool = False,
    clamp_quad: float | None = 80.0,
    verbose_build: bool = False,
) -> tuple[Tensor, dict[str, Tensor | float | str]]:
    delta, tile_ptr, gaussian_ids_sorted = gaussian_field_forward_reference(
        row_coords=row_coords,
        col_coords=col_coords,
        mu=mu,
        chol_raw=chol_raw,
        amp=amp,
        tile_out=tile_out,
        tile_in=tile_in,
        sigma_multiplier=sigma_multiplier,
        normalize=normalize,
        clamp_quad=clamp_quad,
        verbose_build=verbose_build,
    )
    per_tile_counts = tile_ptr[1:] - tile_ptr[:-1]
    tiles_touched = torch.full(
        (mu.size(0),),
        -1,
        device=mu.device,
        dtype=torch.int64,
    )
    metadata: dict[str, Tensor | float | str] = {
        "execution_mode": "cuda_field_stage2_validation",
        "tile_ptr": tile_ptr,
        "gaussian_ids": gaussian_ids_sorted,
        "gaussian_ids_sorted": gaussian_ids_sorted,
        "per_tile_counts": per_tile_counts,
        "tiles_touched": tiles_touched,
        "num_tile_rows": float((row_coords.numel() + tile_out - 1) // tile_out),
        "num_tile_cols": float((col_coords.numel() + tile_in - 1) // tile_in),
        "accumulate_backend": "cuda_field_stage2_validation",
    }
    return delta, metadata
