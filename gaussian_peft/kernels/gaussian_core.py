from __future__ import annotations

import math

import torch
from torch import Tensor

from gaussian_peft.kernels.covariance import inverse_covariance_from_cholesky


def validate_gaussian_inputs(coords: Tensor, mu: Tensor, chol: Tensor, amp: Tensor) -> None:
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [M, 2]")
    if mu.ndim != 2 or mu.shape[-1] != 2:
        raise ValueError("mu must have shape [K, 2]")
    if chol.ndim != 3 or chol.shape[-2:] != (2, 2):
        raise ValueError("chol must have shape [K, 2, 2]")
    if amp.ndim not in {1, 2}:
        raise ValueError("amp must have shape [K] or [K, 1]")
    if mu.shape[0] != chol.shape[0] or mu.shape[0] != amp.shape[0]:
        raise ValueError("coords, mu, chol, and amp disagree on gaussian count")


def resolve_effective_compute_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    if compute_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return compute_dtype


def prepare_gaussian_inputs(
    coords: Tensor,
    mu: Tensor,
    chol: Tensor,
    amp: Tensor,
    compute_dtype: torch.dtype,
) -> dict[str, Tensor]:
    effective_dtype = resolve_effective_compute_dtype(compute_dtype)
    coords_compute = coords.to(dtype=effective_dtype)
    mu_compute = mu.to(dtype=effective_dtype)
    chol_compute = chol.to(dtype=effective_dtype)
    amp_compute = amp.to(dtype=effective_dtype)
    cov = chol_compute @ chol_compute.transpose(-1, -2)
    inv_cov = inverse_covariance_from_cholesky(chol_compute)
    det_cov = torch.linalg.det(cov)
    return {
        "coords": coords_compute,
        "mu": mu_compute,
        "amp": amp_compute,
        "inv_cov": inv_cov,
        "det_cov": det_cov,
        "effective_dtype": effective_dtype,
    }


def compute_gaussian_basis(
    coords_compute: Tensor,
    mu_compute: Tensor,
    inv_cov: Tensor,
    normalize: bool,
    det_cov: Tensor,
    compute_dtype: torch.dtype,
    clamp_quad: float | None,
) -> Tensor:
    diff = coords_compute.unsqueeze(0) - mu_compute.unsqueeze(1)
    quad = torch.einsum("kmd,kde,kme->km", diff, inv_cov, diff)
    quad = apply_quad_clamp(quad, clamp_quad)
    basis = torch.exp(-0.5 * quad)

    if normalize:
        norm = 1.0 / (
            2.0
            * math.pi
            * torch.sqrt(det_cov).clamp_min(torch.finfo(compute_dtype).eps)
        )
        basis = basis * norm.unsqueeze(-1)
    return basis


def apply_quad_clamp(quad: Tensor, clamp_quad: float | None) -> Tensor:
    if clamp_quad is None:
        return quad
    return quad.clamp(min=0.0, max=clamp_quad)
