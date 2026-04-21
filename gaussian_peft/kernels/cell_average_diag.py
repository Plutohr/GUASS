from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian_peft.kernels.gaussian_core import resolve_effective_compute_dtype


def build_cell_edges_minus_one_to_one(
    num_bins: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    target_dtype = dtype or torch.float32
    return torch.linspace(-1.0, 1.0, steps=num_bins + 1, device=device, dtype=target_dtype)


def normal_cdf(
    value: Tensor,
    *,
    mu: Tensor,
    sigma: Tensor,
) -> Tensor:
    z = (value - mu) / (math.sqrt(2.0) * sigma)
    return 0.5 * (1.0 + torch.erf(z))


def interval_average_1d(
    left: Tensor,
    right: Tensor,
    *,
    mu: Tensor,
    sigma: Tensor,
) -> Tensor:
    width = right - left
    if torch.any(width <= 0):
        raise ValueError("interval_average_1d requires strictly positive interval widths")
    cdf_right = normal_cdf(right, mu=mu, sigma=sigma)
    cdf_left = normal_cdf(left, mu=mu, sigma=sigma)
    return (cdf_right - cdf_left) / width


def gaussian_field_cell_average_diag_v1(
    *,
    mu_raw: Tensor,
    chol_raw: Tensor,
    amp: Tensor,
    out_features: int,
    in_features: int,
    sigma_min: float,
    chunk_size: int | None = None,
    compute_dtype: torch.dtype = torch.float32,
    return_metadata: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor | float | str]]:
    if mu_raw.ndim != 2 or mu_raw.shape[-1] != 2:
        raise ValueError("mu_raw must have shape [K, 2]")
    if chol_raw.ndim != 2 or chol_raw.shape[-1] != 3:
        raise ValueError("chol_raw must have shape [K, 3]")
    if amp.ndim not in {1, 2}:
        raise ValueError("amp must have shape [K] or [K, 1]")
    if mu_raw.shape[0] != chol_raw.shape[0] or mu_raw.shape[0] != amp.shape[0]:
        raise ValueError("mu_raw, chol_raw, and amp must agree on gaussian count")
    if out_features <= 0 or in_features <= 0:
        raise ValueError("out_features and in_features must be positive")
    if sigma_min <= 0:
        raise ValueError("sigma_min must be positive")

    effective_dtype = resolve_effective_compute_dtype(compute_dtype)
    device = mu_raw.device

    mu_compute = torch.tanh(mu_raw.to(dtype=effective_dtype))
    chol_compute = chol_raw.to(dtype=effective_dtype)
    amp_compute = amp.to(dtype=effective_dtype).reshape(-1)

    mu_x = mu_compute[:, 0:1]
    mu_y = mu_compute[:, 1:2]
    sigma_x = F.softplus(chol_compute[:, 0:1]) + sigma_min
    sigma_y = F.softplus(chol_compute[:, 2:3]) + sigma_min

    x_edges = build_cell_edges_minus_one_to_one(
        in_features,
        device=device,
        dtype=effective_dtype,
    )
    y_edges = build_cell_edges_minus_one_to_one(
        out_features,
        device=device,
        dtype=effective_dtype,
    )
    x_left = x_edges[:-1].unsqueeze(0)
    x_right = x_edges[1:].unsqueeze(0)
    y_left = y_edges[:-1].unsqueeze(0)
    y_right = y_edges[1:].unsqueeze(0)

    gaussian_chunk_size = _resolve_gaussian_chunk_size(mu_raw.shape[0], chunk_size)
    delta_weight = torch.zeros(
        out_features,
        in_features,
        device=device,
        dtype=effective_dtype,
    )
    for start, end in _iter_gaussian_chunks(mu_raw.shape[0], gaussian_chunk_size):
        avg_x = interval_average_1d(
            x_left,
            x_right,
            mu=mu_x[start:end],
            sigma=sigma_x[start:end],
        )
        avg_y = interval_average_1d(
            y_left,
            y_right,
            mu=mu_y[start:end],
            sigma=sigma_y[start:end],
        )
        weighted_y = avg_y * amp_compute[start:end].unsqueeze(-1)
        delta_weight = delta_weight + weighted_y.transpose(0, 1) @ avg_x

    delta_weight = delta_weight.to(dtype=mu_raw.dtype)
    if not return_metadata:
        return delta_weight

    empty = torch.empty(0, device=device, dtype=torch.int64)
    metadata: dict[str, Tensor | float | str] = {
        "readout_scheme": "cell_average_diag_v1",
        "execution_backend": "torch_reference",
        "execution_mode": "cell_average_diag_v1_torch_reference",
        "tiles_touched": empty,
        "tile_ptr": empty,
        "gaussian_ids": empty,
        "per_tile_counts": empty,
        "num_tile_rows": 1.0,
        "num_tile_cols": 1.0,
        "sigma_x_mean": float(sigma_x.detach().mean().item()),
        "sigma_x_min": float(sigma_x.detach().min().item()),
        "sigma_x_max": float(sigma_x.detach().max().item()),
        "sigma_y_mean": float(sigma_y.detach().mean().item()),
        "sigma_y_min": float(sigma_y.detach().min().item()),
        "sigma_y_max": float(sigma_y.detach().max().item()),
    }
    return delta_weight, metadata


def _resolve_gaussian_chunk_size(num_gaussians: int, chunk_size: int | None) -> int:
    if num_gaussians <= 0:
        raise ValueError("num_gaussians must be positive")
    if chunk_size is None:
        return num_gaussians
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")
    return min(num_gaussians, chunk_size)


def _iter_gaussian_chunks(total: int, chunk_size: int):
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        yield start, end
        start = end
