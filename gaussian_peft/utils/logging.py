from __future__ import annotations

import torch
from torch import nn

from gaussian_peft.layers.gaussian_linear import GaussianLinear


def collect_gaussian_layer_counts(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, GaussianLinear):
            counts[module_name] = module.num_gaussians
    return counts


def collect_gaussian_diagnostics(model: nn.Module) -> dict[str, float]:
    delta_norms: list[float] = []
    tiles_touched_mean_values: list[float] = []
    tiles_touched_max_values: list[float] = []
    occupancy_mean_values: list[float] = []
    occupancy_max_values: list[float] = []
    grad_mu_values: list[float] = []
    grad_chol_values: list[float] = []
    grad_amp_values: list[float] = []

    for module in model.modules():
        if not isinstance(module, GaussianLinear):
            continue

        metadata = getattr(module, "last_forward_metadata", None) or {}
        delta_norm = metadata.get("delta_w_norm")
        if isinstance(delta_norm, (int, float)):
            delta_norms.append(float(delta_norm))

        tiles_touched = metadata.get("tiles_touched")
        if isinstance(tiles_touched, torch.Tensor) and tiles_touched.numel() > 0:
            tiles = tiles_touched.detach().float()
            tiles_touched_mean_values.append(float(tiles.mean().item()))
            tiles_touched_max_values.append(float(tiles.max().item()))

        per_tile_counts = metadata.get("per_tile_counts")
        if isinstance(per_tile_counts, torch.Tensor) and per_tile_counts.numel() > 0:
            counts = per_tile_counts.detach().float()
            occupancy_mean_values.append(float(counts.mean().item()))
            occupancy_max_values.append(float(counts.max().item()))

        if module.mu_raw.grad is not None:
            grad_mu_values.append(float(module.mu_raw.grad.detach().float().norm().item()))
        if module.chol_raw.grad is not None:
            grad_chol_values.append(float(module.chol_raw.grad.detach().float().norm().item()))
        if module.amp.grad is not None:
            grad_amp_values.append(float(module.amp.grad.detach().float().norm().item()))

    diagnostics: dict[str, float] = {}
    _assign_mean_max(diagnostics, "delta_w_norm", delta_norms)
    _assign_mean_max(diagnostics, "tiles_touched", tiles_touched_mean_values, tiles_touched_max_values)
    _assign_mean_max(diagnostics, "gaussians_per_tile", occupancy_mean_values, occupancy_max_values)
    _assign_mean_max(diagnostics, "grad_mu", grad_mu_values)
    _assign_mean_max(diagnostics, "grad_chol", grad_chol_values)
    _assign_mean_max(diagnostics, "grad_amp", grad_amp_values)
    return diagnostics


def _assign_mean_max(
    target: dict[str, float],
    prefix: str,
    mean_values: list[float],
    max_values: list[float] | None = None,
) -> None:
    if not mean_values:
        return
    target[f"{prefix}_mean"] = float(sum(mean_values) / len(mean_values))
    values_for_max = max_values if max_values is not None and max_values else mean_values
    target[f"{prefix}_max"] = float(max(values_for_max))


def format_train_log(metrics: dict[str, float]) -> str:
    ordered = sorted(metrics.items())
    return " ".join(f"{key}={value:.6f}" for key, value in ordered)
