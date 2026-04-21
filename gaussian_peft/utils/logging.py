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
    delta_raw_norms: list[float] = []
    component_count_scale_factors: list[float] = []
    tiles_touched_mean_values: list[float] = []
    tiles_touched_max_values: list[float] = []
    occupancy_mean_values: list[float] = []
    occupancy_max_values: list[float] = []
    grad_mu_values: list[float] = []
    grad_chol_values: list[float] = []
    grad_amp_values: list[float] = []
    sigma_x_mean_values: list[float] = []
    sigma_x_min_values: list[float] = []
    sigma_x_max_values: list[float] = []
    sigma_y_mean_values: list[float] = []
    sigma_y_min_values: list[float] = []
    sigma_y_max_values: list[float] = []

    for module in model.modules():
        if not isinstance(module, GaussianLinear):
            continue

        metadata = getattr(module, "last_forward_metadata", None) or {}
        delta_norm = metadata.get("delta_w_norm")
        if isinstance(delta_norm, (int, float)):
            delta_norms.append(float(delta_norm))
        delta_raw_norm = metadata.get("delta_w_raw_norm")
        if isinstance(delta_raw_norm, (int, float)):
            delta_raw_norms.append(float(delta_raw_norm))
        component_count_scale_factor = metadata.get("component_count_scale_factor")
        if isinstance(component_count_scale_factor, (int, float)):
            component_count_scale_factors.append(float(component_count_scale_factor))

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
            effective_chol_grad = module.project_effective_chol_tensor(module.chol_raw.grad)
            if effective_chol_grad is not None:
                grad_chol_values.append(float(effective_chol_grad.detach().float().norm().item()))
        if module.amp.grad is not None:
            grad_amp_values.append(float(module.amp.grad.detach().float().norm().item()))
        if module.adapter_config.uses_cell_average_v1():
            sigma_x, sigma_y = module.materialize_sigma()
            sigma_x_f = sigma_x.detach().float()
            sigma_y_f = sigma_y.detach().float()
            sigma_x_mean_values.append(float(sigma_x_f.mean().item()))
            sigma_x_min_values.append(float(sigma_x_f.min().item()))
            sigma_x_max_values.append(float(sigma_x_f.max().item()))
            sigma_y_mean_values.append(float(sigma_y_f.mean().item()))
            sigma_y_min_values.append(float(sigma_y_f.min().item()))
            sigma_y_max_values.append(float(sigma_y_f.max().item()))

    diagnostics: dict[str, float] = {}
    _assign_mean_max(diagnostics, "delta_w_norm", delta_norms)
    _assign_mean_max(diagnostics, "delta_w_raw_norm", delta_raw_norms)
    _assign_mean_max(
        diagnostics,
        "component_count_scale_factor",
        component_count_scale_factors,
    )
    _assign_mean_max(diagnostics, "tiles_touched", tiles_touched_mean_values, tiles_touched_max_values)
    _assign_mean_max(diagnostics, "gaussians_per_tile", occupancy_mean_values, occupancy_max_values)
    _assign_mean_max(diagnostics, "grad_mu", grad_mu_values)
    _assign_mean_max(diagnostics, "grad_chol", grad_chol_values)
    _assign_mean_max(diagnostics, "grad_amp", grad_amp_values)
    _assign_mean_min_max(diagnostics, "sigma_x", sigma_x_mean_values, sigma_x_min_values, sigma_x_max_values)
    _assign_mean_min_max(diagnostics, "sigma_y", sigma_y_mean_values, sigma_y_min_values, sigma_y_max_values)
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


def _assign_mean_min_max(
    target: dict[str, float],
    prefix: str,
    mean_values: list[float],
    min_values: list[float],
    max_values: list[float],
) -> None:
    if not mean_values:
        return
    target[f"{prefix}_mean"] = float(sum(mean_values) / len(mean_values))
    target[f"{prefix}_min"] = float(min(min_values))
    target[f"{prefix}_max"] = float(max(max_values))


def format_train_log(metrics: dict[str, float]) -> str:
    ordered = sorted(metrics.items())
    return " ".join(f"{key}={value:.6f}" for key, value in ordered)
