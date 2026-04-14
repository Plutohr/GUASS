from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from gaussian_peft.checkpoints.state_dict import (
    export_adapter_metadata,
    gaussian_adapter_state_dict,
    load_gaussian_adapter_state_dict,
)
from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.layers.gaussian_linear import GaussianLinear


def save_adapter_checkpoint(
    path: str,
    model: nn.Module,
    target_modules: list[str],
    config: GaussianAdapterConfig,
    step: int | None = None,
) -> None:
    payload = {
        "format": "gaussian-peft-adapter",
        "step": step,
        "metadata": export_adapter_metadata(model, target_modules, config),
        "adapters": gaussian_adapter_state_dict(model),
    }
    _ensure_parent_dir(path)
    torch.save(payload, path)


def load_adapter_checkpoint(path: str, model: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adapters = payload.get("adapters", {})
    load_gaussian_adapter_state_dict(model, adapters)
    return payload


def save_full_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    step: int,
    extra_state: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format": "gaussian-peft-full",
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra_state": extra_state or {},
    }
    _ensure_parent_dir(path)
    torch.save(payload, path)


def load_full_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    replacements = _reshape_gaussian_parameters_for_full_checkpoint(model, payload["model"])
    if optimizer is not None and replacements:
        _sync_optimizer_parameter_references(optimizer, replacements)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _reshape_gaussian_parameters_for_full_checkpoint(
    model: nn.Module,
    model_state: dict[str, Any],
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    replacements: list[tuple[nn.Parameter, nn.Parameter]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, GaussianLinear):
            continue
        prefix = f"{module_name}." if module_name else ""
        mu_key = prefix + "mu_raw"
        chol_key = prefix + "chol_raw"
        amp_key = prefix + "amp"
        if mu_key not in model_state:
            continue
        if module.mu_raw.shape == model_state[mu_key].shape:
            continue
        old_mu = module.mu_raw
        old_chol = module.chol_raw
        old_amp = module.amp
        module.replace_gaussian_parameters_(
            mu_raw=model_state[mu_key],
            chol_raw=model_state[chol_key],
            amp=model_state[amp_key],
        )
        replacements.extend(
            [
                (old_mu, module.mu_raw),
                (old_chol, module.chol_raw),
                (old_amp, module.amp),
            ]
        )
    return replacements


def _sync_optimizer_parameter_references(
    optimizer,
    replacements: list[tuple[nn.Parameter, nn.Parameter]],
) -> None:
    replacement_map = {old_param: new_param for old_param, new_param in replacements}
    for group in optimizer.param_groups:
        params = group["params"]
        for index, param in enumerate(params):
            if param in replacement_map:
                params[index] = replacement_map[param]
