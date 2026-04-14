from __future__ import annotations

from torch import nn

from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.layers.gaussian_linear import GaussianLinear
from gaussian_peft.patchers.freeze import freeze_non_gaussian_params
from gaussian_peft.patchers.target_modules import collect_target_linear_names, normalize_target_modules


def convert_linear_to_gaussian(
    linear: nn.Linear,
    config: GaussianAdapterConfig,
) -> GaussianLinear:
    return GaussianLinear.from_linear(linear=linear, adapter_config=config)


def replace_target_linears(
    model: nn.Module,
    target_modules: list[str],
    config: GaussianAdapterConfig,
) -> list[str]:
    replaced: list[str] = []
    normalized_targets = normalize_target_modules(target_modules)
    target_names = set(collect_target_linear_names(model, normalized_targets))
    for module_name in sorted(target_names):
        parent_module, child_name = _resolve_parent_module(model, module_name)
        child = getattr(parent_module, child_name)
        if not isinstance(child, nn.Linear):
            continue
        setattr(parent_module, child_name, convert_linear_to_gaussian(child, config))
        replaced.append(module_name)
    return replaced


def apply_gaussian_peft(
    model: nn.Module,
    target_modules: list[str] | None,
    adapter_config: GaussianAdapterConfig,
    freeze_base: bool = True,
) -> tuple[nn.Module, list[str]]:
    adapter_config.validate()
    replaced = replace_target_linears(
        model,
        normalize_target_modules(target_modules),
        adapter_config,
    )
    if freeze_base:
        freeze_non_gaussian_params(model, train_bias=adapter_config.train_bias)
    return model, replaced


def _resolve_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if not module_name:
        raise ValueError("module_name must not be empty")
    path = module_name.split(".")
    parent = model
    for part in path[:-1]:
        parent = getattr(parent, part)
    return parent, path[-1]
