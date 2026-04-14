from __future__ import annotations

from torch import nn

from gaussian_peft.layers.gaussian_linear import GaussianLinear

STABLE_DIFFUSION_UNET_TARGET_MODULES = ["to_q", "to_v"]


def stable_diffusion_target_modules() -> list[str]:
    return list(STABLE_DIFFUSION_UNET_TARGET_MODULES)


def normalize_target_modules(target_modules: list[str] | None) -> list[str]:
    if not target_modules:
        return stable_diffusion_target_modules()

    normalized: list[str] = []
    for name in target_modules:
        normalized.append("to_out.0" if name == "to_out" else name)
    return normalized


def is_target_module(module_name: str, target_modules: list[str]) -> bool:
    normalized = normalize_target_modules(target_modules)
    return any(module_name.endswith(target) for target in normalized)


def is_stable_diffusion_attention_linear(module_name: str) -> bool:
    return is_target_module(module_name, stable_diffusion_target_modules())


def collect_target_linear_names(
    model: nn.Module,
    target_modules: list[str],
) -> list[str]:
    normalized = normalize_target_modules(target_modules)
    matched: list[str] = []
    for module_name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and not isinstance(module, GaussianLinear)
            and is_target_module(module_name, normalized)
        ):
            matched.append(module_name)
    return matched
