from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from torch import Tensor, nn

from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.layers.gaussian_linear import GaussianLinear


def gaussian_adapter_state_dict(model: nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, GaussianLinear):
            continue
        state[module_name] = {
            "mu_raw": module.mu_raw.detach().cpu(),
            "chol_raw": module.chol_raw.detach().cpu(),
            "amp": module.amp.detach().cpu(),
            "num_gaussians": module.num_gaussians,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "has_bias": module.bias is not None,
        }
    return state


def load_gaussian_adapter_state_dict(model: nn.Module, state_dict: dict[str, Any]) -> None:
    module_map = dict(model.named_modules())
    for module_name, module_state in state_dict.items():
        if module_name not in module_map:
            raise KeyError(f"Gaussian adapter target module {module_name!r} not found in model")
        module = module_map[module_name]
        if not isinstance(module, GaussianLinear):
            raise TypeError(f"Module {module_name!r} is not a GaussianLinear")
        _validate_module_schema(module_name, module, module_state)
        module.load_adapter_state(
            {
                "mu_raw": _as_tensor(module_state["mu_raw"]),
                "chol_raw": _as_tensor(module_state["chol_raw"]),
                "amp": _as_tensor(module_state["amp"]),
            }
        )


def export_adapter_metadata(
    model: nn.Module,
    target_modules: list[str],
    config: GaussianAdapterConfig,
) -> dict[str, Any]:
    config_payload = asdict(config) if is_dataclass(config) else dict(config)
    config_payload["compute_dtype"] = str(config.compute_dtype)
    return {
        "target_modules": list(target_modules),
        "adapter_config": config_payload,
        "gaussian_modules": [
            module_name
            for module_name, module in model.named_modules()
            if isinstance(module, GaussianLinear)
        ],
    }


def _validate_module_schema(
    module_name: str,
    module: GaussianLinear,
    module_state: dict[str, Any],
) -> None:
    expected = {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "has_bias": module.bias is not None,
    }
    actual = {
        "in_features": module_state["in_features"],
        "out_features": module_state["out_features"],
        "has_bias": module_state["has_bias"],
    }
    if expected != actual:
        raise ValueError(
            f"Adapter schema mismatch for {module_name!r}: "
            f"expected {expected}, got {actual}"
        )


def _as_tensor(value: Tensor | Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    raise TypeError(f"Expected Tensor, got {type(value)!r}")
