from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from torch import Tensor, nn

from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.layers.gaussian_linear import GaussianLinear

CHECKPOINT_SEMANTIC_KEYS = (
    "readout_scheme",
    "coord_domain",
    "mu_parameterization",
    "cov_parameterization",
    "gaussian_normalization",
    "component_count_normalization",
    "domain_renorm",
    "execution_backend",
)


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
    semantics = config.checkpoint_semantics()
    return {
        "target_modules": list(target_modules),
        "adapter_config": config_payload,
        "checkpoint_semantics": semantics,
        "gaussian_modules": [
            module_name
            for module_name, module in model.named_modules()
            if isinstance(module, GaussianLinear)
        ],
        **semantics,
    }


def extract_checkpoint_semantics(metadata: dict[str, Any]) -> dict[str, Any] | None:
    semantics = metadata.get("checkpoint_semantics")
    if isinstance(semantics, dict):
        return _normalize_checkpoint_semantics(semantics)
    extracted = {
        key: metadata[key]
        for key in CHECKPOINT_SEMANTIC_KEYS
        if key in metadata
    }
    if not extracted:
        return None
    return _normalize_checkpoint_semantics(extracted)


def validate_checkpoint_semantics(
    metadata: dict[str, Any],
    *,
    expected_config: GaussianAdapterConfig | None = None,
    force_legacy_load: bool = False,
) -> dict[str, Any]:
    semantics = extract_checkpoint_semantics(metadata)
    if semantics is None:
        if force_legacy_load:
            if expected_config is None:
                raise ValueError(
                    "force_legacy_load requires expected_config when checkpoint semantics are missing."
                )
            return expected_config.checkpoint_semantics()
        raise ValueError(
            "Adapter checkpoint metadata is missing checkpoint semantics. "
            "This load is rejected by default to avoid silent semantic drift. "
            "Re-run with force_legacy_load=True only if you intentionally want legacy behavior."
        )
    if expected_config is None:
        return semantics
    expected = expected_config.checkpoint_semantics()
    if semantics != expected:
        raise ValueError(
            "Adapter checkpoint semantics mismatch.\n"
            f"expected={expected}\n"
            f"actual={semantics}"
        )
    return semantics


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


def _normalize_checkpoint_semantics(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    if "domain_renorm" in normalized:
        normalized["domain_renorm"] = bool(normalized["domain_renorm"])
    normalized.setdefault("component_count_normalization", "none")
    return normalized
