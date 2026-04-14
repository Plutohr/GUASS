"""Model patching utilities."""

from gaussian_peft.patchers.freeze import (
    collect_trainable_parameters,
    freeze_non_gaussian_params,
    mark_only_gaussian_as_trainable,
)
from gaussian_peft.patchers.replace_linear import (
    apply_gaussian_peft,
    convert_linear_to_gaussian,
    replace_target_linears,
)
from gaussian_peft.patchers.target_modules import (
    collect_target_linear_names,
    is_target_module,
    is_stable_diffusion_attention_linear,
    stable_diffusion_target_modules,
)

__all__ = [
    "apply_gaussian_peft",
    "collect_target_linear_names",
    "collect_trainable_parameters",
    "convert_linear_to_gaussian",
    "freeze_non_gaussian_params",
    "is_target_module",
    "is_stable_diffusion_attention_linear",
    "mark_only_gaussian_as_trainable",
    "replace_target_linears",
    "stable_diffusion_target_modules",
]
