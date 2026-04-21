from __future__ import annotations

from torch import nn

from gaussian_peft.layers.gaussian_linear import GaussianLinear


def freeze_non_gaussian_params(model: nn.Module, train_bias: bool = False) -> None:
    for name, param in model.named_parameters():
        if ".bias" in name:
            param.requires_grad = train_bias
            continue
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, GaussianLinear):
            module.mu_raw.requires_grad = True
            module.chol_raw.requires_grad = True
            module.amp.requires_grad = True
            module.enforce_parameter_constraints_()
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = train_bias


def mark_only_gaussian_as_trainable(
    model: nn.Module,
    train_bias: bool = False,
) -> list[str]:
    freeze_non_gaussian_params(model, train_bias=train_bias)
    return [name for name, param in model.named_parameters() if param.requires_grad]


def collect_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]
