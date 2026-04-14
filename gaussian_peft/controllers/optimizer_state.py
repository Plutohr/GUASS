from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn


def replace_param_in_optimizer(
    optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
) -> None:
    _replace_param_reference(optimizer, old_param, new_param)
    old_state = optimizer.state.pop(old_param, {})
    optimizer.state[new_param] = _clone_state_for_replacement(old_state, new_param)


def append_param_rows(
    optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    num_new_rows: int,
) -> None:
    _replace_param_reference(optimizer, old_param, new_param)
    old_state = optimizer.state.pop(old_param, {})
    optimizer.state[new_param] = _clone_state_with_appended_rows(
        old_state=old_state,
        old_param=old_param,
        new_param=new_param,
        num_new_rows=num_new_rows,
    )


def prune_param_rows(
    optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    keep_mask: Tensor,
) -> None:
    _replace_param_reference(optimizer, old_param, new_param)
    old_state = optimizer.state.pop(old_param, {})
    optimizer.state[new_param] = _clone_state_with_pruned_rows(
        old_state=old_state,
        old_param=old_param,
        new_param=new_param,
        keep_mask=keep_mask,
    )


def _replace_param_reference(
    optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
) -> None:
    for group in optimizer.param_groups:
        params = group["params"]
        for index, param in enumerate(params):
            if param is old_param:
                params[index] = new_param


def _clone_state_for_replacement(
    old_state: dict[str, Any],
    new_param: nn.Parameter,
) -> dict[str, Any]:
    new_state: dict[str, Any] = {}
    for key, value in old_state.items():
        if torch.is_tensor(value):
            new_state[key] = value.detach().clone().to(device=new_param.device)
        else:
            new_state[key] = deepcopy(value)
    return new_state


def _clone_state_with_appended_rows(
    old_state: dict[str, Any],
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    num_new_rows: int,
) -> dict[str, Any]:
    new_state: dict[str, Any] = {}
    for key, value in old_state.items():
        if _is_row_state_tensor(value, old_param):
            pad_shape = (num_new_rows, *value.shape[1:])
            padding = torch.zeros(pad_shape, device=new_param.device, dtype=value.dtype)
            new_state[key] = torch.cat([value.detach().to(device=new_param.device), padding], dim=0)
        elif torch.is_tensor(value):
            new_state[key] = value.detach().clone().to(device=new_param.device)
        else:
            new_state[key] = deepcopy(value)
    return new_state


def _clone_state_with_pruned_rows(
    old_state: dict[str, Any],
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    keep_mask: Tensor,
) -> dict[str, Any]:
    keep_mask = keep_mask.to(device=old_param.device, dtype=torch.bool)
    new_state: dict[str, Any] = {}
    for key, value in old_state.items():
        if _is_row_state_tensor(value, old_param):
            new_state[key] = value.detach()[keep_mask].clone().to(device=new_param.device)
        elif torch.is_tensor(value):
            new_state[key] = value.detach().clone().to(device=new_param.device)
        else:
            new_state[key] = deepcopy(value)
    return new_state


def _is_row_state_tensor(value: Any, param: nn.Parameter) -> bool:
    return torch.is_tensor(value) and value.ndim >= 1 and value.shape == param.shape
