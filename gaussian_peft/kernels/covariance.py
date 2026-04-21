from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def split_cholesky_raw(chol_raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if chol_raw.ndim != 2 or chol_raw.shape[-1] != 3:
        raise ValueError("chol_raw must have shape [K, 3]")
    return chol_raw[:, 0], chol_raw[:, 1], chol_raw[:, 2]


def activate_cholesky(chol_raw: Tensor, eps: float) -> Tensor:
    raw_11, raw_21, raw_22 = split_cholesky_raw(chol_raw)
    chol = torch.zeros(
        chol_raw.shape[0],
        2,
        2,
        device=chol_raw.device,
        dtype=chol_raw.dtype,
    )
    chol[:, 0, 0] = F.softplus(raw_11) + eps
    chol[:, 1, 0] = raw_21
    chol[:, 1, 1] = F.softplus(raw_22) + eps
    return chol


def activate_diag_sigma_from_chol_slots(
    chol_raw: Tensor,
    *,
    sigma_min: float,
) -> tuple[Tensor, Tensor]:
    raw_11, _raw_21, raw_22 = split_cholesky_raw(chol_raw)
    sigma_x = F.softplus(raw_11) + sigma_min
    sigma_y = F.softplus(raw_22) + sigma_min
    return sigma_x, sigma_y


def zero_inactive_chol_slot_(chol_raw: Tensor) -> Tensor:
    if chol_raw.ndim != 2 or chol_raw.shape[-1] != 3:
        raise ValueError("chol_raw must have shape [K, 3]")
    chol_raw[..., 1].zero_()
    return chol_raw


def covariance_from_cholesky(chol: Tensor) -> Tensor:
    if chol.ndim != 3 or chol.shape[-2:] != (2, 2):
        raise ValueError("chol must have shape [K, 2, 2]")
    return chol @ chol.transpose(-1, -2)


def inverse_covariance_from_cholesky(chol: Tensor) -> Tensor:
    chol_for_inv = _resolve_linalg_dtype(chol)
    cov = covariance_from_cholesky(chol_for_inv)
    return torch.linalg.inv(cov)


def covariance_diag_scale(chol: Tensor) -> Tensor:
    cov = covariance_from_cholesky(chol)
    return cov.diagonal(dim1=-2, dim2=-1)


def _resolve_linalg_dtype(tensor: Tensor) -> Tensor:
    # torch.linalg.inv does not support float16/bfloat16 on the current stack.
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return tensor.to(dtype=torch.float32)
    return tensor
