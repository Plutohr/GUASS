from __future__ import annotations

import torch
from torch import Tensor


def select_prune_mask(
    scores: dict[str, Tensor],
    amp: Tensor,
    min_gaussians: int,
    amp_threshold: float,
    grad_threshold: float,
) -> Tensor:
    current_num = amp.shape[0]
    if current_num <= min_gaussians:
        return torch.ones(current_num, dtype=torch.bool, device=amp.device)

    amp_score = amp.detach().abs().reshape(current_num, -1).mean(dim=1)
    grad_score = scores["grad_score"].detach().reshape(current_num)

    keep_mask = (amp_score >= amp_threshold) | (grad_score >= grad_threshold)
    keep_count = int(keep_mask.sum().item())
    if keep_count >= min_gaussians:
        return keep_mask

    priority = torch.maximum(amp_score, grad_score)
    topk_values, topk_indices = torch.topk(priority, k=min_gaussians, largest=True)
    del topk_values
    rescued = torch.zeros_like(keep_mask)
    rescued[topk_indices] = True
    return rescued


def prune_gaussians(
    mu_raw: Tensor,
    chol_raw: Tensor,
    amp: Tensor,
    keep_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    keep_mask = keep_mask.to(device=mu_raw.device, dtype=torch.bool)
    return (
        mu_raw[keep_mask].clone(),
        chol_raw[keep_mask].clone(),
        amp[keep_mask].clone(),
    )
