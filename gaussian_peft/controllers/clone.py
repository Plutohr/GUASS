from __future__ import annotations

import torch
from torch import Tensor


def select_clone_indices(
    scores: dict[str, Tensor],
    max_new: int,
    grad_threshold: float,
    max_gaussians: int,
    current_num: int,
) -> Tensor:
    if max_new <= 0 or current_num >= max_gaussians:
        return torch.empty(0, dtype=torch.long)
    grad_score = scores["grad_score"].reshape(-1)
    if grad_score.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    allowed_new = min(max_new, max_gaussians - current_num)
    candidate_mask = grad_score > grad_threshold
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1)
    if candidate_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=grad_score.device)
    candidate_scores = grad_score[candidate_indices]
    topk = min(allowed_new, candidate_indices.numel())
    _, order = torch.topk(candidate_scores, k=topk, largest=True)
    return candidate_indices[order]


def clone_gaussians(
    mu_raw: Tensor,
    chol_raw: Tensor,
    amp: Tensor,
    clone_indices: Tensor,
    noise_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if clone_indices.numel() == 0:
        return mu_raw, chol_raw, amp

    clone_indices = clone_indices.to(device=mu_raw.device, dtype=torch.long)
    mu_clones = mu_raw.index_select(0, clone_indices).clone()
    if noise_scale > 0:
        mu_clones = mu_clones + torch.randn_like(mu_clones) * noise_scale

    chol_clones = chol_raw.index_select(0, clone_indices).clone()
    amp_source = amp.index_select(0, clone_indices).clone()
    amp_clones = amp_source * 0.5

    new_amp = amp.clone()
    new_amp[clone_indices] = new_amp[clone_indices] * 0.5

    mu_out = torch.cat([mu_raw, mu_clones], dim=0)
    chol_out = torch.cat([chol_raw, chol_clones], dim=0)
    amp_out = torch.cat([new_amp, amp_clones], dim=0)
    return mu_out, chol_out, amp_out
