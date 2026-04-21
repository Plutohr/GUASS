from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class _LayerStats:
    grad_mu_accum: Tensor
    grad_cov_accum: Tensor
    grad_amp_accum: Tensor
    contrib_accum: Tensor
    active_steps: Tensor


class GaussianStatsTracker:
    def __init__(self) -> None:
        self._layers: dict[str, _LayerStats] = {}

    def register_layer(
        self,
        layer_name: str,
        num_gaussians: int,
        device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._layers[layer_name] = _LayerStats(
            grad_mu_accum=torch.zeros(num_gaussians, 1, device=device, dtype=dtype),
            grad_cov_accum=torch.zeros(num_gaussians, 1, device=device, dtype=dtype),
            grad_amp_accum=torch.zeros(num_gaussians, 1, device=device, dtype=dtype),
            contrib_accum=torch.zeros(num_gaussians, 1, device=device, dtype=dtype),
            active_steps=torch.zeros(num_gaussians, 1, device=device, dtype=dtype),
        )

    def ensure_layer(
        self,
        layer_name: str,
        num_gaussians: int,
        device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        stats = self._layers.get(layer_name)
        if stats is None or stats.grad_mu_accum.shape[0] != num_gaussians:
            self.register_layer(layer_name, num_gaussians, device, dtype=dtype)

    def update_from_layer_grads(
        self,
        layer_name: str,
        mu_grad: Tensor | None,
        chol_grad: Tensor | None,
        amp_grad: Tensor | None,
        amp_value: Tensor | None = None,
    ) -> None:
        stats = self._layers[layer_name]
        grad_mu = _columnize(compute_grad_score(mu_grad), stats.grad_mu_accum)
        grad_cov = _columnize(compute_grad_score(chol_grad), stats.grad_cov_accum)
        grad_amp = _columnize(compute_grad_score(amp_grad), stats.grad_amp_accum)

        stats.grad_mu_accum.add_(grad_mu)
        stats.grad_cov_accum.add_(grad_cov)
        stats.grad_amp_accum.add_(grad_amp)
        if amp_value is not None:
            stats.contrib_accum.add_(_columnize(compute_contrib_score(amp_value), stats.contrib_accum))
        stats.active_steps.add_(1.0)

    def update_from_model(self, model) -> None:
        for module_name, module in model.named_modules():
            if not all(hasattr(module, name) for name in ("mu_raw", "chol_raw", "amp")):
                continue
            chol_grad = module.chol_raw.grad
            project_chol = getattr(module, "project_effective_chol_tensor", None)
            if callable(project_chol):
                chol_grad = project_chol(chol_grad)
            self.ensure_layer(
                module_name,
                int(module.mu_raw.shape[0]),
                module.mu_raw.device,
                dtype=torch.float32,
            )
            self.update_from_layer_grads(
                module_name,
                mu_grad=module.mu_raw.grad,
                chol_grad=chol_grad,
                amp_grad=module.amp.grad,
                amp_value=module.amp.detach(),
            )

    def get_scores(self, layer_name: str) -> dict[str, Tensor]:
        stats = self._layers[layer_name]
        grad_score = stats.grad_mu_accum + stats.grad_cov_accum + stats.grad_amp_accum
        active = stats.active_steps.clamp_min(1.0)
        return {
            "grad_mu": stats.grad_mu_accum / active,
            "grad_cov": stats.grad_cov_accum / active,
            "grad_amp": stats.grad_amp_accum / active,
            "grad_score": grad_score / active,
            "contrib_score": stats.contrib_accum / active,
            "active_steps": stats.active_steps.clone(),
        }

    def reset_layer(self, layer_name: str) -> None:
        stats = self._layers[layer_name]
        for tensor in (
            stats.grad_mu_accum,
            stats.grad_cov_accum,
            stats.grad_amp_accum,
            stats.contrib_accum,
            stats.active_steps,
        ):
            tensor.zero_()

    def reset_all(self) -> None:
        for layer_name in list(self._layers):
            self.reset_layer(layer_name)


def compute_grad_score(grad: Tensor | None) -> Tensor:
    if grad is None:
        return torch.zeros(0, dtype=torch.float32)
    grad_detached = grad.detach().to(dtype=torch.float32)
    if grad_detached.ndim == 1:
        return grad_detached.abs()
    flat = grad_detached.reshape(grad_detached.shape[0], -1)
    return torch.linalg.vector_norm(flat, dim=1)


def compute_contrib_score(amp: Tensor | None) -> Tensor:
    if amp is None:
        return torch.zeros(0, dtype=torch.float32)
    amp_detached = amp.detach().to(dtype=torch.float32)
    flat = amp_detached.reshape(amp_detached.shape[0], -1)
    return flat.abs().mean(dim=1)


def _columnize(values: Tensor, reference: Tensor) -> Tensor:
    if values.numel() == 0:
        return torch.zeros_like(reference)
    return values.reshape(-1, 1).to(device=reference.device, dtype=reference.dtype)
