from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gaussian_peft.checkpoints.io import save_full_checkpoint
from gaussian_peft.config.training import TrainingConfig
from gaussian_peft.controllers.clone import clone_gaussians, select_clone_indices
from gaussian_peft.controllers.optimizer_state import append_param_rows, prune_param_rows
from gaussian_peft.controllers.prune import prune_gaussians, select_prune_mask
from gaussian_peft.controllers.scheduler import DensifyScheduler
from gaussian_peft.controllers.stats import GaussianStatsTracker
from gaussian_peft.layers.gaussian_linear import GaussianLinear
from gaussian_peft.utils.logging import collect_gaussian_diagnostics, collect_gaussian_layer_counts
from gaussian_peft.utils.memory import get_peak_memory_mb


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        training_config: TrainingConfig,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        checkpoint_dir: str | None = None,
        device: torch.device | str | None = None,
        stats_tracker: GaussianStatsTracker | None = None,
        densify_scheduler: DensifyScheduler | None = None,
    ) -> None:
        training_config.validate()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.training_config = training_config
        self.device = torch.device(device or training_config.device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.global_step = 0
        self.stats_tracker = stats_tracker
        self.densify_scheduler = densify_scheduler
        self.last_structure_events = {"clone": 0, "prune": 0}
        self.last_densify_flags = {"clone": False, "prune": False}
        self.last_grad_clip_metrics: dict[str, float] = {}
        self.model.to(self.device)

    @abstractmethod
    def compute_loss(self, batch: Any) -> Tensor:
        raise NotImplementedError

    def prepare_batch(self, batch: Any) -> Any:
        return _move_batch_to_device(batch, self.device)

    def backward(self, loss: Tensor) -> None:
        loss.backward()
        if self.stats_tracker is not None:
            self.stats_tracker.update_from_model(self.model)

    def step_optimizer(self) -> None:
        self.optimizer.step()
        self._enforce_adapter_parameter_constraints()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def train_step(self, batch: Any) -> dict[str, float]:
        self.model.train()
        batch = self.prepare_batch(batch)
        self.zero_grad()
        loss = self.compute_loss(batch)
        if not torch.isfinite(loss).all():
            raise RuntimeError(
                f"Non-finite loss encountered at step {self.global_step + 1}: {float(loss.detach().float().item())}"
            )
        self.backward(loss)
        grad_norm = self.clip_gradients()
        if grad_norm is not None and not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError(
                f"Non-finite gradient norm encountered at step {self.global_step + 1}: {grad_norm}"
            )
        self.step_optimizer()
        self.global_step += 1
        self.last_structure_events = self.maybe_run_densify()
        self.maybe_reset_stats()
        return self.collect_metrics(loss, grad_norm=grad_norm)

    def clip_gradients(self) -> float | None:
        self.last_grad_clip_metrics = {}
        if self.training_config.uses_per_group_grad_clipping():
            return self._clip_gradients_by_optimizer_group()

        max_grad_norm = self.training_config.max_grad_norm
        if max_grad_norm is None:
            return None
        params = [param for param in self.model.parameters() if param.requires_grad and param.grad is not None]
        if not params:
            return None
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
        return float(grad_norm.detach().float().item())

    def _clip_gradients_by_optimizer_group(self) -> float | None:
        max_observed_grad_norm: float | None = None
        for group in self.optimizer.param_groups:
            group_name = str(group.get("name", "unnamed"))
            max_grad_norm = self.training_config.resolve_max_grad_norm(group_name)
            if max_grad_norm is None:
                continue
            params = [
                param
                for param in group["params"]
                if isinstance(param, nn.Parameter) and param.requires_grad and param.grad is not None
            ]
            if not params:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
            grad_norm_value = float(grad_norm.detach().float().item())
            self.last_grad_clip_metrics[f"grad_norm_{group_name}"] = grad_norm_value
            if max_observed_grad_norm is None:
                max_observed_grad_norm = grad_norm_value
            else:
                max_observed_grad_norm = max(max_observed_grad_norm, grad_norm_value)
        return max_observed_grad_norm

    def collect_metrics(self, loss: Tensor, grad_norm: float | None = None) -> dict[str, float]:
        metrics: dict[str, float] = {
            "loss": float(loss.detach().item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        metrics.update(self.last_grad_clip_metrics)
        metrics["total_gaussians"] = float(sum(collect_gaussian_layer_counts(self.model).values()))
        metrics.update(collect_gaussian_diagnostics(self.model))
        peak_memory = get_peak_memory_mb(self.device)
        if peak_memory is not None:
            metrics["peak_memory_mb"] = peak_memory
        if self.densify_scheduler is not None:
            metrics["should_clone"] = float(self.last_densify_flags["clone"])
            metrics["should_prune"] = float(self.last_densify_flags["prune"])
        metrics["clone_count"] = float(self.last_structure_events["clone"])
        metrics["prune_count"] = float(self.last_structure_events["prune"])
        return metrics

    def maybe_reset_stats(self) -> None:
        if self.stats_tracker is None or self.densify_scheduler is None:
            return
        if self.densify_scheduler.should_reset_stats(self.global_step):
            self.stats_tracker.reset_all()

    def maybe_run_densify(self) -> dict[str, int]:
        if self.stats_tracker is None or self.densify_scheduler is None:
            self.last_densify_flags = {"clone": False, "prune": False}
            return {"clone": 0, "prune": 0}

        clone_total = 0
        prune_total = 0
        clone_flag = False
        prune_flag = False
        for module_name, module in self.model.named_modules():
            if not isinstance(module, GaussianLinear):
                continue
            if module_name not in self.stats_tracker._layers:
                continue

            scores = self.stats_tracker.get_scores(module_name)
            current_num = module.num_gaussians
            config = self.densify_scheduler.config
            cloned_this_module = False

            if self.densify_scheduler.should_clone(self.global_step):
                clone_indices = select_clone_indices(
                    scores=scores,
                    max_new=config.max_new_gaussians_per_step,
                    grad_threshold=config.grad_threshold,
                    max_gaussians=config.max_gaussians_per_layer,
                    current_num=current_num,
                )
                if clone_indices.numel() > 0:
                    clone_flag = True
                    clone_total += int(clone_indices.numel())
                    self._apply_clone(module, clone_indices)
                    self.stats_tracker.register_layer(
                        module_name,
                        module.num_gaussians,
                        module.mu_raw.device,
                    )
                    cloned_this_module = True

            if self.densify_scheduler.should_prune(self.global_step) and not cloned_this_module:
                keep_mask = select_prune_mask(
                    scores=scores,
                    amp=module.amp,
                    min_gaussians=config.min_gaussians_per_layer,
                    amp_threshold=config.amp_threshold,
                    grad_threshold=config.grad_threshold,
                )
                if int(keep_mask.sum().item()) < module.num_gaussians:
                    prune_flag = True
                    prune_total += int(module.num_gaussians - int(keep_mask.sum().item()))
                    self._apply_prune(module, keep_mask)
                    self.stats_tracker.register_layer(
                        module_name,
                        module.num_gaussians,
                        module.mu_raw.device,
                    )

        self.last_densify_flags = {"clone": clone_flag, "prune": prune_flag}
        return {"clone": clone_total, "prune": prune_total}

    def _apply_clone(self, module: GaussianLinear, clone_indices: Tensor) -> None:
        old_params = module.get_gaussian_parameters()
        mu_raw, chol_raw, amp = clone_gaussians(
            mu_raw=old_params["mu_raw"].detach(),
            chol_raw=old_params["chol_raw"].detach(),
            amp=old_params["amp"].detach(),
            clone_indices=clone_indices,
            noise_scale=self.densify_scheduler.config.clone_noise_scale,
        )
        module.replace_gaussian_parameters_(mu_raw=mu_raw, chol_raw=chol_raw, amp=amp)
        new_params = module.get_gaussian_parameters()
        num_new_rows = int(clone_indices.numel())
        append_param_rows(self.optimizer, old_params["mu_raw"], new_params["mu_raw"], num_new_rows)
        append_param_rows(
            self.optimizer,
            old_params["chol_raw"],
            new_params["chol_raw"],
            num_new_rows,
        )
        append_param_rows(self.optimizer, old_params["amp"], new_params["amp"], num_new_rows)

    def _apply_prune(self, module: GaussianLinear, keep_mask: Tensor) -> None:
        old_params = module.get_gaussian_parameters()
        mu_raw, chol_raw, amp = prune_gaussians(
            mu_raw=old_params["mu_raw"].detach(),
            chol_raw=old_params["chol_raw"].detach(),
            amp=old_params["amp"].detach(),
            keep_mask=keep_mask,
        )
        module.replace_gaussian_parameters_(mu_raw=mu_raw, chol_raw=chol_raw, amp=amp)
        new_params = module.get_gaussian_parameters()
        prune_param_rows(self.optimizer, old_params["mu_raw"], new_params["mu_raw"], keep_mask)
        prune_param_rows(self.optimizer, old_params["chol_raw"], new_params["chol_raw"], keep_mask)
        prune_param_rows(self.optimizer, old_params["amp"], new_params["amp"], keep_mask)

    def save_checkpoint(self, name: str, extra_state: dict[str, Any] | None = None) -> str | None:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / name
        save_full_checkpoint(
            str(path),
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            step=self.global_step,
            extra_state=extra_state,
        )
        return str(path)

    def _enforce_adapter_parameter_constraints(self) -> None:
        for module in self.model.modules():
            if isinstance(module, GaussianLinear):
                module.enforce_parameter_constraints_()

    def fit(
        self,
        dataloader,
        max_steps: int | None = None,
        on_step: Any | None = None,
    ) -> list[dict[str, float]]:
        target_steps = max_steps or self.training_config.max_steps
        history: list[dict[str, float]] = []
        for batch in dataloader:
            if self.global_step >= target_steps:
                break
            metrics = self.train_step(batch)
            history.append(metrics)
            if on_step is not None:
                on_step(self.global_step, metrics)
        return history


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        converted = [_move_batch_to_device(value, device) for value in batch]
        return type(batch)(converted)
    return batch
