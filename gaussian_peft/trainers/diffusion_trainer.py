from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gaussian_peft.checkpoints.io import load_full_checkpoint, save_adapter_checkpoint
from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.config.diffusion import DiffusionExperimentConfig
from gaussian_peft.controllers.scheduler import DensifyScheduler
from gaussian_peft.controllers.stats import GaussianStatsTracker
from gaussian_peft.trainers.base_trainer import BaseTrainer
from gaussian_peft.utils.diffusion import (
    RollingGradThreshold,
    build_gaussian_peft_adamw,
    collect_gaussian_layers,
)


class DiffusionTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        unet: nn.Module,
        text_encoder: nn.Module,
        vae: nn.Module,
        noise_scheduler: Any,
        config: DiffusionExperimentConfig,
    ) -> None:
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.config = config
        self.model_dtype = _resolve_mixed_precision(config.runtime.mixed_precision)
        self.gaussian_layers = collect_gaussian_layers(unet)
        optimizer = build_gaussian_peft_adamw(
            self.gaussian_layers,
            lr_amp=config.training.lr_amp,
            lr_mu=config.training.lr_mu,
            lr_cov=config.training.lr_chol,
            weight_decay=config.training.weight_decay,
        )
        print(_describe_optimizer_parameter_coverage(self.gaussian_layers, optimizer), flush=True)
        stats_tracker = GaussianStatsTracker()
        densify_scheduler = DensifyScheduler(config.densify) if config.densify.enabled else None
        super().__init__(
            model=unet,
            optimizer=optimizer,
            training_config=config.training,
            scheduler=None,
            scaler=None,
            checkpoint_dir=str(config.output_dir),
            device=config.training.device,
            stats_tracker=stats_tracker,
            densify_scheduler=densify_scheduler,
        )
        self.dynamic_threshold = RollingGradThreshold(window_size=100, scale=1.25, min_threshold=1e-5)
        self.current_grad_threshold = config.densify.grad_threshold
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        _freeze_module(self.vae)
        _freeze_module(self.text_encoder)

        if config.runtime.resume_from_checkpoint:
            load_full_checkpoint(
                config.runtime.resume_from_checkpoint,
                self.model,
                optimizer=self.optimizer,
            )
        self._apply_group_learning_rates(step=max(self.global_step, 0) + 1)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        pixel_values = batch["pixel_values"].to(device=self.device, dtype=self.vae.dtype, non_blocking=True)
        input_ids = batch["input_ids"].to(device=self.device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device, non_blocking=True)

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with _maybe_autocast(self.device.type, self.model_dtype):
            model_pred = self.model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

        per_example_loss = F.mse_loss(
            model_pred.float(),
            noise.float(),
            reduction="none",
        )
        per_example_loss = per_example_loss.reshape(per_example_loss.shape[0], -1).mean(dim=1)

        if self.config.data.with_prior_preservation:
            is_class = batch["is_class"].to(device=self.device)
            instance_mask = ~is_class
            class_mask = is_class
            instance_loss = per_example_loss[instance_mask].mean() if instance_mask.any() else per_example_loss.mean()
            class_loss = per_example_loss[class_mask].mean() if class_mask.any() else torch.zeros_like(instance_loss)
            return instance_loss + self.config.data.prior_loss_weight * class_loss
        return per_example_loss.mean()

    def backward(self, loss: Tensor) -> None:
        loss.backward()
        if self.stats_tracker is not None:
            self.stats_tracker.update_from_model(self.model)

    def collect_metrics(self, loss: Tensor, grad_norm: float | None = None) -> dict[str, float]:
        metrics = super().collect_metrics(loss, grad_norm=grad_norm)
        metrics.update(self._collect_group_learning_rates())
        metrics["dynamic_grad_threshold"] = self.dynamic_threshold.current_threshold()
        return metrics

    def step_optimizer(self) -> None:
        self._apply_group_learning_rates(step=self.global_step + 1)
        current = self.dynamic_threshold.update_from_layers(self.gaussian_layers)
        self.current_grad_threshold = max(self.config.densify.grad_threshold, current)
        if self.densify_scheduler is not None:
            self.densify_scheduler.config.grad_threshold = self.current_grad_threshold
        super().step_optimizer()

    def save_adapter(self, step: int) -> None:
        save_adapter_checkpoint(
            path=str(Path(self.config.runtime.output_dir) / f"adapter_step_{step}.pt"),
            model=self.model,
            target_modules=list(self.config.model.target_modules),
            config=GaussianAdapterConfig(**asdict(self.config.adapter)),
            step=step,
        )

    def _apply_group_learning_rates(self, *, step: int) -> None:
        self._set_group_learning_rate("gaussian_mu", self.config.training.resolve_lr_mu(step))
        self._set_group_learning_rate("gaussian_amp", self.config.training.resolve_lr_amp(step))
        self._set_group_learning_rate("gaussian_cov", self.config.training.resolve_lr_chol(step))

    def _set_group_learning_rate(self, group_name: str, lr: float) -> None:
        for group in self.optimizer.param_groups:
            if group.get("name") == group_name:
                group["lr"] = lr
                return

    def _collect_group_learning_rates(self) -> dict[str, float]:
        learning_rates: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name == "gaussian_amp":
                learning_rates["lr_amp"] = float(group["lr"])
            elif name == "gaussian_mu":
                learning_rates["lr_mu"] = float(group["lr"])
            elif name == "gaussian_cov":
                learning_rates["lr_chol"] = float(group["lr"])
            elif name == "gaussian_other":
                learning_rates["lr_other"] = float(group["lr"])
        return learning_rates


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def _describe_optimizer_parameter_coverage(
    gaussian_layers: list[nn.Module],
    optimizer: torch.optim.Optimizer,
) -> str:
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    trainable_weights = 0
    optimizer_has_trainable_weights = 0
    frozen_weights = 0
    optimizer_has_frozen_weights = 0
    adapter_params = 0
    optimizer_has_adapter_params = 0

    for layer in gaussian_layers:
        for name, param in layer.named_parameters(recurse=False):
            if name == "weight":
                if param.requires_grad:
                    trainable_weights += 1
                    optimizer_has_trainable_weights += int(id(param) in optimizer_param_ids)
                else:
                    frozen_weights += 1
                    optimizer_has_frozen_weights += int(id(param) in optimizer_param_ids)
            elif name in {"mu_raw", "chol_raw", "amp"}:
                adapter_params += 1
                optimizer_has_adapter_params += int(id(param) in optimizer_param_ids)

    return (
        "optimizer_param_check:"
        f" gaussian_layers={len(gaussian_layers)}"
        f" adapter_params_in_optimizer={optimizer_has_adapter_params}/{adapter_params}"
        f" trainable_weights_in_optimizer={optimizer_has_trainable_weights}/{trainable_weights}"
        f" frozen_weights_in_optimizer={optimizer_has_frozen_weights}/{frozen_weights}"
    )


def _resolve_mixed_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _maybe_autocast(device_type: str, dtype: torch.dtype):
    if device_type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if dtype not in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)
