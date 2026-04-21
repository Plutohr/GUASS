from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingConfig:
    lr_mu: float = 1e-3
    lr_chol: float = 5e-4
    lr_amp: float = 1e-3
    lr_amp_warmup_steps: int = 0
    lr_amp_warmup_value: float = 0.0
    lr_mu_schedule_steps: list[int] = field(default_factory=list)
    lr_mu_schedule_values: list[float] = field(default_factory=list)
    lr_chol_schedule_steps: list[int] = field(default_factory=list)
    lr_chol_schedule_values: list[float] = field(default_factory=list)
    lr_amp_schedule_steps: list[int] = field(default_factory=list)
    lr_amp_schedule_values: list[float] = field(default_factory=list)
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    max_grad_norm_amp: float | None = None
    max_grad_norm_mu: float | None = None
    max_grad_norm_chol: float | None = None
    max_grad_norm_other: float | None = None
    max_steps: int = 1000
    log_interval: int = 10
    save_interval: int = 200
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 42

    def validate(self) -> None:
        if self.lr_mu <= 0 or self.lr_chol <= 0 or self.lr_amp <= 0:
            raise ValueError("All learning rates must be positive")
        if self.lr_amp_warmup_steps < 0:
            raise ValueError("lr_amp_warmup_steps must be non-negative")
        if self.lr_amp_warmup_value < 0:
            raise ValueError("lr_amp_warmup_value must be non-negative")
        _validate_lr_schedule(
            "lr_mu",
            self.lr_mu_schedule_steps,
            self.lr_mu_schedule_values,
        )
        _validate_lr_schedule(
            "lr_chol",
            self.lr_chol_schedule_steps,
            self.lr_chol_schedule_values,
        )
        _validate_lr_schedule(
            "lr_amp",
            self.lr_amp_schedule_steps,
            self.lr_amp_schedule_values,
        )
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided")
        for name, value in (
            ("max_grad_norm_amp", self.max_grad_norm_amp),
            ("max_grad_norm_mu", self.max_grad_norm_mu),
            ("max_grad_norm_chol", self.max_grad_norm_chol),
            ("max_grad_norm_other", self.max_grad_norm_other),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")

    def resolve_lr_mu(self, step: int) -> float:
        if step <= 0:
            raise ValueError("step must be positive")
        return _resolve_piecewise_lr(
            step=step,
            base_lr=self.lr_mu,
            schedule_steps=self.lr_mu_schedule_steps,
            schedule_values=self.lr_mu_schedule_values,
        )

    def resolve_lr_chol(self, step: int) -> float:
        if step <= 0:
            raise ValueError("step must be positive")
        return _resolve_piecewise_lr(
            step=step,
            base_lr=self.lr_chol,
            schedule_steps=self.lr_chol_schedule_steps,
            schedule_values=self.lr_chol_schedule_values,
        )

    def resolve_lr_amp(self, step: int) -> float:
        if step <= 0:
            raise ValueError("step must be positive")
        if self.lr_amp_warmup_steps > 0 and step <= self.lr_amp_warmup_steps:
            return self.lr_amp_warmup_value
        return _resolve_piecewise_lr(
            step=step,
            base_lr=self.lr_amp,
            schedule_steps=self.lr_amp_schedule_steps,
            schedule_values=self.lr_amp_schedule_values,
        )

    def uses_per_group_grad_clipping(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_grad_norm_amp,
                self.max_grad_norm_mu,
                self.max_grad_norm_chol,
                self.max_grad_norm_other,
            )
        )

    def resolve_max_grad_norm(self, group_name: str | None) -> float | None:
        per_group = {
            "gaussian_amp": self.max_grad_norm_amp,
            "gaussian_mu": self.max_grad_norm_mu,
            "gaussian_cov": self.max_grad_norm_chol,
            "gaussian_other": self.max_grad_norm_other,
        }
        if group_name in per_group and per_group[group_name] is not None:
            return per_group[group_name]
        return self.max_grad_norm


def _validate_lr_schedule(name: str, schedule_steps: list[int], schedule_values: list[float]) -> None:
    if len(schedule_steps) != len(schedule_values):
        raise ValueError(f"{name}_schedule_steps and {name}_schedule_values must have the same length")
    previous_step = 0
    for step in schedule_steps:
        if step <= 0:
            raise ValueError(f"{name}_schedule_steps must be positive")
        if step <= previous_step:
            raise ValueError(f"{name}_schedule_steps must be strictly increasing")
        previous_step = step
    for value in schedule_values:
        if value <= 0:
            raise ValueError(f"{name}_schedule_values must be positive")


def _resolve_piecewise_lr(
    *,
    step: int,
    base_lr: float,
    schedule_steps: list[int],
    schedule_values: list[float],
) -> float:
    if not schedule_steps:
        return base_lr
    for boundary, value in zip(schedule_steps, schedule_values):
        if step <= boundary:
            return value
    return schedule_values[-1]
