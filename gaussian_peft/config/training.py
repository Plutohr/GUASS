from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    lr_mu: float = 1e-3
    lr_chol: float = 5e-4
    lr_amp: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    max_steps: int = 1000
    log_interval: int = 10
    save_interval: int = 200
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 42

    def validate(self) -> None:
        if self.lr_mu <= 0 or self.lr_chol <= 0 or self.lr_amp <= 0:
            raise ValueError("All learning rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
