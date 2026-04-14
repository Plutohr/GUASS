from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DensifyConfig:
    enabled: bool = False
    densify_from_step: int = 500
    densify_until_step: int = 5000
    densification_interval: int = 200
    prune_interval: int = 400
    stats_reset_interval: int = 1000
    grad_threshold: float = 0.0
    amp_threshold: float = 1e-6
    max_gaussians_per_layer: int = 128
    min_gaussians_per_layer: int = 4
    max_new_gaussians_per_step: int = 1
    clone_noise_scale: float = 1e-2
    prune_warmup_steps: int = 500

    def validate(self) -> None:
        if self.densify_from_step < 0:
            raise ValueError("densify_from_step must be non-negative")
        if self.densify_until_step < self.densify_from_step:
            raise ValueError("densify_until_step must be >= densify_from_step")
        if self.densification_interval <= 0:
            raise ValueError("densification_interval must be positive")
        if self.prune_interval <= 0:
            raise ValueError("prune_interval must be positive")
        if self.stats_reset_interval <= 0:
            raise ValueError("stats_reset_interval must be positive")
        if self.min_gaussians_per_layer <= 0:
            raise ValueError("min_gaussians_per_layer must be positive")
        if self.max_gaussians_per_layer < self.min_gaussians_per_layer:
            raise ValueError(
                "max_gaussians_per_layer must be >= min_gaussians_per_layer"
            )
        if self.max_new_gaussians_per_step <= 0:
            raise ValueError("max_new_gaussians_per_step must be positive")
