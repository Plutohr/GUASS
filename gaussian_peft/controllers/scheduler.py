from __future__ import annotations

from gaussian_peft.config.densify import DensifyConfig


class DensifyScheduler:
    def __init__(self, config: DensifyConfig) -> None:
        config.validate()
        self.config = config

    def in_active_window(self, step: int) -> bool:
        if not self.config.enabled:
            return False
        return self.config.densify_from_step <= step <= self.config.densify_until_step

    def should_clone(self, step: int) -> bool:
        if not self.in_active_window(step):
            return False
        return self._is_interval_hit(step, self.config.densification_interval)

    def should_prune(self, step: int) -> bool:
        if not self.in_active_window(step):
            return False
        if step < self.config.prune_warmup_steps:
            return False
        return self._is_interval_hit(step, self.config.prune_interval)

    def should_reset_stats(self, step: int) -> bool:
        if not self.config.enabled:
            return False
        return self._is_interval_hit(step, self.config.stats_reset_interval)

    def _is_interval_hit(self, step: int, interval: int) -> bool:
        return step > 0 and step % interval == 0
