from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.config.densify import DensifyConfig
from gaussian_peft.config.diffusion import (
    DiffusionExperimentConfig,
    DiffusionModelConfig,
    DiffusionRuntimeConfig,
    DreamBoothDataConfig,
)
from gaussian_peft.config.training import TrainingConfig
from gaussian_peft.utils.precision import get_compute_dtype


def load_diffusion_config(path: str | Path) -> DiffusionExperimentConfig:
    raw = load_raw_config(path)
    adapter = dict(raw.get("adapter", {}))
    if "compute_dtype" in adapter:
        adapter["compute_dtype"] = get_compute_dtype(adapter["compute_dtype"])
    config = DiffusionExperimentConfig(
        model=DiffusionModelConfig(**raw.get("model", {})),
        data=DreamBoothDataConfig(**raw.get("data", {})),
        adapter=GaussianAdapterConfig(**adapter),
        training=TrainingConfig(**raw.get("training", {})),
        densify=DensifyConfig(**raw.get("densify", {})),
        runtime=DiffusionRuntimeConfig(**raw.get("runtime", {})),
    )
    config.validate()
    return config


def load_raw_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Top-level config must be a mapping")
    return payload
