from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.config.densify import DensifyConfig
from gaussian_peft.config.training import TrainingConfig


@dataclass(slots=True)
class DiffusionModelConfig:
    model_root: str = "models/stable-diffusion-v1-5"
    target_modules: list[str] = field(
        default_factory=lambda: ["to_q", "to_v"]
    )

    def validate(self) -> None:
        if not self.target_modules:
            raise ValueError("model.target_modules must not be empty")


@dataclass(slots=True)
class DreamBoothDataConfig:
    instance_data_dir: str = "data/instance"
    instance_prompt: str = "a photo of sks subject"
    class_data_dir: str | None = None
    class_prompt: str | None = None
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    resolution: int = 512
    center_crop: bool = True
    num_workers: int = 4
    train_batch_size: int = 2
    tokenizer_max_length: int = 77

    def validate(self) -> None:
        if self.train_batch_size <= 0:
            raise ValueError("data.train_batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
        if self.resolution <= 0:
            raise ValueError("data.resolution must be positive")
        if self.tokenizer_max_length <= 0:
            raise ValueError("data.tokenizer_max_length must be positive")
        if self.prior_loss_weight < 0:
            raise ValueError("data.prior_loss_weight must be non-negative")
        if self.with_prior_preservation:
            if not self.class_data_dir:
                raise ValueError("data.class_data_dir is required when prior preservation is enabled")
            if not self.class_prompt:
                raise ValueError("data.class_prompt is required when prior preservation is enabled")


@dataclass(slots=True)
class DiffusionRuntimeConfig:
    output_dir: str = "outputs/dreambooth_sd"
    mixed_precision: str = "fp16"
    pin_memory: bool = True
    drop_last: bool = True
    seed: int = 42
    resume_from_checkpoint: str | None = None
    save_adapter: bool = True
    save_full_model: bool = True

    def validate(self) -> None:
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("runtime.mixed_precision must be one of {'no', 'fp16', 'bf16'}")


@dataclass(slots=True)
class DiffusionExperimentConfig:
    model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    data: DreamBoothDataConfig = field(default_factory=DreamBoothDataConfig)
    adapter: GaussianAdapterConfig = field(default_factory=GaussianAdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    densify: DensifyConfig = field(default_factory=DensifyConfig)
    runtime: DiffusionRuntimeConfig = field(default_factory=DiffusionRuntimeConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.adapter.validate()
        self.training.validate()
        self.densify.validate()
        self.runtime.validate()

    @property
    def output_dir(self) -> Path:
        return Path(self.runtime.output_dir).expanduser().resolve()
