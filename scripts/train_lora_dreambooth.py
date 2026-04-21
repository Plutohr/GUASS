from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_peft.data.dreambooth import DreamBoothExampleDataset, dreambooth_collate_fn
from gaussian_peft.utils.hf_loading import load_local_clip_tokenizer
from gaussian_peft.utils.logging import format_train_log
from gaussian_peft.utils.training_artifacts import TrainingArtifactWriter


@dataclass(slots=True)
class ModelConfig:
    model_root: str
    target_modules: list[str]


@dataclass(slots=True)
class DataConfig:
    instance_data_dir: str
    instance_prompt: str
    class_data_dir: str | None = None
    class_prompt: str | None = None
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    resolution: int = 512
    center_crop: bool = True
    num_workers: int = 8
    train_batch_size: int = 1
    tokenizer_max_length: int = 77


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.0
    init_std: float = 0.01
    train_bias: bool = False


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float | None = 1.0
    max_steps: int = 1000
    log_interval: int = 10
    save_interval: int = 100
    device: str = "cuda"
    seed: int = 42


@dataclass(slots=True)
class RuntimeConfig:
    output_dir: str = "outputs/dreambooth_sd_lora"
    mixed_precision: str = "fp16"
    pin_memory: bool = True
    drop_last: bool = True
    seed: int = 42
    save_adapter: bool = True
    save_full_model: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    lora: LoRAConfig
    training: TrainingConfig
    runtime: RuntimeConfig


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        init_std: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        weight = linear.weight.detach().clone()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, dtype=weight.dtype, device=weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, dtype=weight.dtype, device=weight.device)
        )
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: LoRAConfig) -> "LoRALinear":
        return cls(
            linear,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_std=config.init_std,
        )

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.lora_dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)
        return base + lora * self.scaling

    def adapter_state_dict(self) -> dict[str, Any]:
        return {
            "lora_A": self.lora_A.detach().cpu(),
            "lora_B": self.lora_B.detach().cpu(),
            "rank": self.rank,
            "alpha": self.alpha,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.bias is not None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable Diffusion DreamBooth with standard LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to LoRA DreamBooth YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_distributed()
    config = load_config(args.config)
    set_seed(config.runtime.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stable Diffusion DreamBooth training.")

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_sd_components_from_local(
        model_root=Path(config.model.model_root).expanduser().resolve(),
        dtype=resolve_mixed_precision(config.runtime.mixed_precision),
    )
    enable_gradient_checkpointing_if_available(unet)
    patched_unet, replaced = apply_lora(
        unet,
        target_modules=config.model.target_modules,
        config=config.lora,
    )
    if is_main_process():
        print(f"patched_modules={len(replaced)} targets={replaced}")

    train_dataset = build_dreambooth_dataset(
        tokenizer=tokenizer,
        instance_data_dir=config.data.instance_data_dir,
        instance_prompt=config.data.instance_prompt,
        class_data_dir=config.data.class_data_dir,
        class_prompt=config.data.class_prompt,
        with_prior_preservation=config.data.with_prior_preservation,
        size=config.data.resolution,
        center_crop=config.data.center_crop,
        max_length=config.data.tokenizer_max_length,
    )
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(train_dataset, shuffle=True)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.runtime.pin_memory,
        drop_last=config.runtime.drop_last,
        collate_fn=dreambooth_collate_fn,
    )

    device = torch.device(config.training.device)
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)

    vae.to(device)
    text_encoder.to(device)
    patched_unet.to(device)
    freeze_module(vae)
    freeze_module(text_encoder)

    if dist.is_available() and dist.is_initialized():
        patched_unet = DistributedDataParallel(
            patched_unet,
            device_ids=[device.index],
            output_device=device.index,
        )

    optimizer = torch.optim.AdamW(
        collect_lora_parameters(patched_unet),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.eps,
        weight_decay=config.training.weight_decay,
    )

    artifact_writer = TrainingArtifactWriter(
        config.runtime.output_dir,
        plot_every=config.training.log_interval,
    )

    for step, batch in enumerate(cycle(dataloader), start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_dreambooth_loss(
            model=patched_unet,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            batch=batch,
            device=device,
            model_dtype=resolve_mixed_precision(config.runtime.mixed_precision),
            with_prior_preservation=config.data.with_prior_preservation,
            prior_loss_weight=config.data.prior_loss_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach().item())}")
        loss.backward()
        grad_norm = clip_gradients(patched_unet, config.training.max_grad_norm)
        optimizer.step()

        metrics = {
            "loss": float(loss.detach().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "trainable_params": float(count_trainable_parameters(patched_unet)),
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
        }
        artifact_writer.log_step(step, metrics)

        if is_main_process() and (step % config.training.log_interval == 0 or step == config.training.max_steps):
            print(f"step={step} {format_train_log(metrics)}")
        if is_main_process() and (step % config.training.save_interval == 0 or step == config.training.max_steps):
            if config.runtime.save_adapter:
                save_lora_adapter(
                    path=Path(config.runtime.output_dir) / f"adapter_step_{step}.pt",
                    model=unwrap_model(patched_unet),
                    target_modules=config.model.target_modules,
                    config=config.lora,
                    step=step,
                )
            if config.runtime.save_full_model:
                torch.save(
                    {
                        "step": step,
                        "model": unwrap_model(patched_unet).state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    Path(config.runtime.output_dir) / f"full_step_{step}.pt",
                )

        if step >= config.training.max_steps:
            break

    if is_main_process():
        artifact_writer.finalize()


def load_config(path: str | os.PathLike[str]) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping")
    return ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoRAConfig(**raw.get("lora", {})),
        training=TrainingConfig(**raw.get("training", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )


def apply_lora(model: nn.Module, *, target_modules: list[str], config: LoRAConfig) -> tuple[nn.Module, list[str]]:
    replaced: list[str] = []
    normalized_targets = normalize_target_modules(target_modules)
    for module_name in collect_target_linear_names(model, normalized_targets):
        parent_module, child_name = resolve_parent_module(model, module_name)
        linear = getattr(parent_module, child_name)
        if not isinstance(linear, nn.Linear):
            continue
        setattr(parent_module, child_name, LoRALinear.from_linear(linear, config))
        replaced.append(module_name)
    freeze_non_lora_params(model, train_bias=config.train_bias)
    return model, replaced


def normalize_target_modules(target_modules: list[str] | None) -> list[str]:
    if not target_modules:
        return ["to_q", "to_k", "to_v", "to_out.0"]
    return ["to_out.0" if name == "to_out" else name for name in target_modules]


def collect_target_linear_names(model: nn.Module, target_modules: list[str]) -> list[str]:
    matched: list[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(module_name.endswith(target) for target in target_modules):
            matched.append(module_name)
    return matched


def resolve_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    path = module_name.split(".")
    parent = model
    for part in path[:-1]:
        parent = getattr(parent, part)
    return parent, path[-1]


def freeze_non_lora_params(model: nn.Module, *, train_bias: bool) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = False
        if train_bias and name.endswith(".bias"):
            param.requires_grad = True
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = train_bias


def collect_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable LoRA parameters found after patching")
    return params


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def save_lora_adapter(
    *,
    path: Path,
    model: nn.Module,
    target_modules: list[str],
    config: LoRAConfig,
    step: int,
) -> None:
    adapters: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapters[module_name] = module.adapter_state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "lora-linear-adapter",
            "step": step,
            "metadata": {
                "target_modules": list(target_modules),
                "lora_config": {
                    "rank": config.rank,
                    "alpha": config.alpha,
                    "dropout": config.dropout,
                    "init_std": config.init_std,
                    "train_bias": config.train_bias,
                },
            },
            "adapters": adapters,
        },
        path,
    )


def compute_dreambooth_loss(
    *,
    model: nn.Module,
    vae: nn.Module,
    text_encoder: nn.Module,
    noise_scheduler: Any,
    batch: dict[str, Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
    with_prior_preservation: bool,
    prior_loss_weight: float,
) -> Tensor:
    pixel_values = batch["pixel_values"].to(device=device, dtype=vae.dtype, non_blocking=True)
    input_ids = batch["input_ids"].to(device=device, non_blocking=True)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device, non_blocking=True)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        encoder_hidden_states = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    with maybe_autocast(device.type, model_dtype):
        model_pred = model(
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

    if with_prior_preservation:
        is_class = batch["is_class"].to(device=device)
        instance_mask = ~is_class
        class_mask = is_class
        instance_loss = per_example_loss[instance_mask].mean() if instance_mask.any() else per_example_loss.mean()
        class_loss = per_example_loss[class_mask].mean() if class_mask.any() else torch.zeros_like(instance_loss)
        return instance_loss + prior_loss_weight * class_loss
    return per_example_loss.mean()


def clip_gradients(model: nn.Module, max_grad_norm: float | None) -> float | None:
    if max_grad_norm is None:
        return None
    params = [param for param in model.parameters() if param.requires_grad and param.grad is not None]
    if not params:
        return None
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
    return float(grad_norm.detach().float().item())


def build_dreambooth_dataset(
    *,
    tokenizer: Any,
    instance_data_dir: str,
    instance_prompt: str,
    class_data_dir: str | None,
    class_prompt: str | None,
    with_prior_preservation: bool,
    size: int,
    center_crop: bool,
    max_length: int,
):
    instance_dataset = DreamBoothExampleDataset(
        data_dir=instance_data_dir,
        tokenizer=tokenizer,
        prompt=instance_prompt,
        size=size,
        center_crop=center_crop,
        max_length=max_length,
        is_class=False,
    )
    if not with_prior_preservation:
        return instance_dataset
    class_dataset = DreamBoothExampleDataset(
        data_dir=class_data_dir,
        tokenizer=tokenizer,
        prompt=class_prompt,
        size=size,
        center_crop=center_crop,
        max_length=max_length,
        is_class=True,
    )
    return ConcatDataset([instance_dataset, class_dataset])


def enable_gradient_checkpointing_if_available(unet: Any) -> None:
    for method_name in ("enable_gradient_checkpointing", "gradient_checkpointing_enable"):
        method = getattr(unet, method_name, None)
        if callable(method):
            method()
            return


def load_sd_components_from_local(
    model_root: Path,
    dtype: torch.dtype,
) -> tuple[Any, nn.Module, nn.Module, nn.Module, Any]:
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel
    except ImportError as exc:
        raise RuntimeError("diffusers / transformers are required in the HPC environment.") from exc

    tokenizer = load_local_clip_tokenizer(model_root)
    text_encoder = CLIPTextModel.from_pretrained(
        str(model_root / "text_encoder"),
        local_files_only=True,
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        str(model_root / "vae"),
        local_files_only=True,
        torch_dtype=dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(model_root / "unet"),
        local_files_only=True,
        torch_dtype=dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        str(model_root / "scheduler"),
        local_files_only=True,
    )
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def resolve_mixed_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def maybe_autocast(device_type: str, dtype: torch.dtype):
    if device_type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if dtype not in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_distributed() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


if __name__ == "__main__":
    main()
