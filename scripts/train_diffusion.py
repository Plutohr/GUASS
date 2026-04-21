from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from gaussian_peft.config.loader import load_diffusion_config
from gaussian_peft.data.dreambooth import DreamBoothExampleDataset, dreambooth_collate_fn
from gaussian_peft.patchers.replace_linear import apply_gaussian_peft
from gaussian_peft.trainers.diffusion_trainer import DiffusionTrainer
from gaussian_peft.utils.hf_loading import load_local_clip_tokenizer
from gaussian_peft.utils.logging import format_train_log
from gaussian_peft.utils.training_artifacts import TrainingArtifactWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable Diffusion DreamBooth with Gaussian-PEFT")
    parser.add_argument("--config", type=str, required=True, help="Path to DreamBooth YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_distributed()
    config = load_diffusion_config(args.config)
    set_seed(config.runtime.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stable Diffusion DreamBooth training.")

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_sd_components_from_local(
        model_root=Path(config.model.model_root).expanduser().resolve(),
        dtype=_resolve_mixed_precision(config.runtime.mixed_precision),
    )
    _enable_gradient_checkpointing_if_available(unet)
    _model, replaced = apply_gaussian_peft(
        unet,
        config.model.target_modules,
        config.adapter,
        freeze_base=True,
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

    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        torch.cuda.set_device(local_rank)
        unet.to(local_rank)
        unet = DistributedDataParallel(unet, device_ids=[local_rank], output_device=local_rank)

    trainer = DiffusionTrainer(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        config=config,
    )
    artifact_writer = TrainingArtifactWriter(
        config.runtime.output_dir,
        plot_every=config.training.log_interval,
    )

    def handle_step(step: int, metrics: dict[str, float]) -> None:
        artifact_writer.log_step(step, metrics)
        if is_main_process() and (step % config.training.log_interval == 0 or step == config.training.max_steps):
            print(f"step={step} {format_train_log(metrics)}")
        if is_main_process() and (step % config.training.save_interval == 0 or step == config.training.max_steps):
            if config.runtime.save_adapter:
                trainer.save_adapter(step)
            if config.runtime.save_full_model:
                trainer.save_checkpoint(f"full_step_{step}.pt")

    trainer.fit(
        _cycle(dataloader),
        max_steps=config.training.max_steps,
        on_step=handle_step if is_main_process() else None,
    )
    if is_main_process():
        artifact_writer.finalize()


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


def _enable_gradient_checkpointing_if_available(unet: Any) -> None:
    for method_name in ("enable_gradient_checkpointing", "gradient_checkpointing_enable"):
        method = getattr(unet, method_name, None)
        if callable(method):
            method()
            return


def load_sd_components_from_local(
    model_root: Path,
    dtype: torch.dtype,
) -> tuple[Any, torch.nn.Module, torch.nn.Module, torch.nn.Module, Any]:
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


def _cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def _resolve_mixed_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


if __name__ == "__main__":
    main()
