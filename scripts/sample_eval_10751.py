from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
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
import yaml
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from gaussian_peft.checkpoints.state_dict import load_gaussian_adapter_state_dict
from gaussian_peft.config.adapter import GaussianAdapterConfig, normalize_execution_mode
from gaussian_peft.config.loader import load_diffusion_config
from gaussian_peft.patchers.replace_linear import apply_gaussian_peft
from gaussian_peft.utils.precision import get_compute_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent sampling evaluation for experiment 10751")
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Training YAML config. Used only for model/data/adapter defaults; not modified.",
    )
    parser.add_argument(
        "--sampling-config",
        type=str,
        required=True,
        help="Independent sampling config YAML/JSON.",
    )
    parser.add_argument(
        "--allow-metadata-rebuild",
        action="store_true",
        help="Rebuild adapter config from checkpoint metadata after normalization when mismatch is found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = load_diffusion_config(args.train_config)
    sampling_config = load_sampling_config(args.sampling_config)
    validate_sampling_config(sampling_config)

    device = torch.device(str(sampling_config["runtime"].get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sampling evaluation.")

    pipeline_dtype = resolve_dtype(str(sampling_config["runtime"].get("pipeline_dtype", "fp16")))
    output_root = Path(str(sampling_config["output_dir"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    log_path = output_root / "run_log.json"

    prompts: list[str] = list(sampling_config["prompts"])
    seeds: list[int] = [int(seed) for seed in sampling_config["seeds"]]
    models: list[dict[str, Any]] = list(sampling_config["models"])
    scheduler_name = str(sampling_config["runtime"]["scheduler"])
    num_inference_steps = int(sampling_config["runtime"]["num_inference_steps"])
    guidance_scale = float(sampling_config["runtime"]["guidance_scale"])
    width = int(sampling_config["runtime"]["width"])
    height = int(sampling_config["runtime"]["height"])
    negative_prompt = sampling_config["runtime"].get("negative_prompt")

    manifest_rows: list[dict[str, str]] = []
    run_log: dict[str, Any] = {
        "train_config": str(Path(args.train_config).expanduser().resolve()),
        "sampling_config": str(Path(args.sampling_config).expanduser().resolve()),
        "output_dir": str(output_root),
        "scheduler": scheduler_name,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "device": str(device),
        "pipeline_dtype": str(pipeline_dtype),
        "models": [],
    }

    for model_spec in models:
        model_label = str(model_spec["label"])
        model_output_dir = output_root / model_label
        model_output_dir.mkdir(parents=True, exist_ok=True)

        pipe, resolved_kind, resolved_checkpoint = build_pipeline(
            train_config=train_config,
            model_spec=model_spec,
            scheduler_name=scheduler_name,
            pipeline_dtype=pipeline_dtype,
            device=device,
            allow_metadata_rebuild=args.allow_metadata_rebuild,
        )
        pipe.set_progress_bar_config(disable=True)

        run_log["models"].append(
            {
                "label": model_label,
                "kind": resolved_kind,
                "checkpoint_path": resolved_checkpoint,
            }
        )

        with torch.inference_mode():
            for prompt_index, prompt in enumerate(prompts):
                prompt_dir = model_output_dir / f"prompt_{prompt_index:02d}"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                for seed in seeds:
                    image_path = prompt_dir / f"seed_{seed:04d}.png"
                    image = generate_image(
                        pipe=pipe,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        width=width,
                        height=height,
                        device=device,
                    )
                    image.save(image_path)
                    manifest_rows.append(
                        {
                            "model_kind": resolved_kind,
                            "checkpoint_name": checkpoint_name_for_manifest(model_label, resolved_checkpoint),
                            "checkpoint_path": resolved_checkpoint or "",
                            "prompt_index": str(prompt_index),
                            "prompt": prompt,
                            "seed": str(seed),
                            "scheduler": scheduler_name,
                            "num_inference_steps": str(num_inference_steps),
                            "guidance_scale": str(guidance_scale),
                            "resolution": f"{width}x{height}",
                            "config_source": str(Path(args.sampling_config).expanduser().resolve()),
                            "image_path": str(image_path),
                        }
                    )

        del pipe
        torch.cuda.empty_cache()

    write_manifest(manifest_path, manifest_rows)
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"wrote_manifest={manifest_path}")
    print(f"wrote_log={log_path}")
    print(f"num_images={len(manifest_rows)}")


def load_sampling_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(payload)
    else:
        loaded = yaml.safe_load(payload)
    if not isinstance(loaded, dict):
        raise ValueError("Sampling config must be a mapping.")
    return loaded


def validate_sampling_config(config: dict[str, Any]) -> None:
    required_top_level = {"output_dir", "prompts", "seeds", "models", "runtime"}
    missing = sorted(required_top_level - set(config))
    if missing:
        raise ValueError(f"Sampling config missing required keys: {missing}")
    if not config["prompts"]:
        raise ValueError("Sampling config prompts must not be empty.")
    if not config["seeds"]:
        raise ValueError("Sampling config seeds must not be empty.")
    if not config["models"]:
        raise ValueError("Sampling config models must not be empty.")
    runtime = config["runtime"]
    required_runtime = {"scheduler", "num_inference_steps", "guidance_scale", "width", "height"}
    runtime_missing = sorted(required_runtime - set(runtime))
    if runtime_missing:
        raise ValueError(f"Sampling config runtime missing required keys: {runtime_missing}")


def build_pipeline(
    *,
    train_config: Any,
    model_spec: dict[str, Any],
    scheduler_name: str,
    pipeline_dtype: torch.dtype,
    device: torch.device,
    allow_metadata_rebuild: bool,
) -> tuple[StableDiffusionPipeline, str, str | None]:
    model_root = Path(str(model_spec.get("model_root", train_config.model.model_root))).expanduser().resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")
    tokenizer, text_encoder, vae, unet, scheduler = load_sd_components_from_local(
        model_root=model_root,
        dtype=pipeline_dtype,
    )

    kind = str(model_spec["kind"])
    checkpoint_path: str | None = None
    if kind == "base":
        pass
    elif kind == "adapter":
        checkpoint_path = str(Path(str(model_spec["checkpoint_path"])).expanduser().resolve())
        adapter_config, target_modules, payload = resolve_adapter_load_spec(
            checkpoint_path=checkpoint_path,
            train_config=train_config,
            model_spec=model_spec,
            allow_metadata_rebuild=allow_metadata_rebuild,
        )
        _patched_model, replaced = apply_gaussian_peft(
            unet,
            target_modules,
            adapter_config,
            freeze_base=True,
        )
        if not replaced:
            raise RuntimeError("No target linear modules were patched for adapter sampling.")
        load_gaussian_adapter_state_dict(unet, payload["adapters"])
    else:
        raise NotImplementedError(
            "First version only supports 'base' and 'adapter' model kinds. "
            "Full checkpoints are intentionally not enabled by default."
        )

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = build_scheduler(scheduler_name, pipe.scheduler.config)
    pipe = pipe.to(device=device, torch_dtype=pipeline_dtype)
    return pipe, kind, checkpoint_path


def resolve_adapter_load_spec(
    *,
    checkpoint_path: str,
    train_config: Any,
    model_spec: dict[str, Any],
    allow_metadata_rebuild: bool,
) -> tuple[GaussianAdapterConfig, list[str], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Adapter checkpoint missing metadata: {checkpoint_path}")

    expected_target_modules = list(model_spec.get("target_modules", train_config.model.target_modules))
    expected_adapter_config = GaussianAdapterConfig(**asdict(train_config.adapter))
    metadata_target_modules = list(metadata.get("target_modules", []))
    metadata_adapter_raw = metadata.get("adapter_config")
    if not isinstance(metadata_adapter_raw, dict):
        raise ValueError(f"Adapter checkpoint missing adapter_config metadata: {checkpoint_path}")

    normalized_metadata_adapter = normalize_adapter_metadata(metadata_adapter_raw)
    metadata_adapter_config = GaussianAdapterConfig(**normalized_metadata_adapter)
    metadata_adapter_config.validate()

    target_modules_match = metadata_target_modules == expected_target_modules
    adapter_match = adapter_configs_match(expected_adapter_config, metadata_adapter_config)
    if target_modules_match and adapter_match:
        return expected_adapter_config, expected_target_modules, payload

    if not allow_metadata_rebuild:
        raise ValueError(
            "Adapter checkpoint metadata mismatch.\n"
            f"expected_target_modules={expected_target_modules}\n"
            f"metadata_target_modules={metadata_target_modules}\n"
            f"expected_adapter_config={adapter_config_to_comparable_dict(expected_adapter_config)}\n"
            f"metadata_adapter_config={adapter_config_to_comparable_dict(metadata_adapter_config)}\n"
            "Re-run with --allow-metadata-rebuild to patch using normalized metadata values."
        )

    return metadata_adapter_config, metadata_target_modules, payload


def normalize_adapter_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    if "compute_dtype" in normalized:
        normalized["compute_dtype"] = get_compute_dtype(str(normalized["compute_dtype"]).replace("torch.", ""))
    if "execution_mode" in normalized:
        normalized["execution_mode"] = normalize_execution_mode(str(normalized["execution_mode"]))
    return normalized


def adapter_configs_match(left: GaussianAdapterConfig, right: GaussianAdapterConfig) -> bool:
    return adapter_config_to_comparable_dict(left) == adapter_config_to_comparable_dict(right)


def adapter_config_to_comparable_dict(config: GaussianAdapterConfig) -> dict[str, Any]:
    data = asdict(config)
    data["compute_dtype"] = str(config.compute_dtype)
    return data


def load_sd_components_from_local(
    model_root: Path,
    dtype: torch.dtype,
) -> tuple[Any, torch.nn.Module, torch.nn.Module, torch.nn.Module, Any]:
    tokenizer = CLIPTokenizer.from_pretrained(str(model_root / "tokenizer"), local_files_only=True)
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
    scheduler = DDPMScheduler.from_pretrained(
        str(model_root / "scheduler"),
        local_files_only=True,
    )
    return tokenizer, text_encoder, vae, unet, scheduler


def build_scheduler(name: str, config: Any):
    normalized = name.strip().lower()
    if normalized not in {"dpmsolvermultistep", "dpm_solver_multistep", "dpmsolver"}:
        raise ValueError(f"Unsupported scheduler for evaluation: {name!r}")
    return DPMSolverMultistepScheduler.from_config(config)


def generate_image(
    *,
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str | None,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    width: int,
    height: int,
    device: torch.device,
) -> Image.Image:
    generator = torch.Generator(device=device.type).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    return result.images[0]


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported pipeline dtype: {name!r}")


def checkpoint_name_for_manifest(model_label: str, checkpoint_path: str | None) -> str:
    if checkpoint_path is None:
        return model_label
    return Path(checkpoint_path).name


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "model_kind",
        "checkpoint_name",
        "checkpoint_path",
        "prompt_index",
        "prompt",
        "seed",
        "scheduler",
        "num_inference_steps",
        "guidance_scale",
        "resolution",
        "config_source",
        "image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
