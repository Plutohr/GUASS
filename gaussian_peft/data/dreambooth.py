from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(slots=True)
class _DreamBoothSample:
    image_path: Path
    prompt: str


class DreamBoothExampleDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        *,
        data_dir: str | Path,
        tokenizer: Any,
        prompt: str | None,
        size: int,
        center_crop: bool,
        max_length: int,
        is_class: bool,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"DreamBooth data directory not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"DreamBooth data path is not a directory: {self.data_dir}")

        self.tokenizer = tokenizer
        self.prompt = prompt.strip() if isinstance(prompt, str) else None
        self.max_length = int(max_length)
        self.is_class = bool(is_class)
        self.samples = _load_samples(self.data_dir, self.prompt)
        self.image_transform = _build_image_transform(size=size, center_crop=center_crop)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.samples[index]
        pixel_values = _load_pixel_values(sample.image_path, self.image_transform)
        tokenized = self.tokenizer(
            sample.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        batch = {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0),
            "is_class": torch.tensor(self.is_class, dtype=torch.bool),
        }
        attention_mask = getattr(tokenized, "attention_mask", None)
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask.squeeze(0)
        return batch


def dreambooth_collate_fn(examples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")

    batch = {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]).contiguous(),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "is_class": torch.stack([example["is_class"] for example in examples]),
    }
    if all("attention_mask" in example for example in examples):
        batch["attention_mask"] = torch.stack([example["attention_mask"] for example in examples])
    return batch


def _load_samples(data_dir: Path, prompt_override: str | None) -> list[_DreamBoothSample]:
    metadata_path = _find_metadata_path(data_dir)
    if metadata_path is not None:
        samples = _load_samples_from_metadata(metadata_path, data_dir, prompt_override)
        if samples:
            return samples

    image_paths = sorted(path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No image files found under {data_dir}")
    return [
        _DreamBoothSample(
            image_path=image_path,
            prompt=_resolve_prompt(prompt_override, image_path, None),
        )
        for image_path in image_paths
    ]


def _find_metadata_path(data_dir: Path) -> Path | None:
    candidates = [
        data_dir / "metadata.jsonl",
        data_dir.parent / "metadata.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_samples_from_metadata(
    metadata_path: Path,
    data_dir: Path,
    prompt_override: str | None,
) -> list[_DreamBoothSample]:
    samples: list[_DreamBoothSample] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_value = row.get("image_path")
            if not isinstance(image_value, str) or not image_value.strip():
                continue
            image_path = _resolve_metadata_image_path(image_value, data_dir, metadata_path)
            if image_path is None:
                continue
            prompt_from_metadata = row.get("prompt")
            prompt = _resolve_prompt(
                prompt_override,
                image_path,
                prompt_from_metadata if isinstance(prompt_from_metadata, str) else None,
            )
            samples.append(_DreamBoothSample(image_path=image_path, prompt=prompt))
    return samples


def _resolve_metadata_image_path(image_value: str, data_dir: Path, metadata_path: Path) -> Path | None:
    raw_path = Path(image_value)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                (Path.cwd() / raw_path).resolve(),
                (metadata_path.parent / raw_path).resolve(),
                (data_dir / raw_path).resolve(),
                (data_dir / raw_path.name).resolve(),
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_prompt(prompt_override: str | None, image_path: Path, metadata_prompt: str | None) -> str:
    if prompt_override:
        return prompt_override
    sidecar_prompt = _read_sidecar_prompt(image_path)
    if sidecar_prompt is not None:
        return sidecar_prompt
    if metadata_prompt is not None and metadata_prompt.strip():
        return metadata_prompt.strip()
    raise ValueError(f"No prompt found for image: {image_path}")


def _read_sidecar_prompt(image_path: Path) -> str | None:
    sidecar_path = image_path.with_suffix(".txt")
    if not sidecar_path.exists():
        return None
    text = sidecar_path.read_text(encoding="utf-8").strip()
    return text or None


def _build_image_transform(*, size: int, center_crop: bool):
    crop_transform = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            crop_transform,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _load_pixel_values(image_path: Path, image_transform) -> Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return image_transform(image)
