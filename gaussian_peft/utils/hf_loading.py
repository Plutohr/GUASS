from __future__ import annotations

import json
import os
from hashlib import sha1
from pathlib import Path
from typing import Any


def load_local_clip_tokenizer(model_root: Path) -> Any:
    try:
        from transformers import AutoTokenizer, CLIPTokenizer, CLIPTokenizerFast
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the local tokenizer.") from exc

    tokenizer_dir = str(model_root / "tokenizer")
    last_error: Exception | None = None
    for loader in (
        lambda: AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True),
        lambda: CLIPTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True),
        lambda: CLIPTokenizer.from_pretrained(tokenizer_dir, local_files_only=True),
    ):
        try:
            return loader()
        except Exception as exc:  # pragma: no cover - exercised against local model layouts
            last_error = exc

    try:
        return _load_clip_tokenizer_from_bpe_json(model_root)
    except Exception as exc:  # pragma: no cover - exercised against local model layouts
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_dir}.") from exc


def _load_clip_tokenizer_from_bpe_json(model_root: Path) -> Any:
    from transformers import CLIPTokenizer

    tokenizer_dir = model_root / "tokenizer"
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json under {tokenizer_dir}.")

    with tokenizer_json_path.open(encoding="utf-8") as handle:
        tokenizer_json = json.load(handle)

    model = tokenizer_json.get("model", {})
    vocab = model.get("vocab")
    merges = model.get("merges")
    if model.get("type") != "BPE" or not isinstance(vocab, dict) or not isinstance(merges, list):
        raise ValueError("tokenizer.json does not contain a compatible BPE model.")

    config = _load_tokenizer_config(tokenizer_dir / "tokenizer_config.json")
    compat_dir = _get_compat_tokenizer_dir(tokenizer_json_path)
    vocab_path = compat_dir / "vocab.json"
    merges_path = compat_dir / "merges.txt"

    if not vocab_path.exists():
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    if not merges_path.exists():
        merge_lines = ["#version: 0.2", *(_format_merge_entry(entry) for entry in merges)]
        merges_path.write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

    return CLIPTokenizer(
        vocab_file=str(vocab_path),
        merges_file=str(merges_path),
        unk_token=config.get("unk_token", "<|endoftext|>"),
        bos_token=config.get("bos_token", "<|startoftext|>"),
        eos_token=config.get("eos_token", "<|endoftext|>"),
        pad_token=config.get("pad_token", config.get("eos_token", "<|endoftext|>")),
        errors=config.get("errors", "replace"),
        do_lower_case=bool(config.get("do_lower_case", True)),
        model_max_length=int(config.get("model_max_length", 77)),
    )


def _load_tokenizer_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _get_compat_tokenizer_dir(tokenizer_json_path: Path) -> Path:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    digest_input = f"{tokenizer_json_path.resolve()}::{tokenizer_json_path.stat().st_mtime_ns}"
    compat_dir = cache_root / "gaussian_peft_tokenizers" / sha1(digest_input.encode("utf-8")).hexdigest()
    compat_dir.mkdir(parents=True, exist_ok=True)
    return compat_dir


def _format_merge_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list) and len(entry) == 2 and all(isinstance(part, str) for part in entry):
        return f"{entry[0]} {entry[1]}"
    raise ValueError(f"Unsupported BPE merge entry: {entry!r}")
