from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


HF_API_ROOT = "https://huggingface.co/api/datasets/google/dreambooth/tree/main/dataset"
HF_RESOLVE_ROOT = "https://huggingface.co/datasets/google/dreambooth/resolve/main/dataset"
DEFAULT_USER_AGENT = "gauss-dreambooth-prep/1.0"
DEFAULT_TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1024 * 1024


@dataclass(slots=True)
class SubjectInfo:
    subject_name: str
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare a Google DreamBooth subject subset with metadata."
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="dog",
        help="Subject subset from google/dreambooth, e.g. dog, cat, backpack.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/google_dreambooth"),
        help="Base output directory for prepared data.",
    )
    parser.add_argument(
        "--unique-token",
        type=str,
        default="sks",
        help="DreamBooth unique token used to build prompts.",
    )
    parser.add_argument(
        "--caption-template",
        type=str,
        default="a photo of {unique_token} {class_name}",
        help="Template used when building captions. Supported fields: unique_token, class_name, subject_name, stem.",
    )
    parser.add_argument(
        "--metadata-format",
        type=str,
        choices=("jsonl", "csv", "both"),
        default="both",
        help="Metadata output format.",
    )
    parser.add_argument(
        "--write-sidecar-txt",
        action="store_true",
        help="Also write one .txt caption file next to each downloaded image.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading files that already exist locally.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        subject = args.subject.strip()
        if not subject:
            raise ValueError("--subject must not be empty")

        subject_mapping = fetch_subject_mapping(timeout=args.timeout)
        if subject not in subject_mapping:
            available = ", ".join(sorted(subject_mapping))
            raise KeyError(f"Unknown subject {subject!r}. Available subjects: {available}")
        subject_info = subject_mapping[subject]

        file_entries = fetch_subject_files(subject, timeout=args.timeout)
        if not file_entries:
            raise RuntimeError(f"No image files found for subject={subject!r}")

        subject_root = (args.output_dir / subject).resolve()
        images_dir = subject_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        progress = ProgressPrinter(total=len(file_entries), prefix=f"download:{subject}")
        progress.start()

        for entry in file_entries:
            try:
                row = download_and_describe_image(
                    entry=entry,
                    subject_info=subject_info,
                    images_dir=images_dir,
                    unique_token=args.unique_token,
                    caption_template=args.caption_template,
                    write_sidecar_txt=args.write_sidecar_txt,
                    timeout=args.timeout,
                    skip_existing=args.skip_existing,
                )
                rows.append(row)
            except Exception as exc:  # pragma: no cover - runtime safety path
                progress.note(f"failed {entry.get('path', 'unknown')}: {exc}")
            finally:
                progress.advance()
        progress.finish()

        if not rows:
            raise RuntimeError("All downloads failed; no metadata was generated.")

        write_metadata(subject_root, rows, args.metadata_format)
        write_readme(subject_root, subject_info, rows, args.unique_token, args.caption_template)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "subject": subject_info.subject_name,
                    "class_name": subject_info.class_name,
                    "output_dir": str(subject_root),
                    "num_images": len(rows),
                    "metadata_format": args.metadata_format,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


def fetch_subject_mapping(*, timeout: int) -> dict[str, SubjectInfo]:
    payload = http_read_text(f"{HF_RESOLVE_ROOT}/prompts_and_classes.txt", timeout=timeout)
    lines = [line.strip() for line in payload.splitlines()]

    in_class_section = False
    mapping: dict[str, SubjectInfo] = {}
    for line in lines:
        if not line:
            continue
        if line == "Classes":
            in_class_section = True
            continue
        if line == "Prompts":
            break
        if not in_class_section or line == "subject_name,class":
            continue
        subject_name, class_name = [part.strip() for part in line.split(",", maxsplit=1)]
        mapping[subject_name] = SubjectInfo(subject_name=subject_name, class_name=class_name)

    if not mapping:
        raise RuntimeError("Failed to parse prompts_and_classes.txt")
    return mapping


def fetch_subject_files(subject: str, *, timeout: int) -> list[dict[str, Any]]:
    payload = http_read_json(f"{HF_API_ROOT}/{subject}", timeout=timeout)
    entries = [item for item in payload if item.get("type") == "file" and is_image_file(item.get("path", ""))]
    return sorted(entries, key=lambda item: item["path"])


def download_and_describe_image(
    *,
    entry: dict[str, Any],
    subject_info: SubjectInfo,
    images_dir: Path,
    unique_token: str,
    caption_template: str,
    write_sidecar_txt: bool,
    timeout: int,
    skip_existing: bool,
) -> dict[str, Any]:
    relative_name = Path(entry["path"]).name
    image_path = images_dir / relative_name
    image_url = f"{HF_RESOLVE_ROOT}/{subject_info.subject_name}/{relative_name}"

    if image_path.exists() and skip_existing:
        width, height = safe_image_size(image_path)
    else:
        download_file(
            image_url,
            image_path,
            timeout=timeout,
            expected_size=entry.get("size"),
        )
        width, height = safe_image_size(image_path)

    prompt = build_caption(
        unique_token=unique_token,
        class_name=subject_info.class_name,
        subject_name=subject_info.subject_name,
        stem=image_path.stem,
        template=caption_template,
    )

    if write_sidecar_txt:
        image_path.with_suffix(".txt").write_text(prompt + "\n", encoding="utf-8")

    return {
        "image_path": str(image_path.relative_to(Path.cwd())),
        "prompt": prompt,
        "subject_name": subject_info.subject_name,
        "class_name": subject_info.class_name,
        "width": width,
        "height": height,
        "source_url": image_url,
        "source_size_bytes": int(entry.get("size", 0)),
    }


def build_caption(
    *,
    unique_token: str,
    class_name: str,
    subject_name: str,
    stem: str,
    template: str,
) -> str:
    try:
        prompt = template.format(
            unique_token=unique_token,
            class_name=class_name,
            subject_name=subject_name,
            stem=stem,
        ).strip()
    except KeyError as exc:
        raise ValueError(f"Unsupported placeholder in --caption-template: {exc}") from exc
    if not prompt:
        raise ValueError("Generated prompt is empty")
    return prompt


def write_metadata(root: Path, rows: list[dict[str, Any]], metadata_format: str) -> None:
    if metadata_format in {"jsonl", "both"}:
        jsonl_path = root / "metadata.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if metadata_format in {"csv", "both"}:
        csv_path = root / "metadata.csv"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_readme(
    root: Path,
    subject_info: SubjectInfo,
    rows: list[dict[str, Any]],
    unique_token: str,
    caption_template: str,
) -> None:
    payload = {
        "dataset": "google/dreambooth",
        "subject_name": subject_info.subject_name,
        "class_name": subject_info.class_name,
        "num_images": len(rows),
        "unique_token": unique_token,
        "caption_template": caption_template,
    }
    (root / "dataset_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        return None, None


def download_file(url: str, destination: Path, *, timeout: int, expected_size: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=timeout) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)

    if expected_size is not None and temp_path.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded empty file from {url}")
    temp_path.replace(destination)


def http_read_json(url: str, *, timeout: int) -> Any:
    return json.loads(http_read_text(url, timeout=timeout))


def http_read_text(url: str, *, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(content_type)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while requesting {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while requesting {url}: {exc.reason}") from exc


def is_image_file(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ProgressPrinter:
    def __init__(self, *, total: int, prefix: str) -> None:
        self.total = max(total, 0)
        self.prefix = prefix
        self.current = 0
        self.last_line_time = 0.0

    def start(self) -> None:
        self._emit(force=True)

    def advance(self) -> None:
        self.current += 1
        self._emit(force=self.current >= self.total)

    def note(self, message: str) -> None:
        print(f"[{self.prefix}] {message}", file=sys.stderr)

    def finish(self) -> None:
        self._emit(force=True)

    def _emit(self, *, force: bool) -> None:
        now = time.time()
        if not force and (now - self.last_line_time) < 0.2:
            return
        width = 28
        ratio = 1.0 if self.total == 0 else min(max(self.current / self.total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{self.prefix}] [{bar}] {self.current}/{self.total}", file=sys.stderr)
        self.last_line_time = now


if __name__ == "__main__":
    raise SystemExit(main())
