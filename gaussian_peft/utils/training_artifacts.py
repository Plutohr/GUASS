from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


class TrainingArtifactWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        plot_every: int = 20,
        smoothing_window: int = 20,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path = self.output_dir / "metrics.csv"
        self.loss_curve_path = self.output_dir / "loss_curve.png"
        self.plot_every = max(int(plot_every), 1)
        self.smoothing_window = max(int(smoothing_window), 1)
        self._fieldnames: list[str] | None = None
        self._rows: list[dict[str, float]] = []

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        row = {"step": float(step)}
        row.update({key: float(value) for key, value in metrics.items()})
        self._rows.append(row)
        self._append_csv_row(row)
        if step == 1 or step % self.plot_every == 0:
            self.render_loss_curve()

    def finalize(self) -> None:
        if self._rows:
            self.render_loss_curve()

    def _append_csv_row(self, row: dict[str, float]) -> None:
        fieldnames = list(row.keys())
        if self._fieldnames is None:
            self._fieldnames = fieldnames
            with self.metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        if fieldnames != self._fieldnames:
            raise ValueError(
                "Metric keys changed during training. "
                f"expected {self._fieldnames}, got {fieldnames}"
            )

        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow(row)

    def render_loss_curve(self) -> None:
        losses = [row["loss"] for row in self._rows if "loss" in row]
        if not losses:
            return

        smoothed = moving_average(losses, self.smoothing_window)
        image = Image.new("RGB", (1280, 720), color="white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        left = 90
        right = 40
        top = 60
        bottom = 100
        width = image.width - left - right
        height = image.height - top - bottom

        draw.rectangle([(left, top), (left + width, top + height)], outline="black", width=2)

        y_min = min(min(losses), min(smoothed))
        y_max = max(max(losses), max(smoothed))
        if y_max <= y_min:
            y_max = y_min + 1.0
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

        for tick in range(6):
            y_value = y_min + (y_max - y_min) * (1.0 - tick / 5.0)
            y = top + height * tick / 5.0
            draw.line([(left, y), (left + width, y)], fill=(225, 225, 225), width=1)
            draw.text((12, y - 8), f"{y_value:.4f}", fill="black", font=font)

        max_step = max(len(losses) - 1, 1)
        for tick in range(6):
            x = left + width * tick / 5.0
            draw.line([(x, top), (x, top + height)], fill=(235, 235, 235), width=1)
            step_value = 1 + round(max_step * tick / 5.0)
            draw.text((x - 10, top + height + 12), str(step_value), fill="black", font=font)

        raw_points = build_points(losses, left, top, width, height, y_min, y_max)
        smooth_points = build_points(smoothed, left, top, width, height, y_min, y_max)

        if len(raw_points) >= 2:
            draw.line(raw_points, fill=(43, 108, 176), width=2)
        if len(smooth_points) >= 2:
            draw.line(smooth_points, fill=(220, 38, 38), width=3)

        final_loss = losses[-1]
        best_loss = min(losses)
        title = "Training Loss Curve"
        summary = (
            f"steps={len(losses)}  final_loss={final_loss:.6f}  "
            f"best_loss={best_loss:.6f}  smooth_window={self.smoothing_window}"
        )
        legend_raw = "blue: raw loss"
        legend_smooth = "red: moving average"

        draw.text((left, 18), title, fill="black", font=font)
        draw.text((left, 36), summary, fill="black", font=font)
        draw.text((left, image.height - 32), legend_raw, fill=(43, 108, 176), font=font)
        draw.text((left + 180, image.height - 32), legend_smooth, fill=(220, 38, 38), font=font)
        draw.text((left + width // 2 - 35, image.height - 52), "step", fill="black", font=font)
        draw.text((10, top - 22), "loss", fill="black", font=font)

        image.save(self.loss_curve_path)


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        denom = min(index + 1, window)
        result.append(running / denom)
    return result


def build_points(
    values: list[float],
    left: int,
    top: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    if len(values) == 1:
        x = left + width // 2
        y = top + height // 2
        return [(x, y)]

    points: list[tuple[int, int]] = []
    denom = max(len(values) - 1, 1)
    for index, value in enumerate(values):
        x = left + round(width * index / denom)
        normalized = (value - y_min) / max(y_max - y_min, 1e-12)
        y = top + round(height * (1.0 - normalized))
        points.append((x, y))
    return points
