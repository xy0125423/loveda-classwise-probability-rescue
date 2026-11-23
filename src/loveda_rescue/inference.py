"""Overlapping sliding-window logits fusion; no training or spatial label filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def window_starts(length: int, window: int, stride: int) -> list[int]:
    """Cover the full axis, including a final window aligned to its far edge."""
    if length < window or not 0 < stride < window:
        raise ValueError("Require length >= window and 0 < stride < window")
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def sliding_logits(model: torch.nn.Module, rgb: np.ndarray,
                   mean: list[float], std: list[float], window: int = 256,
                   stride: int = 128, batch_size: int = 2,
                   classes: int = 7) -> tuple[np.ndarray, torch.Tensor, dict[str, Any]]:
    """Fuse raw logits with positive Hann weights before a single final argmax.

    Half-window reflect padding supplies context at original image borders.
    Windows are normalized exactly as in training. Weighting is independent of
    predictions and labels; no parameter is selected against reference accuracy.
    """
    if model.training:
        raise ValueError("Inference requires model.eval()")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected RGB uint8")
    if window % 2 or batch_size < 1:
        raise ValueError("Even window and positive batch size required")
    pad = window // 2
    original_h, original_w = rgb.shape[:2]
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    height, width = padded.shape[:2]
    ys, xs = window_starts(height, window, stride), window_starts(width, window, stride)
    axis = torch.hann_window(window, periodic=False).clamp_min(1e-3)
    weights = axis[:, None] * axis[None, :]
    accumulated = torch.zeros((classes, height, width), dtype=torch.float32)
    denominator = torch.zeros((height, width), dtype=torch.float32)
    coverage = torch.zeros((height, width), dtype=torch.int16)
    mean_t = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std_t = torch.tensor(std, dtype=torch.float32)[:, None, None]
    pending, positions = [], []

    def flush() -> None:
        if not pending:
            return
        logits = model(torch.stack(pending))
        if logits.shape != (len(pending), classes, window, window) or not torch.isfinite(logits).all():
            raise ValueError("Invalid model logits")
        for index, (top, left) in enumerate(positions):
            accumulated[:, top:top + window, left:left + window] += logits[index].cpu() * weights
            denominator[top:top + window, left:left + window] += weights
            coverage[top:top + window, left:left + window] += 1
        pending.clear()
        positions.clear()

    with torch.inference_mode():
        for top in ys:
            for left in xs:
                crop = padded[top:top + window, left:left + window]
                tensor = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255.0
                pending.append((tensor - mean_t) / std_t)
                positions.append((top, left))
                if len(pending) == batch_size:
                    flush()
        flush()
        if not torch.all(denominator > 0):
            raise AssertionError("Uncovered inference pixels")
        fused = accumulated / denominator[None]
        fused = fused[:, pad:pad + original_h, pad:pad + original_w].clone()
        retained_coverage = coverage[pad:pad + original_h, pad:pad + original_w]
        retained_weight = denominator[pad:pad + original_h, pad:pad + original_w]
        if not torch.isfinite(fused).all():
            raise AssertionError("Nonfinite fused output")
        prediction = fused.argmax(dim=0).numpy().astype(np.uint8)
    return prediction, fused, {
        "window": window, "stride": stride, "overlap_fraction": 1 - stride / window,
        "padding": "reflect", "padding_pixels": pad, "windows": len(ys) * len(xs),
        "fusion": "sum(Hann_2D * raw_logits) / sum(Hann_2D); then argmax",
        "hann_axis_minimum": 1e-3, "original_shape": [original_h, original_w],
        "coverage_min": int(retained_coverage.min()), "coverage_max": int(retained_coverage.max()),
        "weight_sum_min": float(retained_weight.min()), "weight_sum_max": float(retained_weight.max()),
    }
