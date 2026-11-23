"""Full-image source evaluation using deterministic, nonoverlapping inference tiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .training_data import LoveDAStore, map_labels, normalize
from .evaluation import confusion_matrix, summarize


def predict_full(model: torch.nn.Module, rgb: np.ndarray,
                 config: dict[str, Any]) -> tuple[np.ndarray, torch.Tensor]:
    """Reassemble logits at original dimensions; zero-pad partial edge tiles only."""
    height, width = rgb.shape[:2]
    tile = config["evaluation_tile_size"]
    output = torch.empty((config["num_classes"], height, width), dtype=torch.float32)
    pending, positions = [], []

    def flush() -> None:
        if not pending:
            return
        logits = model(torch.stack(pending))
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite validation output")
        for index, (top, left, h, w) in enumerate(positions):
            output[:, top:top + h, left:left + w] = logits[index, :, :h, :w].cpu()
        pending.clear()
        positions.clear()

    for top in range(0, height, tile):
        for left in range(0, width, tile):
            patch = rgb[top:top + tile, left:left + tile]
            h, w = patch.shape[:2]
            padded = np.pad(patch, ((0, tile - h), (0, tile - w), (0, 0)), mode="edge")
            pending.append(normalize(padded, config["normalization_mean"], config["normalization_std"]))
            positions.append((top, left, h, w))
            if len(pending) == config["inference_batch_size"]:
                flush()
    flush()
    return output.argmax(dim=0).numpy().astype(np.uint8), output


def evaluate_source(model: torch.nn.Module, store: LoveDAStore, names: list[str],
                    config: dict[str, Any], export_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate only the caller-supplied source selection names; never find target data."""
    model.eval()
    matrix = np.zeros((7, 7), dtype=np.int64)
    total_loss, valid_pixels = 0.0, 0
    exports = []
    with torch.inference_mode():
        for index, name in enumerate(names):
            rgb, raw = store.read(name)
            target = map_labels(raw)
            prediction, logits = predict_full(model, rgb, config)
            matrix += confusion_matrix(target, prediction)
            count = int((target != config["ignore_index"]).sum())
            total_loss += F.cross_entropy(logits[None], torch.from_numpy(target)[None],
                                         ignore_index=config["ignore_index"], reduction="sum").item()
            valid_pixels += count
            if export_dir is not None and index < config["trial_export_images"]:
                export_dir.mkdir(parents=True, exist_ok=True)
                paths = {kind: export_dir / f"{Path(name).stem}_{kind}.png"
                         for kind in ("RGB", "GT_raw", "prediction_index")}
                for kind, array in (("RGB", rgb), ("GT_raw", raw), ("prediction_index", prediction)):
                    if paths[kind].exists():
                        raise FileExistsError(paths[kind])
                    Image.fromarray(array).save(paths[kind])
                exports.append({"name": name, "source": store.location,
                                "size": list(raw.shape), "CRS": None,
                                "truth_encoding": "LoveDA raw 0..7; 0 ignored",
                                "prediction_encoding": "model 0..6",
                                "files": {k: str(v) for k, v in paths.items()}})
    result = summarize(matrix)
    result.update(loss=total_loss / valid_pixels if valid_pixels else None,
                  images=len(names), names=names, evaluation="full image, nonoverlapping tiles; no TTA")
    if export_dir is not None:
        with (export_dir / "QGIS导入清单.json").open("x", encoding="utf-8") as stream:
            json.dump({"exports": exports, "georeferencing": "None: pixel coordinates only; no fabricated CRS"},
                      stream, ensure_ascii=False, indent=2)
    return result
