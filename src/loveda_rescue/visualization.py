"""Visualize label-change status without input imagery."""
from pathlib import Path
import numpy as np
from PIL import Image

def save_effects(gt: np.ndarray, before: np.ndarray, after: np.ndarray, path: Path) -> None:
    """Gray unchanged, blue B, orange H, purple W, white ignored."""
    rgb = np.full((*gt.shape, 3), 235, np.uint8)
    changed = (gt != 255) & (before != after)
    rgb[changed] = [204, 121, 167]
    rgb[changed & (after == gt)] = [0, 114, 178]
    rgb[changed & (before == gt)] = [213, 94, 0]
    rgb[gt == 255] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).resize((max(320, gt.shape[1]*20), max(160, gt.shape[0]*20)), resample=Image.Resampling.NEAREST).save(path)
