"""On-demand paired LoveDA loading, reproducible epoch-varying augmentation."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import random
from typing import Any
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LoveDAStore:
    """Prefer existing project files; use read-only ZIP only if folders are absent."""

    def __init__(self, root: str, zip_root: str, split: str, domain: str) -> None:
        if split not in ("Train", "Val") or domain not in ("Urban", "Rural"):
            raise ValueError((split, domain))
        self.split, self.domain = split, domain
        candidates = [Path(root) / split / split / domain, Path(root) / split / domain]
        self.folder = next((p for p in candidates if (p / "images_png").is_dir()
                            and (p / "masks_png").is_dir()), None)
        self.archive = Path(zip_root) / f"{split}.zip"
        if self.folder is not None:
            images = [p.name for p in (self.folder / "images_png").glob("*.png")]
            masks = [p.name for p in (self.folder / "masks_png").glob("*.png")]
            self.location = str(self.folder)
            self.backend = "project_files"
        else:
            with zipfile.ZipFile(self.archive) as source:
                names = source.namelist()
            ip, mp = f"{split}/{domain}/images_png/", f"{split}/{domain}/masks_png/"
            images = [n[len(ip):] for n in names if n.startswith(ip) and n.endswith(".png")]
            masks = [n[len(mp):] for n in names if n.startswith(mp) and n.endswith(".png")]
            self.location = str(self.archive)
            self.backend = "readonly_zip"
        if not images or len(images) != len(set(images)) or len(masks) != len(set(masks)):
            raise ValueError("Empty or duplicate image/mask entries")
        if set(images) != set(masks):
            raise ValueError(f"Unpaired files: {split}/{domain}")
        self.names = sorted(images)

    def describe(self) -> dict[str, Any]:
        """Return name-level evidence without loading image/label pixels."""
        return {"split": self.split, "domain": self.domain, "location": self.location,
                "backend": self.backend, "images": len(self.names), "masks": len(self.names),
                "unpaired": 0,
                "names_sha256": hashlib.sha256("\n".join(self.names).encode()).hexdigest()}

    def read(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read one pair without caching; preserve raw 0..7 labels."""
        if self.folder is not None:
            with Image.open(self.folder / "images_png" / name) as image:
                if image.mode != "RGB":
                    raise ValueError(f"Expected RGB: {name}, {image.mode}")
                rgb = np.array(image)
            with Image.open(self.folder / "masks_png" / name) as mask:
                if mask.mode != "L":
                    raise ValueError(f"Expected L: {name}, {mask.mode}")
                raw = np.array(mask)
        else:
            with zipfile.ZipFile(self.archive) as source:
                prefix = f"{self.split}/{self.domain}"
                with Image.open(io.BytesIO(source.read(f"{prefix}/images_png/{name}"))) as image:
                    if image.mode != "RGB":
                        raise ValueError("Expected RGB")
                    rgb = np.array(image)
                with Image.open(io.BytesIO(source.read(f"{prefix}/masks_png/{name}"))) as mask:
                    if mask.mode != "L":
                        raise ValueError("Expected L mask")
                    raw = np.array(mask)
        if rgb.dtype != np.uint8 or raw.dtype != np.uint8:
            raise ValueError("Expected uint8 input")
        if rgb.shape[:2] != raw.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Image/mask shape mismatch")
        if raw.min() < 0 or raw.max() > 7:
            raise ValueError(f"Unexpected raw labels in {name}")
        return rgb, raw


def map_labels(raw: np.ndarray) -> np.ndarray:
    """Map raw 0 to ignore=255, raw 1..7 to model 0..6; avoid uint8 underflow."""
    return np.array([255, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)[raw]


def geometry(array: np.ndarray, top: int, left: int, size: int,
             horizontal: bool, vertical: bool) -> np.ndarray:
    """Shared spatial transform for RGB and raw labels; no interpolation."""
    result = array[top:top + size, left:left + size]
    if horizontal:
        result = result[:, ::-1]
    if vertical:
        result = result[::-1]
    return np.ascontiguousarray(result)


def augmentation_params(seed: int, epoch: int, index: int, height: int,
                        width: int, size: int) -> tuple[int, int, int, bool, bool]:
    """Use epoch-aware independent RNG; reproducible regardless of loader order."""
    if min(height, width) < size:
        raise ValueError("Input smaller than configured crop")
    rng = random.Random(f"{seed}:{epoch}:{index}")
    return (rng.randint(0, height - size), rng.randint(0, width - size), size,
            rng.random() < 0.5, rng.random() < 0.5)


def normalize(rgb: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    """Use ImageNet weight-specific mean/std after uint8-to-[0,1] conversion."""
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    return (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class TrainingDataset(Dataset):
    """One crop per original source image per epoch, without whole-dataset cache."""

    def __init__(self, store: LoveDAStore, config: dict[str, Any], seed: int) -> None:
        self.store, self.config, self.seed = store, config, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the crop/flip RNG before each epoch (workers recreated each time)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.store.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name = self.store.names[index]
        rgb, raw = self.store.read(name)
        params = augmentation_params(self.seed, self.epoch, index, *raw.shape,
                                     self.config["crop_size"])
        rgb, raw = geometry(rgb, *params), geometry(raw, *params)
        return (normalize(rgb, self.config["normalization_mean"], self.config["normalization_std"]),
                torch.from_numpy(map_labels(raw)), name)


def build_splits(stores: dict[str, LoveDAStore], config: dict[str, Any]) -> dict[str, Any]:
    """Partition source Val names independently per direction, without reading labels."""
    result = {}
    for direction, spec in config["directions"].items():
        store = stores[f"Val/{spec['source']}"]
        names = store.names.copy()
        random.Random(spec["split_seed"]).shuffle(names)
        count = int(len(names) * config["model_selection_fraction"])
        if not 0 < count < len(names):
            raise ValueError("Both source subsets must be nonempty")
        selection, calibration = names[:count], names[count:]
        assert not set(selection) & set(calibration)
        assert set(selection) | set(calibration) == set(store.names)
        result[direction] = {**spec, "model_selection": selection, "calibration": calibration,
                             "source_val_manifest": store.describe(),
                             "target_policy": "final evaluation only; no target pixel reads here"}
    return result
