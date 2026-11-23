"""Historical training epoch and source-selection validation functions."""
from __future__ import annotations
import hashlib
import logging
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .training_data import LoveDAStore, TrainingDataset, map_labels
from .selection import predict_full
from .evaluation import confusion_matrix, summarize
EXCLUDED = {'3748.png', '4137.png'}

def train_epoch(model: torch.nn.Module, store: LoveDAStore, config: dict,
                epoch: int, optimizer: torch.optim.Optimizer) -> dict:
    """Use the existing crop/augmentation and effective-pixel CE; add online metrics."""
    dataset = TrainingDataset(store, config, config["seed"])
    dataset.set_epoch(epoch)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"] + epoch),
                        num_workers=0, drop_last=False)
    model.train()
    matrix = np.zeros((7, 7), dtype=np.int64)
    total_loss, valid_pixels, updates, skipped = 0., 0, 0, 0
    seen = []
    start = time.perf_counter()
    for batch, (images, targets, names) in enumerate(loader, 1):
        seen.extend(names)
        optimizer.zero_grad(set_to_none=True)
        count = int((targets != 255).sum())
        if count == 0:
            skipped += 1
            continue
        logits = model(images)
        assert logits.shape == (len(images), 7, config["crop_size"], config["crop_size"])
        loss_sum = F.cross_entropy(logits, targets, ignore_index=255, reduction="sum")
        loss = loss_sum / count
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if updates == 0 and not all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad):
            raise ValueError("Missing/nonfinite gradients")
        matrix += confusion_matrix(targets.numpy(), logits.detach().argmax(1).numpy())
        optimizer.step()
        total_loss += loss_sum.item()
        valid_pixels += count
        updates += 1
        if batch == 1 or batch % 50 == 0 or batch == len(loader):
            logging.info("TRAIN epoch=%d batch=%d/%d images=%d mean_loss=%.6f elapsed=%.1fs",
                         epoch, batch, len(loader), len(seen), total_loss / valid_pixels,
                         time.perf_counter() - start)
    assert len(seen) == len(set(seen)) == len(store.names) and set(seen) == set(store.names)
    assert matrix.sum() == valid_pixels and updates > 0
    result = summarize(matrix)
    result.update(loss=total_loss/valid_pixels, images=len(seen), updates=updates,
                  all_ignore_batches_skipped=skipped, seconds=time.perf_counter()-start,
                  training_order=seen,
                  training_order_sha256=hashlib.sha256("\n".join(seen).encode()).hexdigest(),
                  scope="online train-mode crop predictions BEFORE each batch update; not a fixed-checkpoint evaluation")
    return result

def validate(model: torch.nn.Module, store: LoveDAStore, names: list[str], config: dict,
             epoch: int) -> tuple[dict, dict, list[dict]]:
    """One pass over the fixed source selection set; retain per-image sufficient statistics."""
    model.eval()
    matrix = np.zeros((7, 7), dtype=np.int64)
    clean = matrix.copy()
    loss_sum, clean_loss, pixels, clean_pixels = 0., 0., 0, 0
    rows = []
    start = time.perf_counter()
    with torch.inference_mode():
        for index, name in enumerate(names, 1):
            rgb, raw = store.read(name)
            target = map_labels(raw)
            prediction, logits = predict_full(model, rgb, config)
            cm = confusion_matrix(target, prediction)
            count = int((target != 255).sum())
            loss = F.cross_entropy(logits[None], torch.from_numpy(target)[None],
                                   ignore_index=255, reduction="sum").item()
            assert np.isfinite(loss) and cm.sum() == count
            matrix += cm
            loss_sum += loss
            pixels += count
            if name not in EXCLUDED:
                clean += cm
                clean_loss += loss
                clean_pixels += count
            rows.append(dict(name=name, excluded_from_sensitivity=name in EXCLUDED,
                             valid_pixels=count, loss_sum=loss,
                             matrix_rows_true_columns_predicted=cm.tolist()))
            if index == 1 or index % 25 == 0 or index == len(names):
                logging.info("VAL epoch=%d image=%d/%d elapsed=%.1fs", epoch, index,
                             len(names), time.perf_counter()-start)
    full, reduced = summarize(matrix), summarize(clean)
    full.update(loss=loss_sum/pixels, images=len(names), seconds=time.perf_counter()-start)
    reduced.update(loss=clean_loss/clean_pixels, images=sum(n not in EXCLUDED for n in names))
    assert np.array_equal(matrix, sum(np.array(r["matrix_rows_true_columns_predicted"]) for r in rows))
    return full, reduced, rows
