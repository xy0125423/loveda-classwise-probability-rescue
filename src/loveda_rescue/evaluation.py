"""Pixel confusion matrix and derived metrics, excluding raw-zero ignore pixels."""

from __future__ import annotations

from typing import Any
import numpy as np

CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, classes: int = 7,
                     ignore_index: int = 255) -> np.ndarray:
    """Rows=true, columns=predicted; ignored pixels contribute nothing."""
    if target.shape != prediction.shape:
        raise ValueError("Target/prediction shape mismatch")
    keep = target != ignore_index
    truth, pred = target[keep].astype(np.int64), prediction[keep].astype(np.int64)
    if np.any((truth < 0) | (truth >= classes) | (pred < 0) | (pred >= classes)):
        raise ValueError("Invalid category index")
    return np.bincount(truth * classes + pred, minlength=classes * classes).reshape(classes, classes)


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Represent undefined denominators as NaN, exported as JSON null."""
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator != 0)


def finite_values(values: np.ndarray) -> list[float | None]:
    """Convert nonfinite values to JSON-safe missing values."""
    return [float(x) if np.isfinite(x) else None for x in values]


def summarize(matrix: np.ndarray) -> dict[str, Any]:
    """Compute pooled-pixel class metrics; absent-in-both classes excluded from mIoU."""
    true_count, pred_count = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = np.diag(matrix).astype(float)
    union = true_count + pred_count - tp
    iou = divide(tp, union)
    precision, recall = divide(tp, pred_count), divide(tp, true_count)
    f1 = divide(2 * tp, true_count + pred_count)
    return {"classes": CLASSES, "matrix_rows_true_columns_predicted": matrix.tolist(),
            "IoU": finite_values(iou), "Precision": finite_values(precision),
            "Recall": finite_values(recall), "F1": finite_values(f1),
            "mIoU": float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
            "OA": float(tp.sum() / matrix.sum()) if matrix.sum() else None,
            "valid_pixels": int(matrix.sum()), "true_pixels_per_class": true_count.tolist(),
            "macro_rule": "undefined denominators=null; mIoU excludes only zero-union classes",
            "units": "fractions, not percentages"}
