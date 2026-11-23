"""Source-only exact quantile calibration; no candidate-only negative pool."""
from pathlib import Path
import numpy as np
from .rescue import score, validate_probabilities

def calibrate(records: list[Path], target: int, scoring: str, arithmetic: str = 'float64_then_float32') -> float:
    """Fit q=.999 linear quantile of all valid source GT!=target scores.

    Reference implementation loads negative scores in RAM. Large datasets need
    sufficient RAM; no histogram approximation or subsampling is substituted.
    """
    negatives = []
    for path in records:
        with np.load(path, allow_pickle=False) as item:
            p, gt = item['prob'], item['gt']
            validate_probabilities(p)
            if gt.shape != p.shape[1:]: raise ValueError('Shape mismatch')
            negatives.append(score(p, target, scoring, arithmetic)[(gt != 255) & (gt != target)])
    values = np.concatenate(negatives) if negatives else np.empty(0)
    if not values.size: raise ValueError('Empty source negative pool')
    return float(np.quantile(values, .999, method='linear'))
