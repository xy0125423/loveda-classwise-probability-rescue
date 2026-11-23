"""Historical background-only, highest-nonbackground, one-step rescue."""
import numpy as np

def validate_probabilities(prob: np.ndarray) -> None:
    """Require a finite normalized seven-channel probability raster."""
    if prob.ndim != 3 or prob.shape[0] != 7 or not np.isfinite(prob).all():
        raise ValueError('Expected finite (7,H,W) probabilities')
    if np.any(prob < 0) or np.any(prob > 1) or not np.allclose(prob.sum(0), 1, atol=1e-6):
        raise ValueError('Invalid probability distribution')

def score(prob: np.ndarray, target: int, scoring: str, arithmetic: str = 'float64_then_float32') -> np.ndarray:
    """Preserve the historical branch arithmetic as well as epsilon=1e-8."""
    if not 1 <= target <= 6: raise ValueError('Target must be nonbackground')
    if scoring == 'probability': return prob[target].astype(np.float32)
    if scoring != 'ratio': raise ValueError('Unknown scoring rule')
    if arithmetic not in ('float32', 'float64_then_float32'): raise ValueError('Unknown arithmetic')
    dtype = np.float32 if arithmetic == 'float32' else np.float64
    return np.log((prob[target].astype(dtype) + 1e-8) / (prob[0].astype(dtype) + 1e-8)).astype(np.float32)

def rescue(prob: np.ndarray, target: int, threshold: float, scoring: str,
           arithmetic: str = 'float64_then_float32') -> np.ndarray:
    """Change matching background candidates once; ties choose lowest index."""
    validate_probabilities(prob)
    if not np.isfinite(threshold): raise ValueError('Finite threshold required')
    before = prob.argmax(0).astype(np.uint8)
    candidate = 1 + prob[1:].argmax(0)
    changed = (before == 0) & (candidate == target) & (score(prob, target, scoring, arithmetic) >= threshold)
    result = before.copy(); result[changed] = target
    return result

def effects(gt: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    """Count valid changed pixels: B wrong-right, H right-wrong, W wrong-wrong."""
    if gt.shape != before.shape or gt.shape != after.shape: raise ValueError('Shape mismatch')
    changed = (gt != 255) & (before != after)
    b = int(np.count_nonzero(changed & (before != gt) & (after == gt)))
    h = int(np.count_nonzero(changed & (before == gt) & (after != gt)))
    m = int(changed.sum())
    return dict(B=b, H=h, W=m-b-h, M=m, net=b-h)
