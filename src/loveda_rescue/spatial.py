"""Extracted historical B2/C4 feature and gate implementation."""
import numpy as np
from scipy import ndimage as ndi
BINS = 4096
METHODS = {"C4": ("mixed", True, True)}

def features(prob: np.ndarray, rgb: np.ndarray) -> dict[str, np.ndarray]:
    """GT-free features and simultaneous replacement evidence from original predictions."""
    pred = prob.argmax(0).astype(np.uint8)
    maximum = prob.max(0)
    ordered = np.partition(prob, 5, axis=0)
    local_prob = np.stack([ndi.uniform_filter(p, size=5, mode="reflect") for p in prob])
    candidate = local_prob.argmax(0).astype(np.uint8)
    votes = np.stack([ndi.uniform_filter((pred == c).astype(np.float32), size=5,
                                      mode="reflect") for c in range(7)])
    support = np.take_along_axis(votes, candidate[None], axis=0)[0]
    conflict = 1 - np.take_along_axis(votes, pred[None], axis=0)[0]
    alt_prob = np.take_along_axis(prob, candidate[None], axis=0)[0]
    gray = rgb.astype(np.float32).mean(2) / 255
    edge = np.hypot(ndi.sobel(gray, axis=0, mode="reflect"),
                    ndi.sobel(gray, axis=1, mode="reflect")) / 8
    strong = edge >= np.quantile(edge, .90)
    strong = ndi.binary_dilation(strong, iterations=1)
    return dict(pred=pred, candidate=candidate, majority=votes.argmax(0).astype(np.uint8),
                support=support, gap=np.maximum(0, maximum-alt_prob), conflict=conflict,
                confidence=1-maximum,
                entropy=-(prob*np.log(np.maximum(prob, 1e-12))).sum(0)/np.log(7),
                margin=1-(ordered[6]-ordered[5]), mixed=.5*(1-maximum)+.5*conflict,
                strong_edge=strong)

def score_bins(score: np.ndarray) -> np.ndarray:
    """Quantize risk only for auditable histogram metrics and threshold gates."""
    return np.minimum((np.clip(score, 0, 1)*BINS).astype(np.int32), BINS-1)

def gate(f: dict, method: str, parameter: dict) -> np.ndarray:
    """Apply frozen thresholds using predicted class only, without GT."""
    feature, adaptive, edge = METHODS[method]
    bins = score_bins(f[feature])
    cut = np.array(parameter["class_threshold_bins"])[f["pred"]] if adaptive else parameter["global_threshold_bin"]
    return ((f["candidate"] != f["pred"]) & (f["support"] >= parameter["support"]) &
            (f["gap"] <= parameter["probability_gap"]) & (bins >= cut) &
            (~f["strong_edge"] if edge else True))

def spatial_prediction(prob: np.ndarray, rgb: np.ndarray, method: str, parameters: dict) -> np.ndarray:
    """B2 is 5x5 majority; C4 applies frozen support/gap/risk/edge gates."""
    f = features(prob, rgb)
    if method == 'B2': return f['majority']
    if method != 'C4': raise ValueError('Expected B2 or C4')
    mask = gate(f, method, parameters)
    result = f['pred'].copy(); result[mask] = f['candidate'][mask]
    return result
