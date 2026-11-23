"""LoveDA label mapping; Background remains an evaluated class."""
import numpy as np

CLASSES = ('Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture')

def map_labels(raw: np.ndarray) -> np.ndarray:
    """Map raw 0 to 255 and raw 1..7 to model 0..6 without underflow."""
    if not np.issubdtype(raw.dtype, np.integer) or np.any((raw < 0) | (raw > 7)):
        raise ValueError('Expected integer raw labels 0..7')
    return np.array([255, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)[raw]
