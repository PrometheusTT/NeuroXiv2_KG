
from typing import Dict, List, Sequence, Tuple
import numpy as np


def _vec(keys: Sequence[str], d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d.get(k, 0)) for k in keys], dtype=float)


def _dist(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    metric = (metric or "L1").upper()
    if metric == "L2":
        return float(np.linalg.norm(x - y))
    if metric in ("COS", "COSINE"):
        if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
            return 1.0
        return float(1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return float(np.sum(np.abs(x - y)))  # L1


def mismatch_index(
    morph_vec_a: Dict[str, float], morph_vec_b: Dict[str, float],
    subclass_vec_a: Dict[str, float], subclass_vec_b: Dict[str, float],
    metric: str = "L1"
) -> float:
    """
    | dist(morph_a, morph_b) - dist(subclass_a, subclass_b) |
    Vectors are aligned by keys with zeros filled for missing entries.
    """
    keys_m = sorted(set(morph_vec_a) | set(morph_vec_b))
    keys_s = sorted(set(subclass_vec_a) | set(subclass_vec_b))
    va, vb = _vec(keys_m, morph_vec_a), _vec(keys_m, morph_vec_b)
    sa, sb = _vec(keys_s, subclass_vec_a), _vec(keys_s, subclass_vec_b)
    return abs(_dist(va, vb, metric) - _dist(sa, sb, metric))


def basic_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
