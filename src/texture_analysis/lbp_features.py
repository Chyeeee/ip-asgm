from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def _histogram_stats(hist: np.ndarray) -> dict[str, float]:
    if hist.size == 0 or hist.sum() <= 0:
        return {"entropy": 0.0, "energy": 0.0, "dominant": 0.0, "bin_variance": 0.0}
    nz = hist[hist > 0]
    entropy = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0
    energy = float(np.sum(hist**2))
    dominant = float(np.max(hist))
    idx = np.arange(hist.size, dtype=float)
    mean_idx = float(np.sum(idx * hist))
    bin_variance = float(np.sum(((idx - mean_idx) ** 2) * hist))
    return {
        "entropy": entropy,
        "energy": energy,
        "dominant": dominant,
        "bin_variance": bin_variance,
    }


def extract_lbp_features(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    settings=((8, 1),),
    prefix="lbp",
    include_stats: bool = True,
) -> dict[str, float]:
    """Uniform LBP histogram using ROI pixels only.

    Besides the normalized histogram, compact histogram statistics are included
    to strengthen the baseline without changing the image or ROI.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = mask.astype(bool)
    result: dict[str, float] = {}

    for p, r in settings:
        lbp = local_binary_pattern(gray, P=int(p), R=float(r), method="uniform")
        values = lbp[roi]
        n_bins = int(p) + 2
        hist, _ = np.histogram(values, bins=np.arange(0, n_bins + 1), range=(0, n_bins))
        hist = hist.astype(np.float64)
        if hist.sum() > 0:
            hist /= hist.sum()

        setting_tag = "" if len(settings) == 1 else f"_p{int(p)}r{str(r).replace('.', '_')}"
        for idx, value in enumerate(hist):
            result[f"{prefix}{setting_tag}_bin_{idx}"] = float(value)
        if include_stats:
            for name, value in _histogram_stats(hist).items():
                result[f"{prefix}{setting_tag}_{name}"] = float(value)

    return result


def lbp_response_map(image_bgr: np.ndarray, mask: np.ndarray, p: int = 8, r: float = 1) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=p, R=r, method="uniform").astype(np.float32)
    response = np.zeros_like(lbp, dtype=np.float32)
    roi = mask.astype(bool)
    if np.any(roi):
        vals = lbp[roi]
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            response[roi] = (vals - lo) / (hi - lo)
    return response
