from __future__ import annotations

from itertools import combinations_with_replacement

import cv2
import numpy as np


# Classic Laws' 5-tap texture vectors.
_LAWS_VECTORS: dict[str, np.ndarray] = {
    "L5": np.asarray([1, 4, 6, 4, 1], dtype=np.float32),
    "E5": np.asarray([-1, -2, 0, 2, 1], dtype=np.float32),
    "S5": np.asarray([-1, 0, 2, 0, -1], dtype=np.float32),
    "W5": np.asarray([-1, 2, 0, -2, 1], dtype=np.float32),
    "R5": np.asarray([1, -4, 6, -4, 1], dtype=np.float32),
}

# L5L5 mainly represents local level/illumination rather than texture, so it is
# intentionally excluded. Off-diagonal pairs are symmetrized by averaging the
# two transposed filter responses, which keeps the descriptor compact.
_LAWS_PAIRS = [
    pair
    for pair in combinations_with_replacement(_LAWS_VECTORS.keys(), 2)
    if pair != ("L5", "L5")
]


def _roi_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _normalised_kernel(a: str, b: str) -> np.ndarray:
    kernel = np.outer(_LAWS_VECTORS[a], _LAWS_VECTORS[b]).astype(np.float32)
    norm = float(np.sum(np.abs(kernel)))
    if norm > 1e-12:
        kernel /= norm
    return kernel


def _rms_response(image: np.ndarray, kernel: np.ndarray, roi: np.ndarray) -> float:
    response = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    vals = response[roi]
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals * vals)))


def extract_laws_texture_features(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Extract compact Laws texture-energy evidence inside the existing ROI.

    The strong colour+GLCM+LBP baseline is untouched. Laws descriptors are only
    candidate evidence for PTD+ pair-specific specialists. Illumination is
    suppressed by subtracting a local mean, and the residual is normalized by
    ROI standard deviation before the classical 5x5 Laws filters are applied.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    roi = mask.astype(bool)
    bbox = _roi_bbox(roi)
    if bbox is None:
        return {}

    x0, y0, x1, y1 = bbox
    gray_crop = gray[y0:y1, x0:x1]
    roi_crop = roi[y0:y1, x0:x1]
    if gray_crop.size == 0 or int(roi_crop.sum()) < 25:
        return {}

    # Classic Laws texture energy is computed on a locally mean-removed image.
    # A 15x15 mean window is large enough to remove slow illumination changes
    # while retaining the micro-texture that the 5x5 kernels measure.
    local_mean = cv2.blur(gray_crop, (15, 15), borderType=cv2.BORDER_REFLECT)
    residual = gray_crop - local_mean
    roi_vals = residual[roi_crop]
    scale = float(np.std(roi_vals)) if roi_vals.size else 0.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    residual = residual / scale

    result: dict[str, float] = {}
    energies: list[float] = []
    for a, b in _LAWS_PAIRS:
        e1 = _rms_response(residual, _normalised_kernel(a, b), roi_crop)
        if a == b:
            energy = e1
        else:
            e2 = _rms_response(residual, _normalised_kernel(b, a), roi_crop)
            energy = 0.5 * (e1 + e2)
        name = f"laws_{a.lower()}{b.lower()}_energy"
        result[name] = float(energy)
        energies.append(float(energy))

    # Two compact bank-level summaries are useful when no single filter pair is
    # dominant. They remain texture-only and add negligible dimensionality.
    arr = np.asarray(energies, dtype=float)
    result["laws_bank_mean_energy"] = float(np.mean(arr)) if arr.size else 0.0
    result["laws_bank_std_energy"] = float(np.std(arr)) if arr.size else 0.0
    return result
