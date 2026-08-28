from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

from .config import GLCM_ANGLES_DEG, GLCM_LEVELS, GLCM_PROPERTIES


def quantize_gray(gray: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    q = (gray.astype(np.uint16) * levels // 256).astype(np.uint8)
    return np.clip(q, 0, levels - 1)


def masked_glcm(
    gray: np.ndarray,
    mask: np.ndarray,
    distance: int,
    angle_deg: float,
    levels: int = GLCM_LEVELS,
    symmetric: bool = True,
) -> np.ndarray:
    """Build GLCM only from neighbour pairs where BOTH pixels are inside ROI."""
    q = quantize_gray(gray, levels)
    m = mask.astype(bool)

    theta = math.radians(angle_deg)
    dx = int(round(distance * math.cos(theta)))
    dy = int(round(-distance * math.sin(theta)))
    if dx == 0 and dy == 0:
        dx = distance

    h, w = gray.shape
    y0 = max(0, -dy)
    y1 = min(h, h - dy)
    x0 = max(0, -dx)
    x1 = min(w, w - dx)
    if y1 <= y0 or x1 <= x0:
        return np.zeros((levels, levels), dtype=np.float64)

    src_q = q[y0:y1, x0:x1]
    dst_q = q[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    valid = m[y0:y1, x0:x1] & m[y0 + dy : y1 + dy, x0 + dx : x1 + dx]

    i = src_q[valid].astype(np.int64)
    j = dst_q[valid].astype(np.int64)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    if i.size == 0:
        return matrix

    np.add.at(matrix, (i, j), 1.0)
    if symmetric:
        np.add.at(matrix, (j, i), 1.0)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def glcm_properties(matrix: np.ndarray) -> dict[str, float]:
    if matrix.sum() <= 0:
        return {k: 0.0 for k in GLCM_PROPERTIES}

    n = matrix.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    diff = ii - jj

    contrast = float(np.sum(matrix * diff**2))
    dissimilarity = float(np.sum(matrix * np.abs(diff)))
    homogeneity = float(np.sum(matrix / (1.0 + diff**2)))
    asm = float(np.sum(matrix**2))
    energy = float(np.sqrt(asm))

    px = matrix.sum(axis=1)
    py = matrix.sum(axis=0)
    idx = np.arange(n, dtype=float)
    mu_x = float(np.sum(idx * px))
    mu_y = float(np.sum(idx * py))
    sigma_x = float(np.sqrt(np.sum(((idx - mu_x) ** 2) * px)))
    sigma_y = float(np.sqrt(np.sum(((idx - mu_y) ** 2) * py)))
    if sigma_x > 1e-12 and sigma_y > 1e-12:
        correlation = float(
            np.sum(matrix * (ii - mu_x) * (jj - mu_y)) / (sigma_x * sigma_y)
        )
    else:
        correlation = 1.0

    nz = matrix[matrix > 0]
    entropy = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0

    # Joint grey-level variance. With a symmetric GLCM, row/column variance are
    # equivalent, so one compact descriptor is sufficient.
    joint_mean = float(np.sum(matrix * ii))
    variance = float(np.sum(matrix * (ii - joint_mean) ** 2))

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "asm": asm,
        "entropy": entropy,
        "variance": variance,
    }


def extract_glcm_features(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    distances=(1,),
    angles=GLCM_ANGLES_DEG,
    prefix="glcm",
    keep_per_distance: bool = False,
    include_direction_std: bool = False,
) -> dict[str, float]:
    """Extract masked GLCM descriptors.

    Strong baseline:
      * distance 1
      * 4 directions
      * 8 GLCM properties
      * directional mean + directional standard deviation

    Proposed BP-PTD texture specialist:
      * distances 1, 2 and 3 are kept separately
      * the same directional statistics are retained at every scale
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    by_distance: dict[int, dict[str, list[float]]] = {
        int(d): defaultdict(list) for d in distances
    }

    for d in distances:
        for angle in angles:
            matrix = masked_glcm(gray, mask, int(d), float(angle))
            props = glcm_properties(matrix)
            for name, value in props.items():
                by_distance[int(d)][name].append(value)

    result: dict[str, float] = {}
    if keep_per_distance:
        for d in distances:
            for prop in GLCM_PROPERTIES:
                values = np.asarray(by_distance[int(d)][prop], dtype=float)
                mean_value = float(np.mean(values)) if values.size else 0.0
                if include_direction_std:
                    result[f"{prefix}_d{int(d)}_{prop}_mean"] = mean_value
                    result[f"{prefix}_d{int(d)}_{prop}_dir_std"] = (
                        float(np.std(values)) if values.size else 0.0
                    )
                else:
                    result[f"{prefix}_d{int(d)}_{prop}"] = mean_value
    else:
        # For one-distance baselines, preserve both the mean and directional
        # variation instead of collapsing everything into only one number.
        for prop in GLCM_PROPERTIES:
            values = np.asarray(
                [v for d in distances for v in by_distance[int(d)][prop]], dtype=float
            )
            mean_value = float(np.mean(values)) if values.size else 0.0
            if include_direction_std:
                result[f"{prefix}_{prop}_mean"] = mean_value
                result[f"{prefix}_{prop}_dir_std"] = (
                    float(np.std(values)) if values.size else 0.0
                )
            else:
                result[f"{prefix}_{prop}"] = mean_value
    return result
