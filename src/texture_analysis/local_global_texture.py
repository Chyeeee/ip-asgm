from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from .glcm_features import masked_glcm, glcm_properties


LOCAL_GRID_ROWS = 3
LOCAL_GRID_COLS = 3
LOCAL_GLCM_ANGLES = (0, 45, 90, 135)
LOCAL_GLCM_PROPERTIES = ("contrast", "homogeneity", "energy", "entropy", "variance")
LOCAL_MIN_ROI_PIXELS = 24
GABOR_ORIENTATIONS_DEG = (0, 45, 90, 135)
GABOR_WAVELENGTHS = (4.0, 8.0)


def _aggregate(prefix: str, values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
    }


def _roi_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _patch_slices(length: int, parts: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, parts + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(parts) if edges[i + 1] > edges[i]]


def _lbp_hist_stats(values: np.ndarray, p: int) -> dict[str, float]:
    n_bins = int(p) + 2
    if values.size == 0:
        return {"entropy": 0.0, "energy": 0.0, "dominant": 0.0, "bin_variance": 0.0}
    hist, _ = np.histogram(values, bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    nz = hist[hist > 0]
    entropy = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0
    energy = float(np.sum(hist**2))
    dominant = float(np.max(hist)) if hist.size else 0.0
    idx = np.arange(hist.size, dtype=float)
    mean_idx = float(np.sum(idx * hist))
    bin_variance = float(np.sum(((idx - mean_idx) ** 2) * hist))
    return {
        "entropy": entropy,
        "energy": energy,
        "dominant": dominant,
        "bin_variance": bin_variance,
    }


def extract_local_texture_features(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    grid_rows: int = LOCAL_GRID_ROWS,
    grid_cols: int = LOCAL_GRID_COLS,
) -> dict[str, float]:
    """Describe how texture varies across fixed spatial regions of the existing ROI.

    This is texture analysis only: the incoming image and ROI are used unchanged.
    The ROI bounding box is merely divided into a fixed grid; no segmentation,
    thresholding, morphology, or ROI repair is performed.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = mask.astype(bool)
    bbox = _roi_bbox(roi)
    if bbox is None:
        return {}

    x0, y0, x1, y1 = bbox
    gray_crop = gray[y0:y1, x0:x1]
    mask_crop = roi[y0:y1, x0:x1]
    if gray_crop.size == 0:
        return {}

    # Compute LBP response maps only once, then summarize within each valid patch.
    lbp_maps = {
        (8, 1): local_binary_pattern(gray_crop, P=8, R=1, method="uniform"),
        (16, 2): local_binary_pattern(gray_crop, P=16, R=2, method="uniform"),
    }

    glcm_values: dict[str, list[float]] = defaultdict(list)
    lbp_values: dict[str, list[float]] = defaultdict(list)
    valid_patch_ratios: list[float] = []

    ys = _patch_slices(gray_crop.shape[0], grid_rows)
    xs = _patch_slices(gray_crop.shape[1], grid_cols)
    for py0, py1 in ys:
        for px0, px1 in xs:
            patch_mask = mask_crop[py0:py1, px0:px1]
            roi_pixels = int(patch_mask.sum())
            patch_area = max(1, patch_mask.size)
            if roi_pixels < LOCAL_MIN_ROI_PIXELS:
                continue
            valid_patch_ratios.append(roi_pixels / patch_area)
            patch_gray = gray_crop[py0:py1, px0:px1]

            # Local GLCM at the fine scale. Four directions are summarized within
            # each patch, then patch-to-patch heterogeneity is summarized below.
            per_prop: dict[str, list[float]] = defaultdict(list)
            for angle in LOCAL_GLCM_ANGLES:
                matrix = masked_glcm(patch_gray, patch_mask, distance=1, angle_deg=float(angle))
                props = glcm_properties(matrix)
                for prop in LOCAL_GLCM_PROPERTIES:
                    per_prop[prop].append(float(props[prop]))
            for prop in LOCAL_GLCM_PROPERTIES:
                vals = np.asarray(per_prop[prop], dtype=float)
                glcm_values[f"lg_local_glcm_{prop}_patch_mean"].append(float(np.mean(vals)))
                glcm_values[f"lg_local_glcm_{prop}_patch_dir_std"].append(float(np.std(vals)))

            for (p, r), lbp_map in lbp_maps.items():
                patch_lbp = lbp_map[py0:py1, px0:px1]
                stats = _lbp_hist_stats(patch_lbp[patch_mask], p=p)
                tag = f"lg_local_lbp_p{p}r{r}"
                for stat, value in stats.items():
                    lbp_values[f"{tag}_{stat}"].append(float(value))

    result: dict[str, float] = {
        "lg_local_valid_patch_count": float(len(valid_patch_ratios)),
        "lg_local_valid_patch_fraction": float(len(valid_patch_ratios) / max(1, grid_rows * grid_cols)),
    }
    result.update(_aggregate("lg_local_roi_patch_coverage", valid_patch_ratios))
    for name, values in glcm_values.items():
        result.update(_aggregate(name, values))
    for name, values in lbp_values.items():
        result.update(_aggregate(name, values))
    return result


def _gabor_kernel(wavelength: float, theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    # Scale sigma with wavelength so each bank member captures a meaningful
    # texture band while remaining compact enough for fruit ROIs.
    sigma = 0.56 * wavelength
    ksize = int(max(9, round(wavelength * 4)))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getGaborKernel(
        (ksize, ksize),
        sigma=sigma,
        theta=theta,
        lambd=wavelength,
        gamma=0.5,
        psi=0,
        ktype=cv2.CV_32F,
    )
    norm = float(np.sum(np.abs(kernel)))
    if norm > 1e-12:
        kernel = kernel / norm
    return kernel


def extract_gabor_texture_features(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Gabor-bank texture responses inside the existing ROI only.

    Gabor filtering is used here as a texture descriptor, not as preprocessing:
    it does not replace or modify the processed image passed to other members.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    roi = mask.astype(bool)
    bbox = _roi_bbox(roi)
    if bbox is None:
        return {}
    x0, y0, x1, y1 = bbox
    gray_crop = gray[y0:y1, x0:x1]
    mask_crop = roi[y0:y1, x0:x1]
    if gray_crop.size == 0 or int(mask_crop.sum()) < LOCAL_MIN_ROI_PIXELS:
        return {}

    result: dict[str, float] = {}
    orientation_energy: dict[float, list[float]] = defaultdict(list)
    wavelength_energy: dict[float, list[float]] = defaultdict(list)

    for wavelength in GABOR_WAVELENGTHS:
        for theta in GABOR_ORIENTATIONS_DEG:
            kernel = _gabor_kernel(float(wavelength), float(theta))
            response = cv2.filter2D(gray_crop, cv2.CV_32F, kernel)
            vals = response[mask_crop]
            abs_vals = np.abs(vals)
            energy = float(np.mean(vals**2)) if vals.size else 0.0
            mean_abs = float(np.mean(abs_vals)) if vals.size else 0.0
            std = float(np.std(vals)) if vals.size else 0.0
            p90_abs = float(np.percentile(abs_vals, 90)) if vals.size else 0.0
            tag = f"lg_gabor_l{str(wavelength).replace('.', '_')}_a{int(theta)}"
            result[f"{tag}_mean_abs"] = mean_abs
            result[f"{tag}_std"] = std
            result[f"{tag}_energy"] = energy
            result[f"{tag}_p90_abs"] = p90_abs
            orientation_energy[float(theta)].append(energy)
            wavelength_energy[float(wavelength)].append(energy)

    # Compact anisotropy / scale summaries can discriminate smooth, porous,
    # fibrous and netted surfaces without segmenting any defect region.
    orient_means = [float(np.mean(v)) for _, v in sorted(orientation_energy.items())]
    wave_means = [float(np.mean(v)) for _, v in sorted(wavelength_energy.items())]
    result.update(_aggregate("lg_gabor_orientation_energy", orient_means))
    result.update(_aggregate("lg_gabor_scale_energy", wave_means))
    if orient_means:
        result["lg_gabor_anisotropy"] = float(
            (max(orient_means) - min(orient_means)) / (np.mean(orient_means) + 1e-12)
        )
    else:
        result["lg_gabor_anisotropy"] = 0.0
    return result


def extract_local_global_texture_features(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Final v8 supplemental texture evidence: local heterogeneity + Gabor responses."""
    result = extract_local_texture_features(image_bgr, mask)
    result.update(extract_gabor_texture_features(image_bgr, mask))
    return result
