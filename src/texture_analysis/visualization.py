from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lbp_features import lbp_response_map
from .io_utils import load_binary_mask, load_processed_image
from .evaluation import PROPOSED_METHOD


def _proposed_response(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Local-global texture visual aid for PTD+.

    GLCM is a statistical descriptor rather than a true pixelwise map, so the
    visualization combines coarse LBP structure with Gabor response energy.
    This is a visual aid only; the classifier uses the numerical descriptors.
    """
    roi = mask.astype(bool)
    lbp = lbp_response_map(image_bgr, mask, p=16, r=2)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gabor_sum = np.zeros_like(gray, dtype=np.float32)
    for angle_deg in (0, 45, 90, 135):
        theta = np.deg2rad(angle_deg)
        kernel = cv2.getGaborKernel((25, 25), 3.5, float(theta), 6.0, 0.5, 0, ktype=cv2.CV_32F)
        norm = float(np.sum(np.abs(kernel)))
        if norm > 1e-12:
            kernel = kernel / norm
        resp = np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel))
        gabor_sum += resp
    gabor_sum /= 4.0
    gabor_norm = np.zeros_like(gabor_sum, dtype=np.float32)
    if np.any(roi):
        vals = gabor_sum[roi]
        lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
        if hi > lo:
            gabor_norm[roi] = np.clip((gabor_sum[roi] - lo) / (hi - lo), 0.0, 1.0)

    response = 0.5 * lbp + 0.5 * gabor_norm
    response[~roi] = 0.0
    return response


def _overlay_response(image_bgr: np.ndarray, response: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    heat = np.uint8(np.clip(response, 0, 1) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = base.copy()
    roi = mask.astype(bool)
    blended = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    out[roi] = blended[roi]
    return out


def select_demo_samples(
    predictions_df: pd.DataFrame,
    best_baseline_method: str,
    roi_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Automatically choose a visually usable sample for every fruit.

    Selection priority:
      1. ROI must pass diagnostic quality validation when possible.
      2. Prefer baseline wrong -> proposed correct.
      3. Otherwise prefer both correct.
      4. Prefer the larger/more coherent ROI.

    This automatically replaces tiny/broken Grape examples without requiring
    the user to know the image filename. No ROI is repaired or modified.
    """
    baseline = predictions_df[predictions_df["method"] == best_baseline_method].copy()
    proposed = predictions_df[predictions_df["method"] == PROPOSED_METHOD].copy()
    keys = ["fruit", "category", "image", "relative_path", "processed_path", "mask_path"]
    merged = baseline.merge(proposed, on=keys, suffixes=("_baseline", "_proposed"))

    quality_cols = keys + [
        "roi_area_ratio",
        "roi_bbox_width_ratio",
        "roi_bbox_height_ratio",
        "roi_largest_component_fraction",
        "roi_valid_for_demo",
        "roi_quality_reason",
    ]
    available_quality = [c for c in quality_cols if c in roi_quality_df.columns]
    if available_quality:
        merged = merged.merge(
            roi_quality_df[available_quality].drop_duplicates(keys),
            on=keys,
            how="left",
        )

    if "roi_valid_for_demo" not in merged.columns:
        merged["roi_valid_for_demo"] = True
    if "roi_area_ratio" not in merged.columns:
        merged["roi_area_ratio"] = 0.0
    if "roi_largest_component_fraction" not in merged.columns:
        merged["roi_largest_component_fraction"] = 0.0

    rows = []
    for fruit, g in merged.groupby("fruit", sort=True):
        valid = g[g["roi_valid_for_demo"].fillna(False)].copy()
        candidates = valid if not valid.empty else g.copy()

        candidates["demo_priority"] = 0
        candidates.loc[
            (~candidates["correct_baseline"]) & (candidates["correct_proposed"]),
            "demo_priority",
        ] = 3
        candidates.loc[
            (candidates["correct_baseline"]) & (candidates["correct_proposed"]),
            "demo_priority",
        ] = 2
        candidates.loc[
            (~candidates["correct_baseline"]) & (~candidates["correct_proposed"]),
            "demo_priority",
        ] = 0
        candidates.loc[
            (candidates["correct_baseline"]) & (~candidates["correct_proposed"]),
            "demo_priority",
        ] = 1

        # Large, coherent ROI wins ties. This deliberately avoids the previous
        # tiny grape mask if a valid alternative exists.
        candidates = candidates.sort_values(
            [
                "demo_priority",
                "roi_area_ratio",
                "roi_largest_component_fraction",
                "image",
            ],
            ascending=[False, False, False, True],
        )
        rows.append(candidates.iloc[0])

    return pd.DataFrame(rows)


def save_before_after_figures(
    predictions_df: pd.DataFrame,
    out_dir: Path,
    best_baseline_method: str,
    roi_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    demos = select_demo_samples(predictions_df, best_baseline_method, roi_quality_df)

    master_items = []
    for _, row in demos.iterrows():
        image = load_processed_image(row["processed_path"])
        mask = load_binary_mask(row["mask_path"], image.shape[:2])
        response = _proposed_response(image, mask)
        overlay = _overlay_response(image, response, mask)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        roi_rgb = rgb.copy()
        roi_rgb[~mask.astype(bool)] = 0

        baseline_pred = str(row["prediction_baseline"])
        proposed_pred = str(row["prediction_proposed"])
        truth = str(row["category"])
        fruit = str(row["fruit"])

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
        axes[0].imshow(rgb)
        axes[0].set_title("Before\nMedianFinal processed image")
        axes[1].imshow(roi_rgb)
        axes[1].set_title("Existing ROI\nUsed unchanged")
        axes[2].imshow(overlay)
        axes[2].set_title("After / Visual Aid\nPTD+ local/global texture response")
        for ax in axes:
            ax.axis("off")

        symbol_b = "✓" if baseline_pred == truth else "✗"
        symbol_p = "✓" if proposed_pred == truth else "✗"
        fig.suptitle(
            f"{fruit} | Ground Truth: {truth} | Best Baseline ({best_baseline_method}): "
            f"{baseline_pred} {symbol_b} | Proposed: {proposed_pred} {symbol_p}",
            fontsize=10.5,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{fruit}_before_after.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        master_items.append((fruit, rgb, overlay, truth, baseline_pred, proposed_pred))

    if master_items:
        n = len(master_items)
        fig, axes = plt.subplots(n, 2, figsize=(9, max(3, n * 2.5)))
        if n == 1:
            axes = np.array([axes])
        for r, (fruit, before, after, truth, baseline_pred, proposed_pred) in enumerate(master_items):
            axes[r, 0].imshow(before)
            axes[r, 0].set_title(f"{fruit} - Before")
            axes[r, 1].imshow(after)
            axes[r, 1].set_title(
                f"PTD+ visual | True: {truth} | B: {baseline_pred} | P: {proposed_pred}"
            )
            axes[r, 0].axis("off")
            axes[r, 1].axis("off")
        fig.suptitle("Nine-Fruit Before vs PTD+ Texture Visualization", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "before_after_all_fruits.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    return demos
