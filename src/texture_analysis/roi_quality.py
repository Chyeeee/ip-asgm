from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .config import (
    ROI_ABSOLUTE_MIN_RATIO,
    ROI_MIN_BBOX_DIM_RATIO,
    ROI_MIN_LARGEST_COMPONENT_FRAC,
    ROI_RELATIVE_MEDIAN_FACTOR,
)


def measure_roi(mask: np.ndarray) -> dict[str, float]:
    """Measure an existing ROI without modifying it."""
    roi = mask.astype(bool)
    h, w = roi.shape[:2]
    pixels = int(roi.sum())
    area_ratio = pixels / max(1, h * w)

    ys, xs = np.where(roi)
    if pixels == 0:
        return {
            "roi_pixels": 0,
            "roi_area_ratio": 0.0,
            "roi_bbox_width_ratio": 0.0,
            "roi_bbox_height_ratio": 0.0,
            "roi_largest_component_fraction": 0.0,
        }

    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        roi.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        largest = 0
    else:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())

    return {
        "roi_pixels": pixels,
        "roi_area_ratio": float(area_ratio),
        "roi_bbox_width_ratio": float(bbox_w / max(1, w)),
        "roi_bbox_height_ratio": float(bbox_h / max(1, h)),
        "roi_largest_component_fraction": float(largest / max(1, pixels)),
    }


def finalize_roi_quality(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious ROIs for diagnostics and demo selection only.

    The thresholds are adaptive per fruit so a naturally smaller fruit ROI is
    not compared directly with a large-fruit ROI. No mask is altered.
    """
    if raw_df.empty:
        return raw_df.copy()

    out = raw_df.copy()
    medians = out.groupby("fruit")["roi_area_ratio"].median().to_dict()
    thresholds = []
    valid = []
    reasons = []

    for _, row in out.iterrows():
        fruit_median = float(medians.get(row["fruit"], 0.0))
        adaptive_min = max(
            ROI_ABSOLUTE_MIN_RATIO,
            fruit_median * ROI_RELATIVE_MEDIAN_FACTOR,
        )
        thresholds.append(adaptive_min)

        problems: list[str] = []
        if float(row["roi_area_ratio"]) < adaptive_min:
            problems.append("roi_too_small")
        if float(row["roi_bbox_width_ratio"]) < ROI_MIN_BBOX_DIM_RATIO:
            problems.append("bbox_too_narrow")
        if float(row["roi_bbox_height_ratio"]) < ROI_MIN_BBOX_DIM_RATIO:
            problems.append("bbox_too_short")
        if float(row["roi_largest_component_fraction"]) < ROI_MIN_LARGEST_COMPONENT_FRAC:
            problems.append("highly_fragmented")

        valid.append(len(problems) == 0)
        reasons.append("ok" if not problems else ";".join(problems))

    out["roi_adaptive_min_ratio"] = thresholds
    out["roi_valid_for_demo"] = valid
    out["roi_quality_reason"] = reasons
    return out
