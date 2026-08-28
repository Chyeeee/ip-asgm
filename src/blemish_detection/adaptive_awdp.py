"""
Adaptive Abnormality-Weighted Damage Percentage (A-AWDP)
=========================================================

Member 4 Enhancement

Purpose:
    Improve AWDP by preserving strong blemish evidence while continuing
    to suppress weak Otsu false positives.

Baseline:
    Raw Otsu + Morphology 7x7

Previous enhancement:
    AWDP_A
        Intensity = 0.50
        Colour    = 0.30
        Texture   = 0.20

Evaluation:
    - Overall MAE
    - RMSE
    - Low-damage MAE
    - Moderate-damage MAE
    - High-damage MAE
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from abnormality_weighted_damage import (
    discover_samples,
    load_binary_mask,
    otsu_morphology_segmentation,
    intensity_abnormality,
    colour_abnormality,
    texture_abnormality,
    calculate_raw_damage,
    calculate_gt_damage,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
    / "adaptive_awdp"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# AWDP BASE WEIGHTS
# ============================================================

INTENSITY_WEIGHT = 0.50
COLOUR_WEIGHT = 0.30
TEXTURE_WEIGHT = 0.20


# ============================================================
# DAMAGE GROUP
# ============================================================

def get_damage_group(gt_damage):

    if gt_damage < 10:
        return "Low"

    if gt_damage <= 30:
        return "Moderate"

    return "High"


# ============================================================
# BASE CONFIDENCE
# ============================================================

def calculate_base_confidence(
    intensity,
    colour,
    texture,
):

    confidence = (
        INTENSITY_WEIGHT * intensity
        + COLOUR_WEIGHT * colour
        + TEXTURE_WEIGHT * texture
    )

    return np.clip(
        confidence,
        0.0,
        1.0,
    )


# ============================================================
# DAMAGE FROM CONFIDENCE
# ============================================================

def confidence_damage(
    blemish_mask,
    roi_mask,
    confidence,
):

    fruit_area = np.count_nonzero(
        roi_mask
    )

    if fruit_area == 0:
        return 0.0

    weighted = (
        blemish_mask.astype(np.float32)
        * confidence
    )

    weighted[
        roi_mask == 0
    ] = 0

    return float(
        np.sum(weighted)
        / fruit_area
        * 100.0
    )


# ============================================================
# STANDARD AWDP
# ============================================================

def standard_awdp(
    blemish_mask,
    roi_mask,
    confidence,
):

    return confidence_damage(
        blemish_mask,
        roi_mask,
        confidence,
    )


# ============================================================
# ADAPTIVE METHOD B
# CONFIDENCE POWER 0.75
# ============================================================

def adaptive_b(
    blemish_mask,
    roi_mask,
    confidence,
):

    adjusted = np.power(
        confidence,
        0.75,
    )

    return confidence_damage(
        blemish_mask,
        roi_mask,
        adjusted,
    )


# ============================================================
# ADAPTIVE METHOD C
# CONFIDENCE POWER 0.50
# ============================================================

def adaptive_c(
    blemish_mask,
    roi_mask,
    confidence,
):

    adjusted = np.sqrt(
        confidence
    )

    return confidence_damage(
        blemish_mask,
        roi_mask,
        adjusted,
    )


# ============================================================
# ADAPTIVE METHOD D
# STRONG EVIDENCE BOOST
# ============================================================

def adaptive_d(
    blemish_mask,
    roi_mask,
    confidence,
):

    adjusted = confidence.copy()

    # Strong abnormality receives additional weight.
    strong = confidence >= 0.60

    adjusted[strong] = (
        confidence[strong]
        + 0.35
        * (
            1.0
            - confidence[strong]
        )
    )

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    return confidence_damage(
        blemish_mask,
        roi_mask,
        adjusted,
    )


# ============================================================
# ADAPTIVE METHOD E
# STRONG-ABNORMALITY RATIO
# ============================================================

def adaptive_e(
    blemish_mask,
    roi_mask,
    confidence,
    raw_damage,
):

    candidate_pixels = (
        (blemish_mask > 0)
        & (roi_mask > 0)
    )

    candidate_count = np.count_nonzero(
        candidate_pixels
    )

    if candidate_count == 0:
        return 0.0

    strong_pixels = (
        candidate_pixels
        & (confidence >= 0.60)
    )

    strong_ratio = (
        np.count_nonzero(strong_pixels)
        / candidate_count
    )

    awdp = confidence_damage(
        blemish_mask,
        roi_mask,
        confidence,
    )

    # --------------------------------------------------------
    # Adaptive blend:
    #
    # little strong evidence
    # -> rely mainly on AWDP
    #
    # strong evidence
    # -> preserve more raw segmented area
    # --------------------------------------------------------

    raw_weight = np.clip(
        strong_ratio,
        0.0,
        1.0,
    )

    corrected_damage = (
        (1.0 - raw_weight) * awdp
        + raw_weight * raw_damage
    )

    return float(
        corrected_damage
    )


# ============================================================
# ERROR FUNCTIONS
# ============================================================

def mae(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def rmse(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    actual - predicted
                ) ** 2
            )
        )
    )


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_sample(sample):

    image = cv2.imread(
        str(sample["image"])
    )

    if image is None:
        raise ValueError(
            f"Cannot read image: {sample['image']}"
        )

    h, w = image.shape[:2]

    roi = load_binary_mask(
        sample["roi"],
        (h, w),
    )

    gt = load_binary_mask(
        sample["gt"],
        (h, w),
    )

    blemish = (
        otsu_morphology_segmentation(
            image,
            roi,
        )
    )

    raw_damage = calculate_raw_damage(
        blemish,
        roi,
    )

    gt_damage = calculate_gt_damage(
        gt,
        roi,
    )

    intensity = intensity_abnormality(
        image,
        roi,
    )

    colour = colour_abnormality(
        image,
        roi,
    )

    texture = texture_abnormality(
        image,
        roi,
    )

    confidence = calculate_base_confidence(
        intensity,
        colour,
        texture,
    )

    awdp = standard_awdp(
        blemish,
        roi,
        confidence,
    )

    aa_b = adaptive_b(
        blemish,
        roi,
        confidence,
    )

    aa_c = adaptive_c(
        blemish,
        roi,
        confidence,
    )

    aa_d = adaptive_d(
        blemish,
        roi,
        confidence,
    )

    aa_e = adaptive_e(
        blemish,
        roi,
        confidence,
        raw_damage,
    )

    return {
        "fruit": sample["fruit"],
        "category": sample["category"],
        "image": sample["stem"],

        "gt_damage": gt_damage,
        "damage_group": get_damage_group(
            gt_damage
        ),

        "Raw Otsu": raw_damage,
        "AWDP_A": awdp,
        "A-AWDP_B": aa_b,
        "A-AWDP_C": aa_c,
        "A-AWDP_D": aa_d,
        "A-AWDP_E": aa_e,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(
    df,
    method,
):

    overall_mae = mae(
        df["gt_damage"],
        df[method],
    )

    overall_rmse = rmse(
        df["gt_damage"],
        df[method],
    )

    result = {
        "Method": method,
        "Overall_MAE": overall_mae,
        "RMSE": overall_rmse,
    }

    for group in [
        "Low",
        "Moderate",
        "High",
    ]:

        subset = df[
            df["damage_group"] == group
        ]

        if len(subset) == 0:

            result[
                f"{group}_MAE"
            ] = np.nan

        else:

            result[
                f"{group}_MAE"
            ] = mae(
                subset["gt_damage"],
                subset[method],
            )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "MEMBER 4 - ADAPTIVE AWDP EXPERIMENT"
    )
    print("=" * 72)

    samples = discover_samples()

    if not samples:

        raise RuntimeError(
            "No valid samples found."
        )

    print()
    print(
        f"Samples: {len(samples)}"
    )

    rows = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(samples)}] "
            f"{sample['fruit']} | "
            f"{sample['category']} | "
            f"{sample['stem']}"
        )

        result = process_sample(
            sample
        )

        rows.append(
            result
        )

        print(
            f"  GT          : "
            f"{result['gt_damage']:.2f}%"
        )

        print(
            f"  Raw Otsu    : "
            f"{result['Raw Otsu']:.2f}%"
        )

        print(
            f"  AWDP_A      : "
            f"{result['AWDP_A']:.2f}%"
        )

        print(
            f"  A-AWDP_B    : "
            f"{result['A-AWDP_B']:.2f}%"
        )

        print(
            f"  A-AWDP_C    : "
            f"{result['A-AWDP_C']:.2f}%"
        )

        print(
            f"  A-AWDP_D    : "
            f"{result['A-AWDP_D']:.2f}%"
        )

        print(
            f"  A-AWDP_E    : "
            f"{result['A-AWDP_E']:.2f}%"
        )

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # DAMAGE DISTRIBUTION
    # ========================================================

    print()
    print("=" * 72)
    print("GROUND-TRUTH DAMAGE GROUPS")
    print("=" * 72)

    print(
        df[
            "damage_group"
        ].value_counts()
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    methods = [
        "Raw Otsu",
        "AWDP_A",
        "A-AWDP_B",
        "A-AWDP_C",
        "A-AWDP_D",
        "A-AWDP_E",
    ]

    summary_rows = []

    for method in methods:

        summary_rows.append(
            evaluate_method(
                df,
                method,
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        "Overall_MAE"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "ADAPTIVE AWDP COMPARISON"
    )
    print("=" * 72)

    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # BEST
    # ========================================================

    best = summary.iloc[0]

    raw_row = summary[
        summary["Method"]
        == "Raw Otsu"
    ].iloc[0]

    awdp_row = summary[
        summary["Method"]
        == "AWDP_A"
    ].iloc[0]

    improvement_raw = (
        raw_row["Overall_MAE"]
        - best["Overall_MAE"]
    )

    improvement_awdp = (
        awdp_row["Overall_MAE"]
        - best["Overall_MAE"]
    )

    relative_raw = (
        improvement_raw
        / raw_row["Overall_MAE"]
        * 100
    )

    print()
    print("=" * 72)
    print("BEST METHOD")
    print("=" * 72)

    print(
        f"Method             : "
        f"{best['Method']}"
    )

    print(
        f"Overall MAE        : "
        f"{best['Overall_MAE']:.4f}"
    )

    print(
        f"RMSE               : "
        f"{best['RMSE']:.4f}"
    )

    print(
        f"Low-damage MAE     : "
        f"{best['Low_MAE']:.4f}"
    )

    print(
        f"Moderate-damage MAE: "
        f"{best['Moderate_MAE']:.4f}"
    )

    print(
        f"High-damage MAE    : "
        f"{best['High_MAE']:.4f}"
    )

    print()
    print(
        f"Improvement vs Raw : "
        f"{improvement_raw:+.4f} pp"
    )

    print(
        f"Relative reduction : "
        f"{relative_raw:+.2f}%"
    )

    print(
        f"Improvement vs AWDP: "
        f"{improvement_awdp:+.4f} pp"
    )

    if (
        best["Overall_MAE"]
        < awdp_row["Overall_MAE"]
    ):

        print()
        print(
            "RESULT: Adaptive AWDP improved "
            "the original AWDP."
        )

    else:

        print()
        print(
            "RESULT: Original AWDP_A remains "
            "the best method."
        )

    print("=" * 72)

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_DIR
        / "adaptive_awdp_detailed.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "adaptive_awdp_summary.csv",
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_DIR
        / "adaptive_awdp_detailed.csv"
    )

    print(
        OUTPUT_DIR
        / "adaptive_awdp_summary.csv"
    )


if __name__ == "__main__":
    main()