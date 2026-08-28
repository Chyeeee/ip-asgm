import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# IMPORTANT: adjust these to match your existing evaluation.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = PROJECT_ROOT / "BalancedDataset"

GROUND_TRUTH_DIR = (
    PROJECT_ROOT
    / "results/blemish_detection/ground_truth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/blemish_detection/fabr_enhancement"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred_mask, gt_mask):

    pred = pred_mask > 0
    gt = gt_mask > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    dice = (
        (2 * tp) / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall
    }


# ============================================================
# ORIGINAL OTSU
# ============================================================

def otsu_segmentation(image, roi_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    fruit_pixels = gray[roi_mask > 0]

    if len(fruit_pixels) == 0:
        return np.zeros_like(gray)

    threshold, _ = cv2.threshold(
        fruit_pixels,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = np.zeros_like(gray)

    mask[
        (gray < threshold)
        & (roi_mask > 0)
    ] = 255

    return mask


# ============================================================
# MORPHOLOGY
# ============================================================

def morphology_7x7(mask, roi_mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    refined = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        kernel
    )

    refined[roi_mask == 0] = 0

    return refined


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_inside_roi(values, roi_mask):

    result = np.zeros(
        values.shape,
        dtype=np.float32
    )

    pixels = values[roi_mask > 0].astype(
        np.float32
    )

    if pixels.size == 0:
        return result

    low = np.percentile(
        pixels,
        5
    )

    high = np.percentile(
        pixels,
        95
    )

    if high <= low:
        return result

    result = (
        values.astype(np.float32) - low
    ) / (high - low)

    result = np.clip(
        result,
        0.0,
        1.0
    )

    result[roi_mask == 0] = 0

    return result


# ============================================================
# CUE 1: ADAPTIVE DARKNESS
# ============================================================

def adaptive_darkness_score(image, roi_mask):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness = lab[:, :, 0].astype(
        np.float32
    )

    pixels = lightness[
        roi_mask > 0
    ]

    if pixels.size == 0:
        return np.zeros_like(
            lightness,
            dtype=np.float32
        )

    # Use relatively bright fruit pixels as a reference
    # for healthy surface appearance.
    healthy_reference = np.percentile(
        pixels,
        70
    )

    darkness = (
        healthy_reference - lightness
    ) / max(
        healthy_reference,
        1.0
    )

    darkness = np.clip(
        darkness,
        0.0,
        1.0
    )

    darkness[roi_mask == 0] = 0

    return darkness


# ============================================================
# CUE 2: COLOUR ABNORMALITY
# ============================================================

def colour_abnormality_score(image, roi_mask):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    roi_pixels = lab[
        roi_mask > 0
    ]

    if roi_pixels.size == 0:

        return np.zeros(
            image.shape[:2],
            dtype=np.float32
        )

    # Median Lab colour is more robust than mean
    # when blemishes already exist.
    reference_colour = np.median(
        roi_pixels,
        axis=0
    )

    difference = (
        lab - reference_colour
    )

    distance = np.sqrt(
        np.sum(
            difference ** 2,
            axis=2
        )
    )

    distance = normalize_inside_roi(
        distance,
        roi_mask
    )

    return distance


# ============================================================
# CUE 3: LOCAL DEVIATION
# ============================================================

def local_deviation_score(
    image,
    roi_mask,
    window_size=31
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness = lab[:, :, 0].astype(
        np.float32
    )

    local_mean = cv2.GaussianBlur(
        lightness,
        (
            window_size,
            window_size
        ),
        0
    )

    # Positive only when pixel is darker
    # than its local neighbourhood.
    deviation = (
        local_mean - lightness
    )

    deviation[
        deviation < 0
    ] = 0

    deviation = normalize_inside_roi(
        deviation,
        roi_mask
    )

    return deviation


# ============================================================
# CUE 4: OTSU PRIOR
# ============================================================

def otsu_prior_score(image, roi_mask):

    otsu = otsu_segmentation(
        image,
        roi_mask
    )

    return (
        otsu.astype(np.float32)
        / 255.0
    )


# ============================================================
# REMOVE TINY COMPONENTS
# ============================================================

def remove_small_components(
    mask,
    roi_mask,
    minimum_ratio=0.001
):

    fruit_area = np.count_nonzero(
        roi_mask
    )

    if fruit_area == 0:
        return mask

    minimum_area = max(
        1,
        int(
            fruit_area
            * minimum_ratio
        )
    )

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    output = np.zeros_like(
        mask
    )

    for label in range(
        1,
        number_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area:

            output[
                labels == label
            ] = 255

    output[
        roi_mask == 0
    ] = 0

    return output


# ============================================================
# FABR
#
# B(x) =
# wd * Darkness
# + wc * Colour Abnormality
# + wl * Local Deviation
# + wo * Otsu Prior
# ============================================================

def fabr_segmentation(
    image,
    roi_mask,
    weights,
    threshold,
    window_size=31,
    component_ratio=0.001
):

    darkness = adaptive_darkness_score(
        image,
        roi_mask
    )

    colour = colour_abnormality_score(
        image,
        roi_mask
    )

    local = local_deviation_score(
        image,
        roi_mask,
        window_size
    )

    otsu = otsu_prior_score(
        image,
        roi_mask
    )

    wd = weights["darkness"]
    wc = weights["colour"]
    wl = weights["local"]
    wo = weights["otsu"]

    confidence = (
        wd * darkness
        + wc * colour
        + wl * local
        + wo * otsu
    )

    mask = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )

    mask[
        (confidence >= threshold)
        & (roi_mask > 0)
    ] = 255

    mask = morphology_7x7(
        mask,
        roi_mask
    )

    mask = remove_small_components(
        mask,
        roi_mask,
        minimum_ratio=component_ratio
    )

    return mask, confidence


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [

    {
        "name": "FABR_A",
        "weights": {
            "darkness": 0.35,
            "colour": 0.30,
            "local": 0.20,
            "otsu": 0.15
        },
        "threshold": 0.45
    },

    {
        "name": "FABR_B",
        "weights": {
            "darkness": 0.30,
            "colour": 0.35,
            "local": 0.20,
            "otsu": 0.15
        },
        "threshold": 0.45
    },

    {
        "name": "FABR_C",
        "weights": {
            "darkness": 0.30,
            "colour": 0.30,
            "local": 0.25,
            "otsu": 0.15
        },
        "threshold": 0.40
    },

    {
        "name": "FABR_D",
        "weights": {
            "darkness": 0.25,
            "colour": 0.35,
            "local": 0.25,
            "otsu": 0.15
        },
        "threshold": 0.40
    },

    {
        "name": "FABR_E",
        "weights": {
            "darkness": 0.25,
            "colour": 0.40,
            "local": 0.20,
            "otsu": 0.15
        },
        "threshold": 0.45
    }
]

# ============================================================
# FABR DATASET EVALUATION
# ============================================================

import re


PROCESSED_ROOT = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "MedianFinal"
    / "ProcessedImages"
)

ROI_ROOT = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "MedianFinal"
    / "ROIMasks"
)

GT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
    / "ground_truth"
)

FABR_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
    / "fabr_enhancement"
)

FABR_OUTPUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Find matching file
# ------------------------------------------------------------

def find_matching_file(folder, base_name):
    """
    Find a file whose filename contains the required base name.

    This is intentionally flexible because ROI masks and processed
    images may contain suffixes added by preprocessing.
    """

    if not folder.exists():
        return None

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # First try exact stem
    for file in folder.iterdir():
        if (
            file.is_file()
            and file.suffix.lower() in extensions
            and file.stem == base_name
        ):
            return file

    # Then try filename beginning with base name
    for file in folder.iterdir():
        if (
            file.is_file()
            and file.suffix.lower() in extensions
            and file.stem.startswith(base_name)
        ):
            return file

    # Last fallback: contains base name
    for file in folder.iterdir():
        if (
            file.is_file()
            and file.suffix.lower() in extensions
            and base_name in file.stem
        ):
            return file

    return None


# ------------------------------------------------------------
# Parse GT filename
# ------------------------------------------------------------

def parse_gt_filename(gt_path):
    """
    Example:
        Banana_Ripe_084_gt.png

    returns:
        fruit    = Banana
        category = Ripe
        base     = Banana_Ripe_084
    """

    stem = gt_path.stem

    if stem.endswith("_gt"):
        stem = stem[:-3]

    parts = stem.split("_")

    fruit = parts[0]

    # Special Guava categories
    if fruit == "Guava":
        if "Class_A" in stem:
            category = "Class_A"
        elif "Class_B" in stem:
            category = "Class_B"
        elif "Defect" in stem:
            category = "Defect"
        else:
            raise ValueError(
                f"Cannot determine Guava category: {gt_path.name}"
            )
    else:
        if len(parts) < 3:
            raise ValueError(
                f"Unexpected GT filename: {gt_path.name}"
            )

        category = parts[1]

    return fruit, category, stem


# ------------------------------------------------------------
# Load GT mask
# ------------------------------------------------------------

def load_binary_mask(path, target_shape=None):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(
            f"Cannot read mask: {path}"
        )

    if (
        target_shape is not None
        and mask.shape[:2] != target_shape[:2]
    ):
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return mask


# ------------------------------------------------------------
# Segmentation metrics
# ------------------------------------------------------------

def calculate_metrics(predicted, ground_truth, roi_mask):
    """
    Calculate IoU, Dice, Precision and Recall only inside
    the fruit ROI.
    """

    pred = predicted > 0
    gt = ground_truth > 0
    roi = roi_mask > 0

    pred = pred & roi
    gt = gt & roi

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt & roi).sum()
    fn = np.logical_and(~pred & roi, gt).sum()

    union = np.logical_or(pred, gt).sum()

    if union == 0:
        iou = 0.0
    else:
        iou = tp / union

    denominator = (2 * tp) + fp + fn

    if denominator == 0:
        dice = 0.0
    else:
        dice = (2 * tp) / denominator

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
    }


# ------------------------------------------------------------
# Baseline Otsu
# ------------------------------------------------------------

def original_otsu(image, roi_mask):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    roi_pixels = gray[roi_mask > 0]

    if roi_pixels.size == 0:
        return np.zeros_like(gray)

    threshold_value, _ = cv2.threshold(
        roi_pixels,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Blemishes assumed darker than healthy fruit surface
    mask = np.zeros_like(gray)

    mask[
        (gray <= threshold_value)
        & (roi_mask > 0)
    ] = 255

    return mask


# ------------------------------------------------------------
# Otsu + Morphology 7x7
# ------------------------------------------------------------

def otsu_morphology_7x7(image, roi_mask):
    mask = original_otsu(
        image,
        roi_mask,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = cv2.bitwise_and(
        mask,
        roi_mask,
    )

    return mask


# ------------------------------------------------------------
# Run FABR configuration safely
# ------------------------------------------------------------

def run_fabr_configuration(
    image,
    roi_mask,
    config,
):
    """
    Calls the FABR function already defined earlier in this file.

    Supports slightly different function signatures so the
    experiment is easier to integrate with the existing code.
    """

    # Change this name ONLY if your existing FABR function
    # has a different name.
    if "fabr_segmentation" in globals():

        function = globals()["fabr_segmentation"]

    elif "fabr" in globals():

        function = globals()["fabr"]

    elif "apply_fabr" in globals():

        function = globals()["apply_fabr"]

    else:
        raise NameError(
            "\nFABR function not found.\n"
            "Expected one of:\n"
            "  fabr_segmentation()\n"
            "  fabr()\n"
            "  apply_fabr()\n"
        )

    try:
        result = function(
            image,
            roi_mask,
            **config,
        )

    except TypeError:
        result = function(
            image=image,
            roi_mask=roi_mask,
            config=config,
        )

    # Some implementations may return:
    # mask
    # OR
    # (mask, confidence_map)
    # OR
    # dictionary

    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, dict):

        for key in [
            "mask",
            "blemish_mask",
            "final_mask",
            "segmentation",
        ]:
            if key in result:
                result = result[key]
                break

    result = np.asarray(result)

    if result.ndim == 3:
        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2GRAY,
        )

    if result.shape != roi_mask.shape:
        result = cv2.resize(
            result,
            (
                roi_mask.shape[1],
                roi_mask.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    if result.dtype != np.uint8:
        if result.max() <= 1:
            result = result * 255

        result = np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

    _, result = cv2.threshold(
        result,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    result = cv2.bitwise_and(
        result,
        roi_mask,
    )

    return result


# ------------------------------------------------------------
# Discover 27 samples
# ------------------------------------------------------------

def discover_samples():

    gt_files = sorted(
        GT_ROOT.rglob("*_gt.png")
    )

    samples = []

    print()
    print("=" * 70)
    print("FABR SAMPLE DISCOVERY")
    print("=" * 70)

    print(
        f"Ground-truth masks found: {len(gt_files)}"
    )

    for gt_path in gt_files:

        fruit, category, base_name = (
            parse_gt_filename(gt_path)
        )

        processed_folder = (
            PROCESSED_ROOT
            / fruit
            / category
        )

        roi_folder = (
            ROI_ROOT
            / fruit
            / category
        )

        image_path = find_matching_file(
            processed_folder,
            base_name,
        )

        roi_path = find_matching_file(
            roi_folder,
            base_name,
        )

        if image_path is None:
            print(
                f"[MISSING IMAGE] {base_name}"
            )
            continue

        if roi_path is None:
            print(
                f"[MISSING ROI]   {base_name}"
            )
            continue

        samples.append(
            {
                "fruit": fruit,
                "category": category,
                "base_name": base_name,
                "image_path": image_path,
                "roi_path": roi_path,
                "gt_path": gt_path,
            }
        )

    print()
    print(
        f"Valid matched samples: {len(samples)}"
    )

    print("=" * 70)

    return samples


# ------------------------------------------------------------
# Evaluate one method
# ------------------------------------------------------------

def add_result(
    rows,
    sample,
    method_name,
    mask,
    gt_mask,
    roi_mask,
):

    metrics = calculate_metrics(
        mask,
        gt_mask,
        roi_mask,
    )

    roi_area = np.count_nonzero(
        roi_mask
    )

    damage_area = np.count_nonzero(
        mask
    )

    damage_percentage = (
        (damage_area / roi_area) * 100
        if roi_area > 0
        else 0.0
    )

    row = {
        "fruit": sample["fruit"],
        "category": sample["category"],
        "image": sample["base_name"],
        "method": method_name,
        "IoU": metrics["IoU"],
        "Dice": metrics["Dice"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "Damage_Percentage": damage_percentage,
    }

    rows.append(row)

    return metrics


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def evaluate_fabr():

    samples = discover_samples()

    if len(samples) == 0:
        print(
            "\nERROR: No valid samples were found."
        )
        return

    if len(samples) != 27:
        print()
        print(
            "WARNING:"
            f" Expected 27 samples but found {len(samples)}."
        )
        print(
            "The experiment will continue using the "
            "matched samples."
        )

    rows = []

    print()
    print("=" * 70)
    print("FRUIT-ADAPTIVE BLEMISH REFINEMENT (FABR)")
    print("=" * 70)

    print(
        f"Ground-truth images: {len(samples)}"
    )

    print()

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        print(
            f"[{index}/{len(samples)}] "
            f"{sample['fruit']} | "
            f"{sample['category']} | "
            f"{sample['base_name']}.jpg"
        )

        image = cv2.imread(
            str(sample["image_path"])
        )

        if image is None:
            print("  ERROR: Cannot read image.")
            continue

        roi_mask = load_binary_mask(
            sample["roi_path"],
            image.shape,
        )

        gt_mask = load_binary_mask(
            sample["gt_path"],
            image.shape,
        )

        # Ensure GT is restricted to fruit ROI
        gt_mask = cv2.bitwise_and(
            gt_mask,
            roi_mask,
        )

        # ------------------------------------------------
        # Original Otsu
        # ------------------------------------------------

        otsu_mask = original_otsu(
            image,
            roi_mask,
        )

        metrics = add_result(
            rows,
            sample,
            "Original Otsu",
            otsu_mask,
            gt_mask,
            roi_mask,
        )

        print(
            "  Original Otsu        "
            f"IoU={metrics['IoU']:.4f} | "
            f"Dice={metrics['Dice']:.4f} | "
            f"P={metrics['Precision']:.4f} | "
            f"R={metrics['Recall']:.4f}"
        )

        # ------------------------------------------------
        # Otsu + Morphology 7x7
        # ------------------------------------------------

        morph_mask = otsu_morphology_7x7(
            image,
            roi_mask,
        )

        metrics = add_result(
            rows,
            sample,
            "Otsu + Morphology 7x7",
            morph_mask,
            gt_mask,
            roi_mask,
        )

        print(
            "  Otsu + Morph 7x7     "
            f"IoU={metrics['IoU']:.4f} | "
            f"Dice={metrics['Dice']:.4f} | "
            f"P={metrics['Precision']:.4f} | "
            f"R={metrics['Recall']:.4f}"
        )

        # ------------------------------------------------
        # FABR configurations
        # ------------------------------------------------

        for config in CONFIGURATIONS:

            config_name = config["name"]

            fabr_mask, confidence = fabr_segmentation(
                image=image,
                roi_mask=roi_mask,
                weights=config["weights"],
                threshold=config["threshold"],
                window_size=31,
                component_ratio=0.001,
            )

            metrics = add_result(
                rows,
                sample,
                config_name,
                fabr_mask,
                gt_mask,
                roi_mask,
            )

            print(
                f"  {config_name:<20}"
                f"IoU={metrics['IoU']:.4f} | "
                f"Dice={metrics['Dice']:.4f} | "
                f"P={metrics['Precision']:.4f} | "
                f"R={metrics['Recall']:.4f}"
            )

        print()

    # ----------------------------------------------------
    # DataFrame
    # ----------------------------------------------------

    df = pd.DataFrame(rows)

    if df.empty:
        print("No evaluation results generated.")
        return

    detailed_csv = (
        FABR_OUTPUT
        / "fabr_detailed_results.csv"
    )

    df.to_csv(
        detailed_csv,
        index=False,
    )

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    summary = (
        df.groupby("method")
        [
            [
                "IoU",
                "Dice",
                "Precision",
                "Recall",
            ]
        ]
        .mean()
        .reset_index()
    )

    summary["Overall_Score"] = (
        summary[
            [
                "IoU",
                "Dice",
                "Precision",
                "Recall",
            ]
        ].mean(axis=1)
    )

    summary = summary.sort_values(
        "Overall_Score",
        ascending=False,
    )

    summary_csv = (
        FABR_OUTPUT
        / "fabr_summary.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    # ----------------------------------------------------
    # Print summary
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print("FABR ENHANCEMENT SUMMARY")
    print("=" * 70)

    for _, row in summary.iterrows():

        print()
        print(row["method"])

        print(
            f"  Mean IoU       : "
            f"{row['IoU']:.4f}"
        )

        print(
            f"  Mean Dice      : "
            f"{row['Dice']:.4f}"
        )

        print(
            f"  Mean Precision : "
            f"{row['Precision']:.4f}"
        )

        print(
            f"  Mean Recall    : "
            f"{row['Recall']:.4f}"
        )

        print(
            f"  Overall Score  : "
            f"{row['Overall_Score']:.4f}"
        )

    # ----------------------------------------------------
    # Best result
    # ----------------------------------------------------

    best = summary.iloc[0]

    print()
    print("=" * 70)
    print("BEST FABR EXPERIMENT RESULT")
    print("=" * 70)

    print(
        f"Method         : {best['method']}"
    )

    print(
        f"Mean IoU       : {best['IoU']:.4f}"
    )

    print(
        f"Mean Dice      : {best['Dice']:.4f}"
    )

    print(
        f"Mean Precision : {best['Precision']:.4f}"
    )

    print(
        f"Mean Recall    : {best['Recall']:.4f}"
    )

    print(
        f"Overall Score  : "
        f"{best['Overall_Score']:.4f}"
    )

    print("=" * 70)

    print()
    print("Results saved to:")

    print(
        f"  {detailed_csv}"
    )

    print(
        f"  {summary_csv}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MEMBER 4 - FABR ENHANCEMENT EXPERIMENT")
    print("=" * 70)

    evaluate_fabr()


if __name__ == "__main__":
    main()