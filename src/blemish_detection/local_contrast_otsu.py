import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "results/blemish_detection/ground_truth_selection.csv"

PROCESSED_ROOT = (
    "results/preprocessing/MedianFinal/ProcessedImages"
)

ROI_ROOT = (
    "results/preprocessing/MedianFinal/ROIMasks"
)

GROUND_TRUTH_ROOT = (
    "results/blemish_detection/ground_truth"
)

OUTPUT_ROOT = (
    "results/blemish_detection/local_contrast_otsu"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MORPH_KERNEL_SIZE = 7

# Local neighbourhood sizes
WINDOW_SIZES = [
    11,
    21,
    31
]

# Darkness/Otsu weight, Local Contrast weight
WEIGHT_CONFIGURATIONS = [
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40)
]


# ============================================================
# 1. ORIGINAL OTSU
# ============================================================

def otsu_segmentation(image, roi_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    fruit_pixels = gray[
        roi_mask > 0
    ]

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
# 2. MORPHOLOGICAL REFINEMENT
# ============================================================

def apply_morphology(mask, roi_mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    result = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel
    )

    result[
        roi_mask == 0
    ] = 0

    return result


# ============================================================
# 3. DARKNESS SCORE
# ============================================================

def calculate_darkness_score(
    image,
    roi_mask
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    fruit_pixels = gray[
        roi_mask > 0
    ]

    if len(fruit_pixels) == 0:

        return np.zeros(
            gray.shape,
            dtype=np.float32
        )

    threshold, _ = cv2.threshold(
        fruit_pixels.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # Pixels darker than Otsu threshold obtain higher scores.
    # --------------------------------------------------------

    darkness = (
        threshold - gray
    ) / max(
        float(threshold),
        1.0
    )

    darkness = np.clip(
        darkness,
        0.0,
        1.0
    )

    darkness[
        roi_mask == 0
    ] = 0.0

    return darkness


# ============================================================
# 4. LOCAL CONTRAST SCORE
# ============================================================

def calculate_local_contrast(
    image,
    roi_mask,
    window_size
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)


    # --------------------------------------------------------
    # IMPORTANT:
    # Calculate local mean using ONLY valid ROI pixels.
    #
    # This avoids black background pixels influencing fruit
    # pixels close to the ROI boundary.
    # --------------------------------------------------------

    roi_float = (
        roi_mask > 0
    ).astype(np.float32)


    # Sum of intensities inside neighbourhood
    local_sum = cv2.boxFilter(
        gray * roi_float,
        ddepth=-1,
        ksize=(
            window_size,
            window_size
        ),
        normalize=False
    )


    # Number/weight of valid ROI pixels
    local_count = cv2.boxFilter(
        roi_float,
        ddepth=-1,
        ksize=(
            window_size,
            window_size
        ),
        normalize=False
    )


    # ROI-aware local mean
    local_mean = (
        local_sum
        / np.maximum(
            local_count,
            1.0
        )
    )


    # --------------------------------------------------------
    # LOCAL DARKNESS
    #
    # Positive when current pixel is darker than surrounding
    # fruit surface.
    # --------------------------------------------------------

    local_difference = (
        local_mean - gray
    )

    local_difference = np.maximum(
        local_difference,
        0.0
    )

    local_difference[
        roi_mask == 0
    ] = 0.0


    # --------------------------------------------------------
    # NORMALIZE WITHIN ROI
    # --------------------------------------------------------

    roi_values = local_difference[
        roi_mask > 0
    ]

    normalized = np.zeros(
        gray.shape,
        dtype=np.float32
    )


    if len(roi_values) > 0:

        # Robust normalization:
        # use 95th percentile rather than maximum so a single
        # extreme pixel does not dominate the entire map.
        scale = np.percentile(
            roi_values,
            95
        )

        if scale > 1e-8:

            normalized = (
                local_difference
                / scale
            )

            normalized = np.clip(
                normalized,
                0.0,
                1.0
            )


    normalized[
        roi_mask == 0
    ] = 0.0

    return normalized


# ============================================================
# 5. LOCAL-CONTRAST-GUIDED OTSU
# ============================================================

def local_contrast_guided_otsu(
    image,
    roi_mask,
    window_size
):

    otsu_mask = otsu_segmentation(
        image,
        roi_mask
    )

    local_contrast = (
        calculate_local_contrast(
            image,
            roi_mask,
            window_size
        )
    )


    # --------------------------------------------------------
    # Automatically determine unusual local contrast.
    # --------------------------------------------------------

    roi_values = local_contrast[
        roi_mask > 0
    ]

    if len(roi_values) == 0:

        return np.zeros_like(
            roi_mask
        )


    threshold = (
        np.mean(roi_values)
        + 0.5 * np.std(roi_values)
    )


    contrast_mask = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )

    contrast_mask[
        (
            local_contrast
            > threshold
        )
        & (roi_mask > 0)
    ] = 255


    # --------------------------------------------------------
    # Require agreement:
    #
    # Otsu says pixel is dark
    # AND
    # Local contrast says pixel is unusually dark relative
    # to its surrounding fruit surface.
    # --------------------------------------------------------

    refined = cv2.bitwise_and(
        otsu_mask,
        contrast_mask
    )


    refined = apply_morphology(
        refined,
        roi_mask
    )

    return refined


# ============================================================
# 6. WEIGHTED OTSU + LOCAL CONTRAST
# ============================================================

def weighted_local_contrast(
    image,
    roi_mask,
    window_size,
    darkness_weight,
    local_weight
):

    darkness = calculate_darkness_score(
        image,
        roi_mask
    )

    local_contrast = (
        calculate_local_contrast(
            image,
            roi_mask,
            window_size
        )
    )


    # --------------------------------------------------------
    # COMBINED BLEMISH SCORE
    #
    # B(x) =
    # w1 * darkness
    # +
    # w2 * local contrast
    # --------------------------------------------------------

    blemish_score = (
        darkness_weight
        * darkness
        +
        local_weight
        * local_contrast
    )


    blemish_score[
        roi_mask == 0
    ] = 0.0


    # --------------------------------------------------------
    # AUTOMATIC THRESHOLD ON COMBINED SCORE
    # --------------------------------------------------------

    score_uint8 = np.clip(
        blemish_score * 255.0,
        0,
        255
    ).astype(np.uint8)


    roi_scores = score_uint8[
        roi_mask > 0
    ]


    if len(roi_scores) == 0:

        return np.zeros_like(
            roi_mask
        )


    threshold, _ = cv2.threshold(
        roi_scores,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    final_mask = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )


    final_mask[
        (
            score_uint8 > threshold
        )
        & (roi_mask > 0)
    ] = 255


    # --------------------------------------------------------
    # FINAL MORPHOLOGICAL REFINEMENT
    # --------------------------------------------------------

    final_mask = apply_morphology(
        final_mask,
        roi_mask
    )

    return final_mask


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def calculate_metrics(
    predicted_mask,
    ground_truth_mask
):

    predicted = (
        predicted_mask > 0
    )

    ground_truth = (
        ground_truth_mask > 0
    )


    tp = np.logical_and(
        predicted,
        ground_truth
    ).sum()

    fp = np.logical_and(
        predicted,
        np.logical_not(ground_truth)
    ).sum()

    fn = np.logical_and(
        np.logical_not(predicted),
        ground_truth
    ).sum()


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_denominator = (
        tp + fp + fn
    )

    if iou_denominator == 0:
        iou = 1.0
    else:
        iou = (
            tp / iou_denominator
        )


    # --------------------------------------------------------
    # DICE
    # --------------------------------------------------------

    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * tp
            / dice_denominator
        )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    if tp + fp == 0:

        precision = (
            1.0
            if ground_truth.sum() == 0
            else 0.0
        )

    else:

        precision = (
            tp / (tp + fp)
        )


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    if tp + fn == 0:

        recall = (
            1.0
            if predicted.sum() == 0
            else 0.0
        )

    else:

        recall = (
            tp / (tp + fn)
        )


    return (
        iou,
        dice,
        precision,
        recall
    )


# ============================================================
# 8. STORE RESULT
# ============================================================

def add_result(
    results,
    row,
    filename,
    method,
    mask,
    ground_truth,
    window_size=None,
    darkness_weight=None,
    local_weight=None
):

    (
        iou,
        dice,
        precision,
        recall
    ) = calculate_metrics(
        mask,
        ground_truth
    )


    results.append({

        "fruit":
            row["fruit"],

        "category":
            row["category"],

        "image":
            filename,

        "method":
            method,

        "window_size":
            window_size,

        "darkness_weight":
            darkness_weight,

        "local_contrast_weight":
            local_weight,

        "iou":
            iou,

        "dice":
            dice,

        "precision":
            precision,

        "recall":
            recall
    })


# ============================================================
# 9. LOAD GROUND-TRUTH SELECTION
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

results = []


print("\n==========================================")
print("LOCAL CONTRAST-GUIDED OTSU")
print("==========================================")

print(
    f"Ground-truth images: {len(df)}"
)

print(
    f"Window sizes: {WINDOW_SIZES}"
)

print(
    "Weight configurations:"
)

for darkness_weight, local_weight in (
    WEIGHT_CONFIGURATIONS
):

    print(
        f"  Darkness={darkness_weight:.2f}, "
        f"Local Contrast={local_weight:.2f}"
    )


# ============================================================
# 10. PROCESS GROUND-TRUTH IMAGES
# ============================================================

for index, row in df.iterrows():

    relative_path = row[
        "relative_path"
    ]

    relative_folder = os.path.dirname(
        relative_path
    )

    filename = os.path.basename(
        relative_path
    )

    image_name = os.path.splitext(
        filename
    )[0]


    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    image_path = os.path.join(
        PROCESSED_ROOT,
        relative_path
    )

    roi_path = os.path.join(
        ROI_ROOT,
        relative_folder,
        image_name + "_mask.png"
    )

    gt_path = os.path.join(
        GROUND_TRUTH_ROOT,
        row["fruit"],
        image_name + "_gt.png"
    )


    # --------------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    roi_mask = cv2.imread(
        roi_path,
        cv2.IMREAD_GRAYSCALE
    )

    ground_truth = cv2.imread(
        gt_path,
        cv2.IMREAD_GRAYSCALE
    )


    if (
        image is None
        or roi_mask is None
        or ground_truth is None
    ):

        print(
            f"SKIPPED: {filename}"
        )

        continue


    # Evaluate only inside fruit ROI
    ground_truth[
        roi_mask == 0
    ] = 0


    print(
        f"\n[{index + 1}/{len(df)}] "
        f"{row['fruit']} | "
        f"{row['category']} | "
        f"{filename}"
    )


    # ========================================================
    # A. ORIGINAL OTSU
    # ========================================================

    otsu = otsu_segmentation(
        image,
        roi_mask
    )


    add_result(
        results,
        row,
        filename,
        "Original Otsu",
        otsu,
        ground_truth
    )


    # ========================================================
    # B. CURRENT BEST
    # ========================================================

    otsu_morph = apply_morphology(
        otsu,
        roi_mask
    )


    add_result(
        results,
        row,
        filename,
        "Otsu + Morphology 7x7",
        otsu_morph,
        ground_truth
    )


    # ========================================================
    # C/D. LOCAL CONTRAST EXPERIMENTS
    # ========================================================

    for window_size in WINDOW_SIZES:


        # ----------------------------------------------------
        # STRICT LOCAL-CONTRAST GUIDANCE
        # ----------------------------------------------------

        guided = (
            local_contrast_guided_otsu(
                image,
                roi_mask,
                window_size
            )
        )


        add_result(
            results,
            row,
            filename,
            (
                "Local Contrast-Guided Otsu "
                f"({window_size}x{window_size})"
            ),
            guided,
            ground_truth,
            window_size=window_size
        )


        # ----------------------------------------------------
        # WEIGHTED VERSIONS
        # ----------------------------------------------------

        for (
            darkness_weight,
            local_weight
        ) in WEIGHT_CONFIGURATIONS:

            weighted = (
                weighted_local_contrast(
                    image,
                    roi_mask,
                    window_size,
                    darkness_weight,
                    local_weight
                )
            )


            method_name = (
                f"Weighted Otsu-Local "
                f"({darkness_weight:.1f}/"
                f"{local_weight:.1f}) "
                f"{window_size}x{window_size}"
            )


            add_result(
                results,
                row,
                filename,
                method_name,
                weighted,
                ground_truth,
                window_size,
                darkness_weight,
                local_weight
            )


# ============================================================
# 11. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 12. SAVE DETAILED RESULTS
# ============================================================

detailed_path = os.path.join(
    OUTPUT_ROOT,
    "local_contrast_detailed.csv"
)


results_df.to_csv(
    detailed_path,
    index=False
)


# ============================================================
# 13. SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby(
        [
            "method",
            "window_size",
            "darkness_weight",
            "local_contrast_weight"
        ],
        dropna=False
    )[
        [
            "iou",
            "dice",
            "precision",
            "recall"
        ]
    ]
    .mean()
    .reset_index()
)


# ============================================================
# OVERALL SCORE
# ============================================================

summary_df[
    "overall_score"
] = summary_df[
    [
        "iou",
        "dice",
        "precision",
        "recall"
    ]
].mean(
    axis=1
)


# ============================================================
# RANK PRIMARILY BY IoU
# ============================================================

summary_df = summary_df.sort_values(
    by=[
        "iou",
        "dice",
        "overall_score"
    ],
    ascending=False
)


# ============================================================
# 14. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_ROOT,
    "local_contrast_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 15. PRINT SUMMARY
# ============================================================

print("\n\n==========================================")
print("LOCAL CONTRAST ENHANCEMENT SUMMARY")
print("==========================================")


for _, result in summary_df.iterrows():

    print(
        f"\n{result['method']}"
    )

    print(
        f"  Mean IoU       : "
        f"{result['iou']:.4f}"
    )

    print(
        f"  Mean Dice      : "
        f"{result['dice']:.4f}"
    )

    print(
        f"  Mean Precision : "
        f"{result['precision']:.4f}"
    )

    print(
        f"  Mean Recall    : "
        f"{result['recall']:.4f}"
    )

    print(
        f"  Overall Score  : "
        f"{result['overall_score']:.4f}"
    )


# ============================================================
# 16. BEST CONFIGURATION
# ============================================================

best = summary_df.iloc[0]


print("\n==========================================")
print("BEST LOCAL-CONTRAST CONFIGURATION")
print("==========================================")

print(
    f"Method : {best['method']}"
)

print(
    f"Window Size : "
    f"{best['window_size']}"
)

print(
    f"Darkness Weight : "
    f"{best['darkness_weight']}"
)

print(
    f"Local Weight : "
    f"{best['local_contrast_weight']}"
)

print(
    f"Mean IoU : "
    f"{best['iou']:.4f}"
)

print(
    f"Mean Dice : "
    f"{best['dice']:.4f}"
)

print(
    f"Precision : "
    f"{best['precision']:.4f}"
)

print(
    f"Recall : "
    f"{best['recall']:.4f}"
)

print(
    f"Overall Score : "
    f"{best['overall_score']:.4f}"
)

print("==========================================")


print(
    f"\nDetailed results saved to:\n"
    f"{detailed_path}"
)

print(
    f"\nSummary saved to:\n"
    f"{summary_path}"
)