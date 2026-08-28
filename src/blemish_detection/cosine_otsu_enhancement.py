import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = (
    "results/blemish_detection/"
    "ground_truth_selection.csv"
)

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
    "results/blemish_detection/cosine_otsu"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

MORPH_KERNEL_SIZE = 7

# Intensity weight, cosine weight
WEIGHT_CONFIGURATIONS = [
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50)
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
# 2. MORPHOLOGY
# ============================================================

def apply_morphology(mask, roi_mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    closed[
        roi_mask == 0
    ] = 0

    return closed


# ============================================================
# 3. REPRESENTATIVE FRUIT COLOUR
# ============================================================

def get_reference_colour(image, roi_mask):

    # OpenCV loads BGR, convert to RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    fruit_pixels = rgb[
        roi_mask > 0
    ]

    if len(fruit_pixels) == 0:

        return np.array(
            [0, 0, 0],
            dtype=np.float32
        )

    # Median is less affected by abnormal pixels
    # compared with the mean.
    reference = np.median(
        fruit_pixels,
        axis=0
    )

    return reference


# ============================================================
# 4. COSINE ABNORMALITY MAP
# ============================================================

def calculate_cosine_abnormality(
    image,
    roi_mask
):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    reference = get_reference_colour(
        image,
        roi_mask
    )


    # --------------------------------------------------------
    # DOT PRODUCT
    # --------------------------------------------------------

    dot_product = np.sum(
        rgb * reference,
        axis=2
    )


    # --------------------------------------------------------
    # VECTOR MAGNITUDES
    # --------------------------------------------------------

    pixel_norm = np.linalg.norm(
        rgb,
        axis=2
    )

    reference_norm = np.linalg.norm(
        reference
    )


    # Avoid division by zero
    denominator = (
        pixel_norm
        * reference_norm
        + 1e-8
    )


    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    cosine_similarity = (
        dot_product
        / denominator
    )

    cosine_similarity = np.clip(
        cosine_similarity,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # ABNORMALITY
    #
    # Similar colour:
    # cosine ≈ 1
    # abnormality ≈ 0
    #
    # Different colour:
    # cosine lower
    # abnormality higher
    # --------------------------------------------------------

    abnormality = (
        1.0
        - cosine_similarity
    )


    abnormality[
        roi_mask == 0
    ] = 0.0

    return abnormality


# ============================================================
# 5. NORMALIZED DARKNESS MAP
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


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    threshold, _ = cv2.threshold(
        fruit_pixels.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    # --------------------------------------------------------
    # DARKNESS SCORE
    #
    # Darker than Otsu threshold → positive score
    #
    # threshold = 100
    # pixel = 50
    #
    # score = (100 - 50) / 100 = 0.5
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
# 6. COSINE-GUIDED OTSU
# ============================================================

def cosine_guided_otsu(
    image,
    roi_mask
):

    otsu_mask = otsu_segmentation(
        image,
        roi_mask
    )

    cosine_abnormality = (
        calculate_cosine_abnormality(
            image,
            roi_mask
        )
    )


    roi_values = cosine_abnormality[
        roi_mask > 0
    ]

    if len(roi_values) == 0:

        return np.zeros_like(
            roi_mask
        )


    # --------------------------------------------------------
    # ADAPTIVE COSINE THRESHOLD
    #
    # Avoid manually defining one fixed similarity threshold
    # for every fruit.
    # --------------------------------------------------------

    cosine_threshold = (
        np.mean(roi_values)
        + 0.5 * np.std(roi_values)
    )


    abnormal_colour = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )

    abnormal_colour[
        (
            cosine_abnormality
            > cosine_threshold
        )
        & (roi_mask > 0)
    ] = 255


    # --------------------------------------------------------
    # GUIDED SEGMENTATION
    #
    # Pixel must be:
    # 1. Otsu blemish candidate
    # 2. Colour abnormal
    # --------------------------------------------------------

    guided = cv2.bitwise_and(
        otsu_mask,
        abnormal_colour
    )


    guided = apply_morphology(
        guided,
        roi_mask
    )

    return guided


# ============================================================
# 7. WEIGHTED OTSU + COSINE
# ============================================================

def weighted_otsu_cosine(
    image,
    roi_mask,
    intensity_weight,
    cosine_weight
):

    darkness = calculate_darkness_score(
        image,
        roi_mask
    )

    cosine_abnormality = (
        calculate_cosine_abnormality(
            image,
            roi_mask
        )
    )


    # --------------------------------------------------------
    # NORMALIZE COSINE ABNORMALITY
    #
    # Cosine abnormality values can naturally be very small,
    # so normalize within the fruit ROI before combining.
    # --------------------------------------------------------

    roi_cosine = cosine_abnormality[
        roi_mask > 0
    ]

    normalized_cosine = np.zeros_like(
        cosine_abnormality,
        dtype=np.float32
    )


    if len(roi_cosine) > 0:

        minimum = np.min(
            roi_cosine
        )

        maximum = np.max(
            roi_cosine
        )

        difference = (
            maximum - minimum
        )

        if difference > 1e-8:

            normalized_cosine[
                roi_mask > 0
            ] = (
                (
                    roi_cosine
                    - minimum
                )
                / difference
            )


    # --------------------------------------------------------
    # WEIGHTED BLEMISH SCORE
    #
    # B = w1(Darkness) + w2(Cosine abnormality)
    # --------------------------------------------------------

    blemish_score = (
        intensity_weight
        * darkness
        +
        cosine_weight
        * normalized_cosine
    )


    roi_scores = blemish_score[
        roi_mask > 0
    ]


    if len(roi_scores) == 0:

        return np.zeros_like(
            roi_mask
        )


    # --------------------------------------------------------
    # AUTOMATIC THRESHOLD ON COMBINED SCORE
    #
    # Convert score 0-1 → 0-255, then use Otsu again.
    # --------------------------------------------------------

    score_uint8 = np.clip(
        blemish_score * 255,
        0,
        255
    ).astype(np.uint8)


    roi_score_uint8 = score_uint8[
        roi_mask > 0
    ]


    threshold, _ = cv2.threshold(
        roi_score_uint8,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
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
    # MORPHOLOGICAL REFINEMENT
    # --------------------------------------------------------

    final_mask = apply_morphology(
        final_mask,
        roi_mask
    )

    return final_mask


# ============================================================
# 8. METRICS
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

    denominator = (
        tp + fp + fn
    )

    if denominator == 0:
        iou = 1.0
    else:
        iou = tp / denominator


    # --------------------------------------------------------
    # DICE
    # --------------------------------------------------------

    denominator = (
        2 * tp + fp + fn
    )

    if denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * tp
            / denominator
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
# 9. ADD RESULT
# ============================================================

def add_result(
    results,
    row,
    filename,
    method,
    mask,
    ground_truth,
    intensity_weight=None,
    cosine_weight=None
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

        "intensity_weight":
            intensity_weight,

        "cosine_weight":
            cosine_weight,

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
# 10. LOAD GROUND-TRUTH SELECTION
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

results = []


print("\n==========================================")
print("COSINE-GUIDED OTSU ENHANCEMENT")
print("==========================================")

print(
    f"Ground-truth images: {len(df)}"
)

print(
    "Weight configurations:"
)

for intensity_weight, cosine_weight in (
    WEIGHT_CONFIGURATIONS
):

    print(
        f"  Intensity={intensity_weight:.2f}, "
        f"Cosine={cosine_weight:.2f}"
    )


# ============================================================
# 11. PROCESS ALL 27 GROUND-TRUTH IMAGES
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
    # LOAD
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


    # Ground truth should only be evaluated
    # inside fruit ROI
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
    # METHOD A — ORIGINAL OTSU
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
    # METHOD B — CURRENT BEST
    # OTSU + MORPHOLOGY 7x7
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
    # METHOD C — COSINE-GUIDED OTSU
    # ========================================================

    cosine_guided = cosine_guided_otsu(
        image,
        roi_mask
    )

    add_result(
        results,
        row,
        filename,
        "Cosine-Guided Otsu + Morphology",
        cosine_guided,
        ground_truth
    )


    # ========================================================
    # METHOD D — WEIGHTED OTSU + COSINE
    # ========================================================

    for (
        intensity_weight,
        cosine_weight
    ) in WEIGHT_CONFIGURATIONS:

        weighted_mask = (
            weighted_otsu_cosine(
                image,
                roi_mask,
                intensity_weight,
                cosine_weight
            )
        )


        method_name = (
            f"Weighted Otsu-Cosine "
            f"({intensity_weight:.1f}/"
            f"{cosine_weight:.1f}) "
            f"+ Morphology"
        )


        add_result(
            results,
            row,
            filename,
            method_name,
            weighted_mask,
            ground_truth,
            intensity_weight,
            cosine_weight
        )


# ============================================================
# 12. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 13. SAVE DETAILED RESULTS
# ============================================================

detailed_path = os.path.join(
    OUTPUT_ROOT,
    "cosine_otsu_detailed.csv"
)

results_df.to_csv(
    detailed_path,
    index=False
)


# ============================================================
# 14. CREATE SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby(
        [
            "method",
            "intensity_weight",
            "cosine_weight"
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
# 15. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_ROOT,
    "cosine_otsu_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n\n==========================================")
print("COSINE OTSU ENHANCEMENT SUMMARY")
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
# 17. BEST CONFIGURATION
# ============================================================

best = summary_df.iloc[0]


print("\n==========================================")
print("BEST COSINE/OTSU CONFIGURATION")
print("==========================================")

print(
    f"Method : {best['method']}"
)

print(
    f"Intensity Weight : "
    f"{best['intensity_weight']}"
)

print(
    f"Cosine Weight    : "
    f"{best['cosine_weight']}"
)

print(
    f"Mean IoU         : "
    f"{best['iou']:.4f}"
)

print(
    f"Mean Dice        : "
    f"{best['dice']:.4f}"
)

print(
    f"Mean Precision   : "
    f"{best['precision']:.4f}"
)

print(
    f"Mean Recall      : "
    f"{best['recall']:.4f}"
)

print(
    f"Overall Score    : "
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