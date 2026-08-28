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
    "results/blemish_detection/advanced_otsu"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# PARAMETERS TO TEST
# ============================================================

HSV_K_VALUES = [
    0.50,
    0.75,
    1.00
]

COMPONENT_RATIOS = [
    0.0005,   # 0.05%
    0.0010,   # 0.10%
    0.0025    # 0.25%
]

MORPH_KERNEL_SIZE = 7


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
# 2. HSV ABNORMALITY
# ============================================================

def calculate_hsv_abnormality(
    image,
    roi_mask,
    k
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    roi_pixels = hsv[
        roi_mask > 0
    ]

    if len(roi_pixels) == 0:

        return np.zeros(
            roi_mask.shape,
            dtype=np.uint8
        )

    # Typical colour of current fruit
    median_h = np.median(
        roi_pixels[:, 0]
    )

    median_s = np.median(
        roi_pixels[:, 1]
    )

    median_v = np.median(
        roi_pixels[:, 2]
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]


    # --------------------------------------------------------
    # HUE DIFFERENCE
    # Hue is circular in OpenCV: 0-179
    # --------------------------------------------------------

    h_diff = np.abs(
        h - median_h
    )

    h_diff = np.minimum(
        h_diff,
        180 - h_diff
    )

    h_diff = (
        h_diff / 90.0
    )


    # --------------------------------------------------------
    # SATURATION DIFFERENCE
    # --------------------------------------------------------

    s_diff = (
        np.abs(s - median_s)
        / 255.0
    )


    # --------------------------------------------------------
    # VALUE DIFFERENCE
    # --------------------------------------------------------

    v_diff = (
        np.abs(v - median_v)
        / 255.0
    )


    # --------------------------------------------------------
    # COMBINED HSV DISTANCE
    # --------------------------------------------------------

    difference = np.sqrt(
        h_diff ** 2
        + s_diff ** 2
        + v_diff ** 2
    )


    roi_difference = difference[
        roi_mask > 0
    ]


    # Adaptive threshold
    threshold = (
        np.mean(roi_difference)
        + k * np.std(roi_difference)
    )


    abnormal_mask = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )

    abnormal_mask[
        (difference > threshold)
        & (roi_mask > 0)
    ] = 255

    return abnormal_mask


# ============================================================
# 3. HSV-GUIDED OTSU REFINEMENT
# ============================================================

def hsv_guided_otsu(
    image,
    roi_mask,
    otsu_mask,
    hsv_mask
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    fruit_pixels = gray[
        roi_mask > 0
    ]

    if len(fruit_pixels) == 0:
        return np.zeros_like(gray)


    # --------------------------------------------------------
    # STRONG DARKNESS THRESHOLD
    #
    # Pixels considerably darker than the typical fruit
    # colour are retained even if HSV does not identify them.
    # --------------------------------------------------------

    median_intensity = np.median(
        fruit_pixels
    )

    std_intensity = np.std(
        fruit_pixels
    )

    strong_dark_threshold = (
        median_intensity
        - 1.0 * std_intensity
    )


    strong_dark = np.zeros_like(
        gray
    )

    strong_dark[
        (gray < strong_dark_threshold)
        & (roi_mask > 0)
    ] = 255


    # --------------------------------------------------------
    # OTSU + HSV AGREEMENT
    # --------------------------------------------------------

    otsu_hsv_agreement = cv2.bitwise_and(
        otsu_mask,
        hsv_mask
    )


    # --------------------------------------------------------
    # FINAL GUIDED MASK
    #
    # Keep:
    # 1. Otsu pixels also abnormal in HSV
    # OR
    # 2. Very strongly dark Otsu pixels
    # --------------------------------------------------------

    strong_dark_otsu = cv2.bitwise_and(
        otsu_mask,
        strong_dark
    )

    refined = cv2.bitwise_or(
        otsu_hsv_agreement,
        strong_dark_otsu
    )


    refined[
        roi_mask == 0
    ] = 0

    return refined


# ============================================================
# 4. MORPHOLOGY
# ============================================================

def apply_morphology(
    mask,
    roi_mask
):

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
# 5. REMOVE SMALL COMPONENTS
# ============================================================

def remove_small_components(
    mask,
    roi_mask,
    minimum_ratio
):

    fruit_area = np.count_nonzero(
        roi_mask
    )

    if fruit_area == 0:
        return mask


    minimum_area = (
        fruit_area
        * minimum_ratio
    )


    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )


    cleaned = np.zeros_like(
        mask
    )


    # Label 0 = background
    for label in range(
        1,
        number_labels
    ):

        component_area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if component_area >= minimum_area:

            cleaned[
                labels == label
            ] = 255


    cleaned[
        roi_mask == 0
    ] = 0

    return cleaned


# ============================================================
# 6. METRICS
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
# 7. ADD RESULT
# ============================================================

def add_result(
    results,
    row,
    filename,
    method,
    mask,
    ground_truth,
    k=None,
    component_ratio=None
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

        "hsv_k":
            k,

        "component_ratio":
            component_ratio,

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
# LOAD DATA
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

results = []


print("\n==========================================")
print("ADVANCED OTSU ENHANCEMENT")
print("==========================================")

print(
    f"Ground-truth images: {len(df)}"
)

print(
    f"HSV k values: {HSV_K_VALUES}"
)

print(
    f"Component ratios: {COMPONENT_RATIOS}"
)


# ============================================================
# PROCESS 27 GROUND-TRUTH IMAGES
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
    # ORIGINAL OTSU
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
    # CURRENT BEST: OTSU + MORPH 7x7
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
    # ADVANCED CONFIGURATIONS
    # ========================================================

    for k in HSV_K_VALUES:

        hsv_mask = calculate_hsv_abnormality(
            image,
            roi_mask,
            k
        )


        # ----------------------------------------------------
        # HSV-GUIDED OTSU
        # ----------------------------------------------------

        guided = hsv_guided_otsu(
            image,
            roi_mask,
            otsu,
            hsv_mask
        )


        # ----------------------------------------------------
        # GUIDED + MORPHOLOGY
        # ----------------------------------------------------

        guided_morph = apply_morphology(
            guided,
            roi_mask
        )


        method_name = (
            f"Otsu + HSV(k={k}) "
            f"+ Morphology"
        )

        add_result(
            results,
            row,
            filename,
            method_name,
            guided_morph,
            ground_truth,
            k=k
        )


        # ----------------------------------------------------
        # GUIDED + MORPH + COMPONENT REMOVAL
        # ----------------------------------------------------

        for ratio in COMPONENT_RATIOS:

            cleaned = remove_small_components(
                guided_morph,
                roi_mask,
                ratio
            )

            method_name = (
                f"Otsu + HSV(k={k}) "
                f"+ Morph + Components"
                f"({ratio})"
            )

            add_result(
                results,
                row,
                filename,
                method_name,
                cleaned,
                ground_truth,
                k=k,
                component_ratio=ratio
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

detailed_path = os.path.join(
    OUTPUT_ROOT,
    "advanced_otsu_detailed.csv"
)

results_df.to_csv(
    detailed_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby(
        [
            "method",
            "hsv_k",
            "component_ratio"
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
].mean(axis=1)


# Primary ranking: IoU
summary_df = summary_df.sort_values(
    by=[
        "iou",
        "dice",
        "overall_score"
    ],
    ascending=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_ROOT,
    "advanced_otsu_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n\n==========================================")
print("ADVANCED OTSU ENHANCEMENT SUMMARY")
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
# BEST CONFIGURATION
# ============================================================

best = summary_df.iloc[0]


print("\n==========================================")
print("BEST ADVANCED CONFIGURATION")
print("==========================================")

print(
    f"Method    : {best['method']}"
)

print(
    f"HSV k     : {best['hsv_k']}"
)

print(
    f"Component : {best['component_ratio']}"
)

print(
    f"Mean IoU  : {best['iou']:.4f}"
)

print(
    f"Mean Dice : {best['dice']:.4f}"
)

print(
    f"Precision : {best['precision']:.4f}"
)

print(
    f"Recall    : {best['recall']:.4f}"
)

print(
    f"Score     : {best['overall_score']:.4f}"
)

print("==========================================")


print(
    f"\nDetailed results:\n{detailed_path}"
)

print(
    f"\nSummary:\n{summary_path}"
)