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
    "results/blemish_detection/multicue_bcs"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

MORPH_KERNEL_SIZE = 7
LOCAL_WINDOW_SIZE = 31
SPATIAL_WINDOW_SIZE = 11


# D, L, C, D*C, S
CONFIGURATIONS = {

    "MC-BCS A": (
        0.50,
        0.10,
        0.10,
        0.20,
        0.10
    ),

    "MC-BCS B": (
        0.40,
        0.10,
        0.10,
        0.30,
        0.10
    ),

    "MC-BCS C": (
        0.50,
        0.05,
        0.05,
        0.30,
        0.10
    ),

    "MC-BCS D": (
        0.60,
        0.05,
        0.05,
        0.20,
        0.10
    )
}


# ============================================================
# ORIGINAL OTSU
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
# MORPHOLOGY
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
# DARKNESS SCORE — D
# ============================================================

def calculate_darkness(
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

    result = np.zeros_like(
        gray,
        dtype=np.float32
    )

    if len(fruit_pixels) == 0:
        return result

    threshold, _ = cv2.threshold(
        fruit_pixels.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    result = (
        threshold - gray
    ) / max(
        float(threshold),
        1.0
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    result[
        roi_mask == 0
    ] = 0

    return result


# ============================================================
# LOCAL CONTRAST — L
# ============================================================

def calculate_local_contrast(
    image,
    roi_mask
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    roi_float = (
        roi_mask > 0
    ).astype(np.float32)


    local_sum = cv2.boxFilter(
        gray * roi_float,
        -1,
        (
            LOCAL_WINDOW_SIZE,
            LOCAL_WINDOW_SIZE
        ),
        normalize=False
    )

    local_count = cv2.boxFilter(
        roi_float,
        -1,
        (
            LOCAL_WINDOW_SIZE,
            LOCAL_WINDOW_SIZE
        ),
        normalize=False
    )

    local_mean = (
        local_sum
        / np.maximum(
            local_count,
            1.0
        )
    )

    difference = (
        local_mean - gray
    )

    difference = np.maximum(
        difference,
        0.0
    )

    difference[
        roi_mask == 0
    ] = 0


    values = difference[
        roi_mask > 0
    ]

    normalized = np.zeros_like(
        difference,
        dtype=np.float32
    )

    if len(values) > 0:

        scale = np.percentile(
            values,
            95
        )

        if scale > 1e-8:

            normalized = (
                difference / scale
            )

            normalized = np.clip(
                normalized,
                0.0,
                1.0
            )

    normalized[
        roi_mask == 0
    ] = 0

    return normalized


# ============================================================
# COSINE COLOUR ABNORMALITY — C
# ============================================================

def calculate_cosine_abnormality(
    image,
    roi_mask
):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)


    fruit_pixels = rgb[
        roi_mask > 0
    ]


    result = np.zeros(
        roi_mask.shape,
        dtype=np.float32
    )


    if len(fruit_pixels) == 0:
        return result


    reference = np.median(
        fruit_pixels,
        axis=0
    )


    dot_product = np.sum(
        rgb * reference,
        axis=2
    )


    pixel_norm = np.linalg.norm(
        rgb,
        axis=2
    )

    reference_norm = np.linalg.norm(
        reference
    )


    similarity = (
        dot_product
        /
        (
            pixel_norm
            * reference_norm
            + 1e-8
        )
    )


    similarity = np.clip(
        similarity,
        0.0,
        1.0
    )


    abnormality = (
        1.0 - similarity
    )


    # --------------------------------------------------------
    # NORMALIZE INSIDE ROI
    # --------------------------------------------------------

    roi_values = abnormality[
        roi_mask > 0
    ]


    if len(roi_values) > 0:

        scale = np.percentile(
            roi_values,
            95
        )

        if scale > 1e-8:

            result = (
                abnormality / scale
            )

            result = np.clip(
                result,
                0.0,
                1.0
            )


    result[
        roi_mask == 0
    ] = 0

    return result


# ============================================================
# SPATIAL CONFIDENCE — S
# ============================================================

def calculate_spatial_confidence(
    candidate_score,
    roi_mask
):

    # Convert candidate confidence into a preliminary
    # suspicious-pixel map.

    roi_values = candidate_score[
        roi_mask > 0
    ]


    result = np.zeros(
        candidate_score.shape,
        dtype=np.float32
    )


    if len(roi_values) == 0:
        return result


    threshold = np.percentile(
        roi_values,
        65
    )


    suspicious = np.zeros(
        candidate_score.shape,
        dtype=np.float32
    )


    suspicious[
        (
            candidate_score >= threshold
        )
        & (roi_mask > 0)
    ] = 1.0


    roi_float = (
        roi_mask > 0
    ).astype(np.float32)


    suspicious_sum = cv2.boxFilter(
        suspicious,
        -1,
        (
            SPATIAL_WINDOW_SIZE,
            SPATIAL_WINDOW_SIZE
        ),
        normalize=False
    )


    roi_count = cv2.boxFilter(
        roi_float,
        -1,
        (
            SPATIAL_WINDOW_SIZE,
            SPATIAL_WINDOW_SIZE
        ),
        normalize=False
    )


    result = (
        suspicious_sum
        /
        np.maximum(
            roi_count,
            1.0
        )
    )


    result = np.clip(
        result,
        0.0,
        1.0
    )


    result[
        roi_mask == 0
    ] = 0


    return result


# ============================================================
# MC-BCS
# ============================================================

def multicue_bcs(
    image,
    roi_mask,
    weights
):

    (
        w_dark,
        w_local,
        w_cosine,
        w_interaction,
        w_spatial
    ) = weights


    # --------------------------------------------------------
    # CALCULATE INDIVIDUAL CUES
    # --------------------------------------------------------

    darkness = calculate_darkness(
        image,
        roi_mask
    )

    local = calculate_local_contrast(
        image,
        roi_mask
    )

    cosine = calculate_cosine_abnormality(
        image,
        roi_mask
    )


    # --------------------------------------------------------
    # INTERACTION
    #
    # High when pixel is BOTH:
    # - dark
    # - colour abnormal
    # --------------------------------------------------------

    interaction = (
        darkness * cosine
    )


    # --------------------------------------------------------
    # INITIAL CONFIDENCE
    # --------------------------------------------------------

    initial_score = (
        w_dark * darkness
        +
        w_local * local
        +
        w_cosine * cosine
        +
        w_interaction * interaction
    )


    # --------------------------------------------------------
    # SPATIAL CONFIDENCE
    # --------------------------------------------------------

    spatial = calculate_spatial_confidence(
        initial_score,
        roi_mask
    )


    # --------------------------------------------------------
    # FINAL BCS
    # --------------------------------------------------------

    score = (
        initial_score
        +
        w_spatial * spatial
    )


    score[
        roi_mask == 0
    ] = 0


    # Normalize final score inside ROI
    roi_scores = score[
        roi_mask > 0
    ]


    if len(roi_scores) == 0:

        return np.zeros_like(
            roi_mask
        )


    minimum = np.min(
        roi_scores
    )

    maximum = np.max(
        roi_scores
    )


    normalized = np.zeros_like(
        score,
        dtype=np.float32
    )


    if maximum - minimum > 1e-8:

        normalized[
            roi_mask > 0
        ] = (
            roi_scores - minimum
        ) / (
            maximum - minimum
        )


    # --------------------------------------------------------
    # AUTOMATIC OTSU THRESHOLD ON BCS
    # --------------------------------------------------------

    score_uint8 = np.clip(
        normalized * 255,
        0,
        255
    ).astype(np.uint8)


    threshold, _ = cv2.threshold(
        score_uint8[
            roi_mask > 0
        ],
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )


    mask = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )


    mask[
        (
            score_uint8 > threshold
        )
        & (roi_mask > 0)
    ] = 255


    # --------------------------------------------------------
    # FINAL 7x7 MORPHOLOGY
    # --------------------------------------------------------

    mask = apply_morphology(
        mask,
        roi_mask
    )


    return mask


# ============================================================
# METRICS
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
        ~ground_truth
    ).sum()

    fn = np.logical_and(
        ~predicted,
        ground_truth
    ).sum()


    # IoU
    denominator = (
        tp + fp + fn
    )

    iou = (
        tp / denominator
        if denominator > 0
        else 1.0
    )


    # Dice
    denominator = (
        2 * tp + fp + fn
    )

    dice = (
        2 * tp / denominator
        if denominator > 0
        else 1.0
    )


    # Precision
    if tp + fp > 0:

        precision = (
            tp / (tp + fp)
        )

    else:

        precision = (
            1.0
            if ground_truth.sum() == 0
            else 0.0
        )


    # Recall
    if tp + fn > 0:

        recall = (
            tp / (tp + fn)
        )

    else:

        recall = (
            1.0
            if predicted.sum() == 0
            else 0.0
        )


    return (
        iou,
        dice,
        precision,
        recall
    )


# ============================================================
# ADD RESULT
# ============================================================

def add_result(
    results,
    row,
    filename,
    method,
    mask,
    ground_truth
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
print("MULTI-CUE BLEMISH CONFIDENCE SCORE")
print("==========================================")

print(
    f"Ground-truth images: {len(df)}"
)


# ============================================================
# PROCESS
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
    # BASELINE
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
    # CURRENT CHAMPION
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
    # MC-BCS CONFIGURATIONS
    # ========================================================

    for (
        config_name,
        weights
    ) in CONFIGURATIONS.items():

        mask = multicue_bcs(
            image,
            roi_mask,
            weights
        )


        add_result(
            results,
            row,
            filename,
            config_name,
            mask,
            ground_truth
        )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


detailed_path = os.path.join(
    OUTPUT_ROOT,
    "multicue_bcs_detailed.csv"
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
    .groupby("method")[
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


# Primary ranking = IoU
summary_df = summary_df.sort_values(
    by=[
        "iou",
        "dice",
        "overall_score"
    ],
    ascending=False
)


summary_path = os.path.join(
    OUTPUT_ROOT,
    "multicue_bcs_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n\n==========================================")
print("MC-BCS ENHANCEMENT SUMMARY")
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
# BEST
# ============================================================

best = summary_df.iloc[0]


print("\n==========================================")
print("BEST MC-BCS EXPERIMENT RESULT")
print("==========================================")

print(
    f"Method          : {best['method']}"
)

print(
    f"Mean IoU        : {best['iou']:.4f}"
)

print(
    f"Mean Dice       : {best['dice']:.4f}"
)

print(
    f"Mean Precision  : {best['precision']:.4f}"
)

print(
    f"Mean Recall     : {best['recall']:.4f}"
)

print(
    f"Overall Score   : "
    f"{best['overall_score']:.4f}"
)

print("==========================================")


print(
    f"\nDetailed results:\n{detailed_path}"
)

print(
    f"\nSummary:\n{summary_path}"
)