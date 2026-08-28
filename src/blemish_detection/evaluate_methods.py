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
    "results/blemish_detection/evaluation"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# METHOD 1 — OTSU THRESHOLDING
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
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    blemish_mask = np.zeros_like(
        gray
    )

    blemish_mask[
        (gray < threshold)
        & (roi_mask > 0)
    ] = 255

    return blemish_mask


# ============================================================
# METHOD 2 — ADAPTIVE THRESHOLDING
# ============================================================

def adaptive_segmentation(
    image,
    roi_mask
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    adaptive[
        roi_mask == 0
    ] = 0

    return adaptive


# ============================================================
# METHOD 3 — HSV COLOUR SEGMENTATION
# ============================================================

def hsv_segmentation(
    image,
    roi_mask
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    fruit_pixels = hsv[
        roi_mask > 0
    ]

    if len(fruit_pixels) == 0:

        return np.zeros(
            roi_mask.shape,
            dtype=np.uint8
        )

    # Representative fruit colour
    median_h = np.median(
        fruit_pixels[:, 0]
    )

    median_s = np.median(
        fruit_pixels[:, 1]
    )

    median_v = np.median(
        fruit_pixels[:, 2]
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Hue is circular in OpenCV
    hue_difference = np.abs(
        h - median_h
    )

    hue_difference = np.minimum(
        hue_difference,
        180 - hue_difference
    )

    saturation_difference = np.abs(
        s - median_s
    )

    value_difference = np.abs(
        v - median_v
    )

    # Normalise
    hue_difference = (
        hue_difference / 90.0
    )

    saturation_difference = (
        saturation_difference / 255.0
    )

    value_difference = (
        value_difference / 255.0
    )

    # Combined HSV difference
    hsv_difference = np.sqrt(
        hue_difference ** 2
        + saturation_difference ** 2
        + value_difference ** 2
    )

    roi_difference = hsv_difference[
        roi_mask > 0
    ]

    threshold = (
        np.mean(roi_difference)
        + np.std(roi_difference)
    )

    blemish_mask = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )

    blemish_mask[
        (hsv_difference > threshold)
        & (roi_mask > 0)
    ] = 255

    return blemish_mask


# ============================================================
# CALCULATE METRICS
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

    denominator_iou = (
        tp + fp + fn
    )

    if denominator_iou == 0:
        iou = 1.0
    else:
        iou = (
            tp / denominator_iou
        )


    # --------------------------------------------------------
    # DICE
    # --------------------------------------------------------

    denominator_dice = (
        2 * tp + fp + fn
    )

    if denominator_dice == 0:
        dice = 1.0
    else:
        dice = (
            2 * tp
            / denominator_dice
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
# LOAD EVALUATION DATA
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

results = []


print("\n==========================================")
print("MEMBER 4 - SEGMENTATION EVALUATION")
print("==========================================")

print(
    f"Ground-truth images: {len(df)}"
)


# ============================================================
# PROCESS EVERY GROUND-TRUTH IMAGE
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

    ground_truth_path = os.path.join(
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
        ground_truth_path,
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


    # Make sure GT stays inside ROI
    ground_truth[
        roi_mask == 0
    ] = 0


    # --------------------------------------------------------
    # RUN METHODS
    # --------------------------------------------------------

    method_masks = {

        "Otsu":
            otsu_segmentation(
                image,
                roi_mask
            ),

        "Adaptive":
            adaptive_segmentation(
                image,
                roi_mask
            ),

        "HSV":
            hsv_segmentation(
                image,
                roi_mask
            )
    }


    print(
        f"\n[{index + 1}/{len(df)}] "
        f"{row['fruit']} | "
        f"{row['category']} | "
        f"{filename}"
    )


    # --------------------------------------------------------
    # EVALUATE METHODS
    # --------------------------------------------------------

    for method_name, predicted_mask in (
        method_masks.items()
    ):

        (
            iou,
            dice,
            precision,
            recall
        ) = calculate_metrics(
            predicted_mask,
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
                method_name,

            "iou":
                iou,

            "dice":
                dice,

            "precision":
                precision,

            "recall":
                recall

        })


        print(
            f"  {method_name:<9} "
            f"IoU={iou:.4f} | "
            f"Dice={dice:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f}"
        )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

detailed_path = os.path.join(
    OUTPUT_ROOT,
    "segmentation_metrics_detailed.csv"
)

results_df.to_csv(
    detailed_path,
    index=False
)


# ============================================================
# CALCULATE MEAN PERFORMANCE
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


# ============================================================
# ADD OVERALL SCORE
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


# Sort best first
summary_df = summary_df.sort_values(
    by="overall_score",
    ascending=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_ROOT,
    "segmentation_metrics_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n==========================================")
print("AVERAGE SEGMENTATION PERFORMANCE")
print("==========================================")

for _, row in summary_df.iterrows():

    print(
        f"\n{row['method']}"
    )

    print(
        f"  Mean IoU       : "
        f"{row['iou']:.4f}"
    )

    print(
        f"  Mean Dice      : "
        f"{row['dice']:.4f}"
    )

    print(
        f"  Mean Precision : "
        f"{row['precision']:.4f}"
    )

    print(
        f"  Mean Recall    : "
        f"{row['recall']:.4f}"
    )

    print(
        f"  Overall Score  : "
        f"{row['overall_score']:.4f}"
    )


# ============================================================
# BEST METHOD
# ============================================================

best_method = summary_df.iloc[0]

print("\n==========================================")
print("BEST BASELINE METHOD")
print("==========================================")

print(
    f"Method        : "
    f"{best_method['method']}"
)

print(
    f"Overall Score : "
    f"{best_method['overall_score']:.4f}"
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