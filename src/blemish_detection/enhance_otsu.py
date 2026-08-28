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
    "results/blemish_detection/otsu_enhancement"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# ORIGINAL OTSU SEGMENTATION
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

    blemish_mask = np.zeros_like(
        gray
    )

    blemish_mask[
        (gray < threshold)
        & (roi_mask > 0)
    ] = 255

    return blemish_mask


# ============================================================
# MORPHOLOGICAL ENHANCEMENT
# ============================================================

def enhance_mask(
    blemish_mask,
    roi_mask,
    kernel_size
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # --------------------------------------------------------
    # OPENING
    # Removes small isolated false-positive regions
    # --------------------------------------------------------

    enhanced = cv2.morphologyEx(
        blemish_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # CLOSING
    # Fills small gaps inside detected blemishes
    # --------------------------------------------------------

    enhanced = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep result inside fruit ROI
    enhanced[
        roi_mask == 0
    ] = 0

    return enhanced


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    predicted_mask,
    ground_truth_mask
):

    predicted = predicted_mask > 0
    ground_truth = ground_truth_mask > 0

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


    # IoU
    denominator_iou = (
        tp + fp + fn
    )

    if denominator_iou == 0:
        iou = 1.0
    else:
        iou = tp / denominator_iou


    # Dice
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


    # Precision
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


    # Recall
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
# ADD LABEL TO IMAGE
# ============================================================

def add_label(
    image,
    label
):

    display = image.copy()

    if len(display.shape) == 2:

        display = cv2.cvtColor(
            display,
            cv2.COLOR_GRAY2BGR
        )

    label_height = 45

    canvas = np.zeros(
        (
            display.shape[0] + label_height,
            display.shape[1],
            3
        ),
        dtype=np.uint8
    )

    canvas[
        label_height:,
        :
    ] = display

    cv2.putText(
        canvas,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# CREATE VISUAL COMPARISON
# ============================================================

def create_visual(
    image,
    ground_truth,
    original,
    enhanced_3,
    enhanced_5,
    enhanced_7,
    output_path
):

    original_image = add_label(
        image,
        "Original Image"
    )

    gt_image = add_label(
        ground_truth,
        "Ground Truth"
    )

    original_otsu = add_label(
        original,
        "Original Otsu"
    )

    morphology_3 = add_label(
        enhanced_3,
        "Otsu + Morphology 3x3"
    )

    morphology_5 = add_label(
        enhanced_5,
        "Otsu + Morphology 5x5"
    )

    morphology_7 = add_label(
        enhanced_7,
        "Otsu + Morphology 7x7"
    )

    row1 = np.hstack([
        original_image,
        gt_image
    ])

    row2 = np.hstack([
        original_otsu,
        morphology_3
    ])

    row3 = np.hstack([
        morphology_5,
        morphology_7
    ])

    comparison = np.vstack([
        row1,
        row2,
        row3
    ])

    cv2.imwrite(
        output_path,
        comparison
    )


# ============================================================
# LOAD GROUND-TRUTH SELECTION
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

results = []


print("\n==========================================")
print("MEMBER 4 - OTSU ENHANCEMENT TEST")
print("==========================================")

print(
    f"Ground-truth images: {len(df)}"
)


# ============================================================
# PROCESS IMAGES
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


    # ========================================================
    # PATHS
    # ========================================================

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


    # ========================================================
    # LOAD
    # ========================================================

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


    # ========================================================
    # ORIGINAL OTSU
    # ========================================================

    original_otsu = otsu_segmentation(
        image,
        roi_mask
    )


    # ========================================================
    # ENHANCEMENT TESTS
    # ========================================================

    otsu_3 = enhance_mask(
        original_otsu,
        roi_mask,
        3
    )

    otsu_5 = enhance_mask(
        original_otsu,
        roi_mask,
        5
    )

    otsu_7 = enhance_mask(
        original_otsu,
        roi_mask,
        7
    )


    methods = {

        "Original Otsu":
            original_otsu,

        "Otsu + Morphology 3x3":
            otsu_3,

        "Otsu + Morphology 5x5":
            otsu_5,

        "Otsu + Morphology 7x7":
            otsu_7
    }


    print(
        f"\n[{index + 1}/{len(df)}] "
        f"{row['fruit']} | "
        f"{row['category']} | "
        f"{filename}"
    )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    for method_name, mask in methods.items():

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
            f"  {method_name:<25} "
            f"IoU={iou:.4f} | "
            f"Dice={dice:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f}"
        )


    # ========================================================
    # SAVE VISUAL
    # ========================================================

    visual_folder = os.path.join(
        OUTPUT_ROOT,
        "visuals",
        row["fruit"]
    )

    os.makedirs(
        visual_folder,
        exist_ok=True
    )

    visual_path = os.path.join(
        visual_folder,
        image_name + ".jpg"
    )

    create_visual(
        image,
        ground_truth,
        original_otsu,
        otsu_3,
        otsu_5,
        otsu_7,
        visual_path
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

detailed_path = os.path.join(
    OUTPUT_ROOT,
    "otsu_enhancement_detailed.csv"
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
].mean(axis=1)


summary_df = summary_df.sort_values(
    by="overall_score",
    ascending=False
)


summary_path = os.path.join(
    OUTPUT_ROOT,
    "otsu_enhancement_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n==========================================")
print("OTSU ENHANCEMENT SUMMARY")
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
# BEST ENHANCEMENT
# ============================================================

best = summary_df.iloc[0]

print("\n==========================================")
print("BEST OTSU CONFIGURATION")
print("==========================================")

print(
    f"Method        : {best['method']}"
)

print(
    f"Overall Score : "
    f"{best['overall_score']:.4f}"
)

print("==========================================")

print(
    f"\nDetailed results:\n{detailed_path}"
)

print(
    f"\nSummary:\n{summary_path}"
)

print(
    f"\nVisual comparisons:\n"
    f"{os.path.join(OUTPUT_ROOT, 'visuals')}"
)