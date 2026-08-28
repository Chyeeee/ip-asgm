import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "results/texture_analysis/colour_texture_features.csv"

PROCESSED_ROOT = (
    "results/preprocessing/MedianFinal/ProcessedImages"
)

ROI_ROOT = (
    "results/preprocessing/MedianFinal/ROIMasks"
)

OUTPUT_ROOT = (
    "results/blemish_detection/final_analysis"
)

MASK_OUTPUT_ROOT = os.path.join(
    OUTPUT_ROOT,
    "BlemishMasks"
)

VISUAL_OUTPUT_ROOT = os.path.join(
    OUTPUT_ROOT,
    "Visualisations"
)

FINAL_CSV_PATH = os.path.join(
    OUTPUT_ROOT,
    "final_features_with_damage.csv"
)

SUMMARY_CSV_PATH = os.path.join(
    OUTPUT_ROOT,
    "damage_summary.csv"
)

os.makedirs(
    MASK_OUTPUT_ROOT,
    exist_ok=True
)

os.makedirs(
    VISUAL_OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# SELECTED METHOD
# OTSU + MORPHOLOGY 7x7
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
# MORPHOLOGICAL REFINEMENT
# ============================================================

def apply_morphology(mask, roi_mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    # Remove isolated noise
    refined = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps inside blemish regions
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Ensure result remains inside fruit ROI
    refined[
        roi_mask == 0
    ] = 0

    return refined


# ============================================================
# DAMAGE QUANTIFICATION
# ============================================================

def calculate_damage(
    blemish_mask,
    roi_mask
):

    fruit_pixels = np.count_nonzero(
        roi_mask > 0
    )

    blemish_pixels = np.count_nonzero(
        blemish_mask > 0
    )

    if fruit_pixels == 0:

        return (
            0,
            0,
            0.0
        )

    damage_percentage = (
        blemish_pixels
        / fruit_pixels
    ) * 100.0

    return (
        fruit_pixels,
        blemish_pixels,
        damage_percentage
    )



# ============================================================
# AWDP_A DAMAGE QUANTIFICATION
# Final Member 4 enhancement:
# 0.50 Intensity + 0.30 Colour + 0.20 Texture
# ============================================================

def awdp_intensity_abnormality(image, roi_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    roi_values = gray[roi_mask > 0]

    if roi_values.size == 0:
        return np.zeros_like(gray, dtype=np.float32)

    median_intensity = np.median(roi_values)

    score = (
        median_intensity - gray
    ) / max(median_intensity, 1.0)

    score = np.clip(score, 0.0, 1.0)
    score[roi_mask == 0] = 0.0

    return score


def awdp_colour_abnormality(image, roi_mask):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    roi_pixels = lab[roi_mask > 0]

    if roi_pixels.size == 0:
        return np.zeros(
            roi_mask.shape,
            dtype=np.float32
        )

    reference_colour = np.median(
        roi_pixels,
        axis=0
    )

    difference = lab - reference_colour

    distance = np.sqrt(
        np.sum(
            difference ** 2,
            axis=2
        )
    )

    roi_distance = distance[roi_mask > 0]

    scale = np.percentile(
        roi_distance,
        95
    )

    if scale <= 0:
        scale = 1.0

    score = np.clip(
        distance / scale,
        0.0,
        1.0
    )

    score[roi_mask == 0] = 0.0

    return score


def awdp_texture_abnormality(image, roi_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    window = 11

    local_mean = cv2.blur(
        gray,
        (window, window)
    )

    local_mean_square = cv2.blur(
        gray ** 2,
        (window, window)
    )

    variance = (
        local_mean_square
        - local_mean ** 2
    )

    variance = np.maximum(
        variance,
        0
    )

    local_std = np.sqrt(variance)

    roi_std = local_std[roi_mask > 0]

    if roi_std.size == 0:
        return np.zeros_like(
            gray,
            dtype=np.float32
        )

    scale = np.percentile(
        roi_std,
        95
    )

    if scale <= 0:
        scale = 1.0

    score = np.clip(
        local_std / scale,
        0.0,
        1.0
    )

    score[roi_mask == 0] = 0.0

    return score


def calculate_awdp_damage(
    image,
    roi_mask,
    blemish_mask
):

    roi = (
        roi_mask > 0
    ).astype(np.uint8)

    blemish = (
        blemish_mask > 0
    ).astype(np.uint8)

    intensity = awdp_intensity_abnormality(
        image,
        roi
    )

    colour = awdp_colour_abnormality(
        image,
        roi
    )

    texture = awdp_texture_abnormality(
        image,
        roi
    )

    confidence = (
        0.50 * intensity
        + 0.30 * colour
        + 0.20 * texture
    )

    confidence = np.clip(
        confidence,
        0.0,
        1.0
    )

    weighted_blemish = (
        blemish.astype(np.float32)
        * confidence
    )

    weighted_blemish[roi == 0] = 0.0

    fruit_area = np.count_nonzero(roi)

    if fruit_area == 0:
        return 0.0

    damage_percentage = (
        np.sum(weighted_blemish)
        / fruit_area
    ) * 100.0

    return float(damage_percentage)


# ============================================================
# OPTIONAL DAMAGE LEVEL
# ============================================================

def get_damage_level(
    damage_percentage
):

    if damage_percentage < 10:

        return "Low"

    elif damage_percentage < 30:

        return "Moderate"

    elif damage_percentage < 50:

        return "High"

    else:

        return "Severe"


# ============================================================
# VISUALISATION
# ============================================================

def create_visualisation(
    image,
    roi_mask,
    blemish_mask,
    damage_percentage
):

    # Create overlay
    overlay = image.copy()

    # Highlight detected blemish pixels
    overlay[
        blemish_mask > 0
    ] = (
        0,
        0,
        255
    )

    result = cv2.addWeighted(
        image,
        0.65,
        overlay,
        0.35,
        0
    )

    # Add damage percentage
    text = (
        f"Damage: "
        f"{damage_percentage:.2f}%"
    )

    cv2.putText(
        result,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return result


# ============================================================
# LOAD MEMBER 3 FEATURES
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

print("\n==========================================")
print("FINAL BLEMISH DAMAGE QUANTIFICATION")
print("==========================================")

print(
    f"Total images: {len(df)}"
)

print(
    "Selected method: "
    "Otsu + Morphology 7x7"
)

print("==========================================\n")


# ============================================================
# NEW OUTPUT COLUMNS
# ============================================================

fruit_pixel_list = []
blemish_pixel_list = []
raw_damage_percentage_list = []
damage_percentage_list = []
damage_level_list = []
blemish_mask_path_list = []


processed_count = 0
skipped_count = 0


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

for index, row in df.iterrows():

    relative_path = row[
        "relative_path"
    ]

    fruit = row[
        "fruit"
    ]

    category = row[
        "category"
    ]


    # --------------------------------------------------------
    # PATH INFORMATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # READ IMAGE + ROI
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    roi_mask = cv2.imread(
        roi_path,
        cv2.IMREAD_GRAYSCALE
    )


    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if (
        image is None
        or roi_mask is None
    ):

        print(
            f"SKIPPED: {relative_path}"
        )

        fruit_pixel_list.append(
            np.nan
        )

        blemish_pixel_list.append(
            np.nan
        )

        raw_damage_percentage_list.append(
            np.nan
        )

        damage_percentage_list.append(
            np.nan
        )

        damage_level_list.append(
            "Missing"
        )

        blemish_mask_path_list.append(
            ""
        )

        skipped_count += 1

        continue


    # --------------------------------------------------------
    # STEP 1 — OTSU
    # --------------------------------------------------------

    otsu_mask = otsu_segmentation(
        image,
        roi_mask
    )


    # --------------------------------------------------------
    # STEP 2 — MORPHOLOGY 7x7
    # --------------------------------------------------------

    final_mask = apply_morphology(
        otsu_mask,
        roi_mask
    )


    # --------------------------------------------------------
    # STEP 3 — DAMAGE QUANTIFICATION
    # --------------------------------------------------------

    (
        fruit_pixels,
        blemish_pixels,
        raw_damage_percentage
    ) = calculate_damage(
        final_mask,
        roi_mask
    )

    # Final damage feature used for ML training.
    # This MUST match prediction_pipeline.py.
    damage_percentage = calculate_awdp_damage(
        image,
        roi_mask,
        final_mask
    )

    damage_level = get_damage_level(
        damage_percentage
    )


    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    mask_folder = os.path.join(
        MASK_OUTPUT_ROOT,
        fruit,
        category
    )

    os.makedirs(
        mask_folder,
        exist_ok=True
    )


    mask_filename = (
        image_name
        + "_blemish.png"
    )

    mask_output_path = os.path.join(
        mask_folder,
        mask_filename
    )


    cv2.imwrite(
        mask_output_path,
        final_mask
    )


    # --------------------------------------------------------
    # SAVE SOME VISUALISATIONS
    #
    # Saving all 3280 visualisations is unnecessary.
    # Save first 3 images from each fruit/category.
    # --------------------------------------------------------

    category_count = (
        df.iloc[:index + 1]
        [
            (df.iloc[:index + 1]["fruit"] == fruit)
            &
            (
                df.iloc[:index + 1]["category"]
                == category
            )
        ]
        .shape[0]
    )


    if category_count <= 3:

        visual_folder = os.path.join(
            VISUAL_OUTPUT_ROOT,
            fruit,
            category
        )

        os.makedirs(
            visual_folder,
            exist_ok=True
        )


        visual = create_visualisation(
            image,
            roi_mask,
            final_mask,
            damage_percentage
        )


        visual_path = os.path.join(
            visual_folder,
            image_name
            + "_result.jpg"
        )


        cv2.imwrite(
            visual_path,
            visual
        )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    fruit_pixel_list.append(
        fruit_pixels
    )

    blemish_pixel_list.append(
        blemish_pixels
    )

    raw_damage_percentage_list.append(
        raw_damage_percentage
    )

    damage_percentage_list.append(
        damage_percentage
    )

    damage_level_list.append(
        damage_level
    )

    blemish_mask_path_list.append(
        mask_output_path
    )


    processed_count += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        processed_count % 100 == 0
        or processed_count == len(df)
    ):

        print(
            f"Processed "
            f"{processed_count}/"
            f"{len(df)} images"
        )


# ============================================================
# ADD MEMBER 4 FEATURES
# ============================================================

df[
    "fruit_pixels"
] = fruit_pixel_list

df[
    "blemish_pixels"
] = blemish_pixel_list

df[
    "raw_damage_percentage"
] = raw_damage_percentage_list

df[
    "damage_percentage"
] = damage_percentage_list

df[
    "damage_level"
] = damage_level_list

df[
    "blemish_mask_path"
] = blemish_mask_path_list


# ============================================================
# SAVE FINAL INTEGRATED DATASET
# ============================================================

df.to_csv(
    FINAL_CSV_PATH,
    index=False
)


# ============================================================
# CREATE SUMMARY
# ============================================================

valid_df = df.dropna(
    subset=[
        "damage_percentage"
    ]
)


summary_df = (
    valid_df
    .groupby(
        [
            "fruit",
            "category"
        ]
    )
    .agg(

        image_count=(
            "damage_percentage",
            "count"
        ),

        mean_damage_percentage=(
            "damage_percentage",
            "mean"
        ),

        std_damage_percentage=(
            "damage_percentage",
            "std"
        ),

        min_damage_percentage=(
            "damage_percentage",
            "min"
        ),

        max_damage_percentage=(
            "damage_percentage",
            "max"
        )

    )
    .reset_index()
)


summary_df.to_csv(
    SUMMARY_CSV_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("FINAL DAMAGE QUANTIFICATION COMPLETED")
print("==========================================")

print(
    f"Successfully processed : "
    f"{processed_count}"
)

print(
    f"Skipped                : "
    f"{skipped_count}"
)

print(
    f"Total                   : "
    f"{len(df)}"
)

print(
    "\nSelected segmentation method:"
)

print(
    "Otsu Thresholding "
    "+ Morphology 7x7"
)

print(
    "\nDamage calculation:"
)

print(
    "Raw Damage % = "
    "(Blemish Pixels / Fruit Pixels) x 100"
)

print(
    "Final Damage % = AWDP_A: "
    "0.50(Intensity) + 0.30(Colour) + 0.20(Texture), "
    "weighted within Otsu + Morphology blemish candidates"
)

print(
    f"\nFinal integrated CSV:\n"
    f"{FINAL_CSV_PATH}"
)

print(
    f"\nDamage summary:\n"
    f"{SUMMARY_CSV_PATH}"
)

print(
    f"\nBlemish masks:\n"
    f"{MASK_OUTPUT_ROOT}"
)

print(
    f"\nVisualisations:\n"
    f"{VISUAL_OUTPUT_ROOT}"
)

print("==========================================")