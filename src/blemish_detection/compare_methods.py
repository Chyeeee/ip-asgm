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

MASK_ROOT = (
    "results/preprocessing/MedianFinal/ROIMasks"
)

OUTPUT_ROOT = "results/blemish_detection/method_comparison"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)


# ============================================================
# METHOD 1 — OTSU
# ============================================================

def otsu_segmentation(image, roi_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Only fruit pixels
    fruit_pixels = gray[roi_mask > 0]

    if len(fruit_pixels) == 0:
        return np.zeros_like(gray)

    # Calculate Otsu threshold using fruit ROI
    threshold, _ = cv2.threshold(
        fruit_pixels,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Darker pixels considered possible blemishes
    blemish_mask = np.zeros_like(gray)

    blemish_mask[
        (gray < threshold) &
        (roi_mask > 0)
    ] = 255

    return blemish_mask


# ============================================================
# METHOD 2 — ADAPTIVE THRESHOLDING
# ============================================================

def adaptive_segmentation(image, roi_mask):

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

    # Keep only fruit region
    adaptive[roi_mask == 0] = 0

    return adaptive


# ============================================================
# METHOD 3 — HSV COLOUR-BASED SEGMENTATION
# ============================================================

def hsv_segmentation(image, roi_mask):

    # Convert processed image from BGR to HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    # Extract only fruit pixels
    fruit_pixels = hsv[roi_mask > 0]

    if len(fruit_pixels) == 0:
        return np.zeros(
            roi_mask.shape,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # FIND REPRESENTATIVE FRUIT COLOUR
    # --------------------------------------------------------
    # Median is more robust than mean when blemishes exist.

    median_h = np.median(fruit_pixels[:, 0])
    median_s = np.median(fruit_pixels[:, 1])
    median_v = np.median(fruit_pixels[:, 2])


    # --------------------------------------------------------
    # HSV DIFFERENCE
    # --------------------------------------------------------

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Hue is circular in OpenCV:
    # 0 and 179 represent very similar colours.
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


    # --------------------------------------------------------
    # NORMALISE HSV DIFFERENCES
    # --------------------------------------------------------

    hue_difference = (
        hue_difference / 90.0
    )

    saturation_difference = (
        saturation_difference / 255.0
    )

    value_difference = (
        value_difference / 255.0
    )


    # --------------------------------------------------------
    # COMBINED HSV COLOUR DIFFERENCE
    # --------------------------------------------------------

    hsv_difference = np.sqrt(
        hue_difference ** 2
        + saturation_difference ** 2
        + value_difference ** 2
    )


    # --------------------------------------------------------
    # AUTOMATIC THRESHOLD FROM FRUIT ROI
    # --------------------------------------------------------

    roi_difference = hsv_difference[
        roi_mask > 0
    ]

    threshold = (
        np.mean(roi_difference)
        + np.std(roi_difference)
    )


    # --------------------------------------------------------
    # CREATE BLEMISH MASK
    # --------------------------------------------------------

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
# DAMAGE PERCENTAGE
# ============================================================

def calculate_damage(mask, roi_mask):

    fruit_pixels = np.count_nonzero(
        roi_mask
    )

    blemish_pixels = np.count_nonzero(
        mask
    )

    if fruit_pixels == 0:
        return 0.0

    return (
        blemish_pixels
        / fruit_pixels
    ) * 100


# ============================================================
# VISUALISATION
# ============================================================

def create_visual(
    image,
    roi,
    otsu,
    adaptive,
    hsv,
    fruit,
    category,
    image_name
):

    # --------------------------------------------------------
    # CONVERT MASKS TO BGR
    # --------------------------------------------------------

    roi_display = cv2.cvtColor(
        roi,
        cv2.COLOR_GRAY2BGR
    )

    otsu_display = cv2.cvtColor(
        otsu,
        cv2.COLOR_GRAY2BGR
    )

    adaptive_display = cv2.cvtColor(
        adaptive,
        cv2.COLOR_GRAY2BGR
    )

    hsv_display = cv2.cvtColor(
        hsv,
        cv2.COLOR_GRAY2BGR
    )


    # --------------------------------------------------------
    # SETTINGS FOR LABELS
    # --------------------------------------------------------

    label_height = 55

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_thickness = 2

    text_colour = (0, 0, 0)
    background_colour = (255, 255, 255)


    # --------------------------------------------------------
    # FUNCTION TO ADD TITLE ABOVE AN IMAGE
    # --------------------------------------------------------

    def add_label(img, label):

        height, width = img.shape[:2]

        label_area = np.full(
            (label_height, width, 3),
            background_colour,
            dtype=np.uint8
        )

        text_size = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness
        )[0]

        text_x = max(
            10,
            (width - text_size[0]) // 2
        )

        text_y = (
            label_height + text_size[1]
        ) // 2

        cv2.putText(
            label_area,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_colour,
            font_thickness,
            cv2.LINE_AA
        )

        return np.vstack([
            label_area,
            img
        ])


    # --------------------------------------------------------
    # ADD METHOD NAMES
    # --------------------------------------------------------

    image_panel = add_label(
        image,
        "Processed Image"
    )

    roi_panel = add_label(
        roi_display,
        "Fruit ROI Mask"
    )

    otsu_panel = add_label(
        otsu_display,
        "Method 1 - Otsu Thresholding"
    )

    adaptive_panel = add_label(
        adaptive_display,
        "Method 2 - Adaptive Thresholding"
    )

    hsv_panel = add_label(
        hsv_display,
        "Method 3 - HSV Colour Segmentation"
    )


    # --------------------------------------------------------
    # CREATE BLANK PANEL
    # --------------------------------------------------------

    blank = np.zeros_like(
        hsv_display
    )

    blank_panel = add_label(
        blank,
        ""
    )


    # --------------------------------------------------------
    # CREATE ROWS
    # --------------------------------------------------------

    top = np.hstack([
        image_panel,
        roi_panel
    ])

    middle = np.hstack([
        otsu_panel,
        adaptive_panel
    ])

    bottom = np.hstack([
        hsv_panel,
        blank_panel
    ])


    # --------------------------------------------------------
    # COMBINE ALL ROWS
    # --------------------------------------------------------

    comparison = np.vstack([
        top,
        middle,
        bottom
    ])


    # --------------------------------------------------------
    # ADD MAIN TITLE
    # --------------------------------------------------------

    title_height = 65

    title_area = np.full(
        (
            title_height,
            comparison.shape[1],
            3
        ),
        255,
        dtype=np.uint8
    )

    main_title = (
        f"{fruit} | {category} | {image_name}"
    )

    title_font_scale = 0.9
    title_thickness = 2

    title_size = cv2.getTextSize(
        main_title,
        font,
        title_font_scale,
        title_thickness
    )[0]

    title_x = max(
        10,
        (
            comparison.shape[1]
            - title_size[0]
        ) // 2
    )

    title_y = (
        title_height
        + title_size[1]
    ) // 2

    cv2.putText(
        title_area,
        main_title,
        (title_x, title_y),
        font,
        title_font_scale,
        (0, 0, 0),
        title_thickness,
        cv2.LINE_AA
    )

    comparison = np.vstack([
        title_area,
        comparison
    ])


    # --------------------------------------------------------
    # OUTPUT FOLDER
    # --------------------------------------------------------

    output_folder = os.path.join(
        OUTPUT_ROOT,
        fruit,
        category
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    output_path = os.path.join(
        output_folder,
        image_name
    )

    cv2.imwrite(
        output_path,
        comparison
    )


# ============================================================
# TEST SMALL SAMPLE
# ============================================================

# 3 images from every fruit
# Select samples from every fruit AND category
sample_list = []

# Select 2 random images from each fruit + category combination
for (fruit, category), group in df.groupby(["fruit", "category"]):

    n_samples = min(2, len(group))

    selected = group.sample(
        n=n_samples,
        random_state=42
    )

    sample_list.append(selected)

# Combine all selected samples
sample_df = pd.concat(
    sample_list,
    ignore_index=True
)


print("\n==========================================")
print("MEMBER 4 - METHOD COMPARISON TEST")
print("==========================================")

print(
    f"Testing {len(sample_df)} images..."
)


for _, row in sample_df.iterrows():

    relative_path = row["relative_path"]

    image_path = os.path.join(
        PROCESSED_ROOT,
        relative_path
    )

    relative_folder = os.path.dirname(
        relative_path
    )

    filename = os.path.basename(
        relative_path
    )

    filename_without_extension = os.path.splitext(
        filename
    )[0]

    mask_path = os.path.join(
        MASK_ROOT,
        relative_folder,
        filename_without_extension + "_mask.png"
    )

    image = cv2.imread(
        image_path
    )

    roi_mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or roi_mask is None:

        print(
            f"SKIPPED: {relative_path}"
        )

        continue


    # --------------------------------------------------------
    # RUN THREE METHODS
    # --------------------------------------------------------

    otsu_mask = otsu_segmentation(
        image,
        roi_mask
    )

    adaptive_mask = adaptive_segmentation(
        image,
        roi_mask
    )

    hsv_mask = hsv_segmentation(
        image,
        roi_mask
    )


    # --------------------------------------------------------
    # DAMAGE %
    # --------------------------------------------------------

    otsu_percentage = calculate_damage(
        otsu_mask,
        roi_mask
    )

    adaptive_percentage = calculate_damage(
        adaptive_mask,
        roi_mask
    )

    hsv_percentage = calculate_damage(
        hsv_mask,
        roi_mask
    )


    print(
        f"\n{row['fruit']} | "
        f"{row['category']} | "
        f"{filename}"
    )

    print(
        f"  Otsu     : "
        f"{otsu_percentage:.2f}%"
    )

    print(
        f"  Adaptive : "
        f"{adaptive_percentage:.2f}%"
    )

    print(
        f"  HSV      : "
        f"{hsv_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE VISUAL COMPARISON
    # --------------------------------------------------------

    create_visual(
        image,
        roi_mask,
        otsu_mask,
        adaptive_mask,
        hsv_mask,
        row["fruit"],
        row["category"],
        filename_without_extension + ".jpg"
    )


print("\n==========================================")
print("TEST COMPLETED")
print("==========================================")

print(
    f"Results saved to: {OUTPUT_ROOT}"
)