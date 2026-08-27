import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import csv


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = "Dataset"
RESULT_FOLDER = "Results/NonLocalMeans"

IMAGE_SIZE = (600, 600)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)

os.makedirs(RESULT_FOLDER, exist_ok=True)


# ============================================================
# 1. AUTOMATICALLY FIND ONE IMAGE
# ============================================================

def find_first_image(folder):

    image_paths = []

    for root, folders, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(IMAGE_EXTENSIONS):

                full_path = os.path.join(
                    root,
                    filename
                )

                image_paths.append(full_path)

    # Sort so all filters use the SAME image
    image_paths.sort()

    if len(image_paths) == 0:
        return None

    return image_paths[0]


# ============================================================
# 2. NON-LOCAL MEANS FILTER
# ============================================================

def nlm_filter(image):

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )


# ============================================================
# 3. NOISE REDUCTION
# ============================================================

def calculate_noise_reduction(original, filtered):

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    filtered_gray = cv2.cvtColor(
        filtered,
        cv2.COLOR_BGR2GRAY
    )

    original_noise = cv2.Laplacian(
        original_gray,
        cv2.CV_64F
    ).var()

    filtered_noise = cv2.Laplacian(
        filtered_gray,
        cv2.CV_64F
    ).var()

    if original_noise == 0:
        return 0.0

    reduction = (
        (original_noise - filtered_noise)
        / original_noise
    ) * 100

    reduction = max(
        0,
        min(100, reduction)
    )

    return reduction


# ============================================================
# 4. EDGE PRESERVATION
# ============================================================

def calculate_edge_preservation(original, filtered):

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    filtered_gray = cv2.cvtColor(
        filtered,
        cv2.COLOR_BGR2GRAY
    )

    original_edges = cv2.Canny(
        original_gray,
        100,
        200
    )

    filtered_edges = cv2.Canny(
        filtered_gray,
        100,
        200
    )

    # Allow slight edge movement after filtering
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    filtered_dilated = cv2.dilate(
        filtered_edges,
        kernel,
        iterations=1
    )

    original_edge_pixels = (
        original_edges > 0
    )

    if np.sum(original_edge_pixels) == 0:
        return 0.0

    preserved = (
        original_edge_pixels
        & (filtered_dilated > 0)
    )

    score = (
        np.sum(preserved)
        / np.sum(original_edge_pixels)
    ) * 100

    return score


# ============================================================
# 5. AUTOMATIC ROI SEGMENTATION
# ============================================================

def create_roi_mask(image):

    height, width = image.shape[:2]

    mask = np.zeros(
        (height, width),
        np.uint8
    )

    background_model = np.zeros(
        (1, 65),
        np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        np.float64
    )

    # Create GrabCut rectangle
    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)

    rectangle = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    # GrabCut segmentation
    cv2.grabCut(
        image,
        mask,
        rectangle,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Convert GrabCut result to binary mask
    binary_mask = np.where(
        (mask == cv2.GC_FGD)
        | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")


    # ========================================================
    # MORPHOLOGICAL CLEANING
    # ========================================================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # KEEP LARGEST CONTOUR
    # ========================================================

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(
            binary_mask
        )

        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        binary_mask = clean_mask

    return binary_mask


# ============================================================
# 6. ROI SEGMENTATION SCORE
# ============================================================

def calculate_roi_score(filtered, roi_mask):

    # Detect boundary of ROI mask
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    boundary = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_GRADIENT,
        kernel
    )

    # Detect image edges
    gray = cv2.cvtColor(
        filtered,
        cv2.COLOR_BGR2GRAY
    )

    image_edges = cv2.Canny(
        gray,
        100,
        200
    )

    # Dilate image edges slightly
    edge_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    image_edges = cv2.dilate(
        image_edges,
        edge_kernel,
        iterations=1
    )

    boundary_pixels = (
        boundary > 0
    )

    if np.sum(boundary_pixels) == 0:
        return 0.0

    matching_pixels = (
        boundary_pixels
        & (image_edges > 0)
    )

    score = (
        np.sum(matching_pixels)
        / np.sum(boundary_pixels)
    ) * 100

    return score


# ============================================================
# 7. FIND SAME TEST IMAGE
# ============================================================

image_path = find_first_image(
    DATASET_FOLDER
)

if image_path is None:

    print("ERROR: No image found in Dataset folder.")
    exit()

print("\nImage selected:")
print(image_path)


# ============================================================
# 8. LOAD AND RESIZE IMAGE
# ============================================================

image = cv2.imread(
    image_path
)

if image is None:

    print("ERROR: Image cannot be loaded.")
    exit()

resized = cv2.resize(
    image,
    IMAGE_SIZE
)


# ============================================================
# 9. PROCESSING TIME
# ============================================================

# Warm-up
nlm_filter(resized)

number_of_runs = 30

start_time = time.perf_counter()

for _ in range(number_of_runs):

    nlm = nlm_filter(
        resized
    )

end_time = time.perf_counter()

processing_time = (
    (end_time - start_time)
    / number_of_runs
) * 1000


# ============================================================
# 10. CALCULATE RESULTS
# ============================================================

noise_reduction = calculate_noise_reduction(
    resized,
    nlm
)

edge_preservation = calculate_edge_preservation(
    resized,
    nlm
)

roi_mask = create_roi_mask(
    nlm
)

roi_score = calculate_roi_score(
    nlm,
    roi_mask
)


# ============================================================
# 11. CREATE SEGMENTED FRUIT ROI
# ============================================================

segmented_image = cv2.bitwise_and(
    resized,
    resized,
    mask=roi_mask
)


# ============================================================
# 12. SAVE INDIVIDUAL IMAGES
# ============================================================

cv2.imwrite(
    os.path.join(
        RESULT_FOLDER,
        "original.jpg"
    ),
    resized
)

cv2.imwrite(
    os.path.join(
        RESULT_FOLDER,
        "nlm_filtered.jpg"
    ),
    nlm
)

cv2.imwrite(
    os.path.join(
        RESULT_FOLDER,
        "nlm_roi_mask.png"
    ),
    roi_mask
)

cv2.imwrite(
    os.path.join(
        RESULT_FOLDER,
        "nlm_segmented_roi.jpg"
    ),
    segmented_image
)


# ============================================================
# 13. CREATE COMPARISON IMAGE
# ============================================================

original_rgb = cv2.cvtColor(
    resized,
    cv2.COLOR_BGR2RGB
)

nlm_rgb = cv2.cvtColor(
    nlm,
    cv2.COLOR_BGR2RGB
)

segmented_rgb = cv2.cvtColor(
    segmented_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(
    figsize=(12, 10)
)


# Original
plt.subplot(2, 2, 1)

plt.imshow(
    original_rgb
)

plt.title(
    "Original Image"
)

plt.axis(
    "off"
)


# NLM Filtered
plt.subplot(2, 2, 2)

plt.imshow(
    nlm_rgb
)

plt.title(
    "Non-Local Means Filter"
)

plt.axis(
    "off"
)


# Binary ROI Mask
plt.subplot(2, 2, 3)

plt.imshow(
    roi_mask,
    cmap="gray"
)

plt.title(
    "Binary ROI Mask"
)

plt.axis(
    "off"
)


# Segmented Fruit ROI
plt.subplot(2, 2, 4)

plt.imshow(
    segmented_rgb
)

plt.title(
    "Segmented Fruit ROI"
)

plt.axis(
    "off"
)


plt.tight_layout()

comparison_path = os.path.join(
    RESULT_FOLDER,
    "nlm_result_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. SAVE RESULT TO CSV
# ============================================================

csv_path = os.path.join(
    RESULT_FOLDER,
    "nlm_result.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        "Filtering",
        "Noise Reduction (%)",
        "Edge Preservation (%)",
        "ROI Segmentation (%)",
        "Processing Time (ms)"
    ])

    writer.writerow([
        "Non-Local Means",
        round(noise_reduction, 2),
        round(edge_preservation, 2),
        round(roi_score, 2),
        round(processing_time, 4)
    ])


# ============================================================
# 15. DISPLAY FINAL RESULT
# ============================================================

print("\n========================================================")
print("           NON-LOCAL MEANS FILTER RESULT")
print("========================================================")

print(
    f"Noise Reduction     : "
    f"{noise_reduction:.2f} %"
)

print(
    f"Edge Preservation   : "
    f"{edge_preservation:.2f} %"
)

print(
    f"ROI Segmentation    : "
    f"{roi_score:.2f} %"
)

print(
    f"Processing Time     : "
    f"{processing_time:.4f} ms"
)

print("========================================================")

print("\nImages saved to:")
print(RESULT_FOLDER)

print("\nComparison image:")
print(comparison_path)

print("\nCSV result:")
print(csv_path)