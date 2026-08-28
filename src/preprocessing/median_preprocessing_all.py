import cv2
import numpy as np
import os
import time


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = "BalancedDataset"

RESULT_FOLDER = "results/preprocessing/MedianFinal"

# Final processed image size
IMAGE_SIZE = (600, 600)

# Median filter selected from filter comparison
MEDIAN_KERNEL_SIZE = 5

# ROI segmentation size.
# Final version uses full 600x600 resolution for better mask quality.
SEGMENTATION_SIZE = (600, 600)

# Use 5 GrabCut iterations for more stable final ROI segmentation
GRABCUT_ITERATIONS = 5

# Progress report frequency
PROGRESS_INTERVAL = 50

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

PROCESSED_FOLDER = os.path.join(
    RESULT_FOLDER,
    "ProcessedImages"
)

MASK_FOLDER = os.path.join(
    RESULT_FOLDER,
    "ROIMasks"
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)

os.makedirs(
    MASK_FOLDER,
    exist_ok=True
)


# ============================================================
# 1. MEDIAN FILTER
# ============================================================

def median_filter(image):

    return cv2.medianBlur(
        image,
        MEDIAN_KERNEL_SIZE
    )


# ============================================================
# 2. FAST ROI SEGMENTATION
# ============================================================

def create_roi_mask(image):
    """
    Create a more robust single-fruit ROI mask.

    IMPORTANT:
    - The saved/feature-extraction image is still the Median-filtered image.
    - CLAHE is used ONLY to help ROI segmentation.
    - GrabCut remains the main foreground segmentation method.
    - The function is designed for one dominant fruit per image.
    """

    original_height, original_width = image.shape[:2]

    # --------------------------------------------------------
    # 1. SEGMENTATION-SCALE IMAGE
    # --------------------------------------------------------

    small_image = cv2.resize(
        image,
        SEGMENTATION_SIZE,
        interpolation=cv2.INTER_AREA
    )

    height, width = small_image.shape[:2]

    # --------------------------------------------------------
    # 2. SEGMENTATION-ONLY CLAHE
    # --------------------------------------------------------
    # Improve local contrast without replacing the Median-filtered
    # image used later for feature extraction.
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        small_image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_enhanced = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        (
            l_enhanced,
            a_channel,
            b_channel
        )
    )

    segmentation_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------------
    # 3. INITIAL GRABCUT MASK
    # --------------------------------------------------------
    # Border pixels are definite background.
    # A broad central ellipse is probable foreground.
    # This gives GrabCut more information than one huge rectangle.
    # --------------------------------------------------------

    grabcut_mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    border_x = max(
        5,
        int(width * 0.04)
    )

    border_y = max(
        5,
        int(height * 0.04)
    )

    grabcut_mask[
        :border_y,
        :
    ] = cv2.GC_BGD

    grabcut_mask[
        height - border_y:,
        :
    ] = cv2.GC_BGD

    grabcut_mask[
        :,
        :border_x
    ] = cv2.GC_BGD

    grabcut_mask[
        :,
        width - border_x:
    ] = cv2.GC_BGD

    centre = (
        width // 2,
        height // 2
    )

    axes = (
        max(1, int(width * 0.38)),
        max(1, int(height * 0.38))
    )

    probable_foreground = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.ellipse(
        probable_foreground,
        centre,
        axes,
        0,
        0,
        360,
        255,
        thickness=cv2.FILLED
    )

    grabcut_mask[
        probable_foreground > 0
    ] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # 4. COLOUR-BASED FOREGROUND SEED
    # --------------------------------------------------------
    # Highly saturated pixels are useful probable-foreground seeds
    # for many fruits, but are NOT treated as definite fruit.
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        segmentation_image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]

    saturation_values = saturation[
        probable_foreground > 0
    ]

    if saturation_values.size > 0:

        saturation_threshold = max(
            35,
            int(
                np.percentile(
                    saturation_values,
                    55
                )
            )
        )

        colour_seed = (
            (saturation >= saturation_threshold)
            & (probable_foreground > 0)
        )

        grabcut_mask[
            colour_seed
        ] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # 5. GRABCUT WITH MASK INITIALISATION
    # --------------------------------------------------------

    background_model = np.zeros(
        (1, 65),
        np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        segmentation_image,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_MASK
    )

    binary_mask = np.where(
        (
            grabcut_mask == cv2.GC_FGD
        )
        | (
            grabcut_mask == cv2.GC_PR_FGD
        ),
        255,
        0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 6. MORPHOLOGICAL CLEANING
    # --------------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    # --------------------------------------------------------
    # 7. COMPONENT VALIDATION
    # --------------------------------------------------------
    # Do NOT blindly keep the largest component.
    # Score components by:
    # - area,
    # - closeness to image centre,
    # - border contact.
    #
    # This reduces the chance of selecting a large background region.
    # --------------------------------------------------------

    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    image_area = float(
        height * width
    )

    image_centre = np.array(
        [
            width / 2.0,
            height / 2.0
        ],
        dtype=np.float32
    )

    best_label = None
    best_score = -np.inf

    for label_index in range(
        1,
        number_labels
    ):

        area = float(
            stats[
                label_index,
                cv2.CC_STAT_AREA
            ]
        )

        area_ratio = (
            area / image_area
        )

        # Reject tiny noise and implausibly huge foreground regions.
        if (
            area_ratio < 0.01
            or area_ratio > 0.85
        ):
            continue

        centroid = centroids[
            label_index
        ]

        distance = np.linalg.norm(
            centroid - image_centre
        )

        max_distance = np.hypot(
            width / 2.0,
            height / 2.0
        )

        centre_score = (
            1.0
            - min(
                distance / max_distance,
                1.0
            )
        )

        component = (
            labels == label_index
        )

        touches_border = (
            np.any(component[0, :])
            or np.any(component[-1, :])
            or np.any(component[:, 0])
            or np.any(component[:, -1])
        )

        border_penalty = (
            0.40
            if touches_border
            else 0.0
        )

        # sqrt prevents area from completely dominating the score.
        area_score = np.sqrt(
            area_ratio
        )

        score = (
            (0.60 * area_score)
            + (0.40 * centre_score)
            - border_penalty
        )

        if score > best_score:

            best_score = score
            best_label = label_index

    # --------------------------------------------------------
    # 8. FALLBACK
    # --------------------------------------------------------
    # If validation rejects every component, use the largest
    # non-background component rather than returning an empty mask.
    # --------------------------------------------------------

    if best_label is None:

        if number_labels > 1:

            component_areas = stats[
                1:,
                cv2.CC_STAT_AREA
            ]

            best_label = (
                int(
                    np.argmax(
                        component_areas
                    )
                )
                + 1
            )

        else:

            return np.zeros(
                (
                    original_height,
                    original_width
                ),
                dtype=np.uint8
            )

    clean_mask = np.where(
        labels == best_label,
        255,
        0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 9. FILL INTERNAL HOLES
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(
        clean_mask
    )

    if contours:

        cv2.drawContours(
            filled_mask,
            contours,
            -1,
            255,
            thickness=cv2.FILLED
        )

    else:

        filled_mask = clean_mask

    # --------------------------------------------------------
    # 10. RESIZE TO ORIGINAL IMAGE SIZE
    # --------------------------------------------------------

    final_mask = cv2.resize(
        filled_mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    final_mask = np.where(
        final_mask > 0,
        255,
        0
    ).astype(np.uint8)

    return final_mask


# ============================================================
# 3. GET ALL IMAGE PATHS
# ============================================================

def get_all_images():

    image_paths = []

    for root, folders, files in os.walk(
        DATASET_FOLDER
    ):

        folders.sort()
        files.sort()

        for filename in files:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                image_paths.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return image_paths


# ============================================================
# 4. FORMAT TIME
# ============================================================

def format_time(seconds):

    if seconds < 60:

        return f"{seconds:.0f} seconds"

    minutes = seconds / 60

    if minutes < 60:

        return f"{minutes:.1f} minutes"

    hours = minutes / 60

    return f"{hours:.2f} hours"


# ============================================================
# 5. PROCESS DATASET
# ============================================================

def process_dataset():

    print()
    print("======================================================")
    print("          FAST MEDIAN PREPROCESSING")
    print("======================================================")

    print(
        f"Dataset folder      : {DATASET_FOLDER}"
    )

    print(
        f"Final image size    : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    )

    print(
        f"Median kernel       : {MEDIAN_KERNEL_SIZE}x"
        f"{MEDIAN_KERNEL_SIZE}"
    )

    print(
        f"Segmentation size   : "
        f"{SEGMENTATION_SIZE[0]}x"
        f"{SEGMENTATION_SIZE[1]}"
    )

    print(
        f"GrabCut iterations  : {GRABCUT_ITERATIONS}"
    )

    print("======================================================")

    # --------------------------------------------------------
    # Find all selected images
    # --------------------------------------------------------

    image_paths = get_all_images()

    total_images = len(
        image_paths
    )

    print(
        f"\nTotal images found: {total_images}"
    )

    if total_images == 0:

        print(
            "\nERROR: No images found."
        )

        print(
            "Check DATASET_FOLDER."
        )

        return

    print(
        "\nPreprocessing started...\n"
    )

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    processing_times = []

    start_all = time.perf_counter()

    # --------------------------------------------------------
    # Process every selected image
    # --------------------------------------------------------

    for current_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        # ----------------------------------------------------
        # Preserve fruit/category structure
        # ----------------------------------------------------

        relative_folder = os.path.relpath(
            os.path.dirname(image_path),
            DATASET_FOLDER
        )

        processed_output_folder = os.path.join(
            PROCESSED_FOLDER,
            relative_folder
        )

        mask_output_folder = os.path.join(
            MASK_FOLDER,
            relative_folder
        )

        os.makedirs(
            processed_output_folder,
            exist_ok=True
        )

        os.makedirs(
            mask_output_folder,
            exist_ok=True
        )

        filename = os.path.basename(
            image_path
        )

        filename_without_extension = os.path.splitext(
            filename
        )[0]

        processed_path = os.path.join(
            processed_output_folder,
            filename
        )

        mask_path = os.path.join(
            mask_output_folder,
            filename_without_extension
            + "_mask.png"
        )

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if (
            os.path.exists(processed_path)
            and os.path.exists(mask_path)
        ):

            skipped_count += 1

            continue

        image_start = time.perf_counter()

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            failed_count += 1

            print(
                f"[{current_number}/{total_images}] "
                f"FAILED: {image_path}"
            )

            continue

        # ----------------------------------------------------
        # Resize to 600x600
        # ----------------------------------------------------

        resized = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Median filtering
        # ----------------------------------------------------

        median = median_filter(
            resized
        )

        # ----------------------------------------------------
        # Fast ROI segmentation
        # ----------------------------------------------------

        try:

            roi_mask = create_roi_mask(
                median
            )

        except cv2.error as error:

            failed_count += 1

            print(
                f"[{current_number}/{total_images}] "
                "FAILED: ROI segmentation"
            )

            print(error)

            continue

        # ----------------------------------------------------
        # Save median-filtered image
        # ----------------------------------------------------

        success_processed = cv2.imwrite(
            processed_path,
            median
        )

        # ----------------------------------------------------
        # Save ROI mask
        # ----------------------------------------------------

        success_mask = cv2.imwrite(
            mask_path,
            roi_mask
        )

        if (
            success_processed
            and success_mask
        ):

            processed_count += 1

            image_time = (
                time.perf_counter()
                - image_start
            )

            processing_times.append(
                image_time
            )

        else:

            failed_count += 1

            print(
                f"[{current_number}/{total_images}] "
                "FAILED: Could not save output."
            )

        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        if (
            current_number == 1
            or current_number % PROGRESS_INTERVAL == 0
            or current_number == total_images
        ):

            elapsed = (
                time.perf_counter()
                - start_all
            )

            if processing_times:

                average_time = (
                    sum(processing_times)
                    / len(processing_times)
                )

            else:

                average_time = 0

            remaining = (
                total_images
                - current_number
            )

            estimated_remaining = (
                remaining
                * average_time
            )

            percentage = (
                current_number
                / total_images
                * 100
            )

            print()
            print(
                "------------------------------------------------------"
            )

            print(
                f"Progress           : "
                f"{current_number}/{total_images} "
                f"({percentage:.1f}%)"
            )

            print(
                f"Newly processed    : "
                f"{processed_count}"
            )

            print(
                f"Already processed  : "
                f"{skipped_count}"
            )

            print(
                f"Failed             : "
                f"{failed_count}"
            )

            print(
                f"Average/image      : "
                f"{average_time:.2f} seconds"
            )

            print(
                f"Elapsed            : "
                f"{format_time(elapsed)}"
            )

            print(
                f"Estimated remaining: "
                f"{format_time(estimated_remaining)}"
            )

            print(
                "------------------------------------------------------"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.perf_counter()
        - start_all
    )

    print()
    print("======================================================")
    print("       MEDIAN PREPROCESSING COMPLETED")
    print("======================================================")

    print(
        f"Total images       : {total_images}"
    )

    print(
        f"Newly processed    : {processed_count}"
    )

    print(
        f"Already processed  : {skipped_count}"
    )

    print(
        f"Failed             : {failed_count}"
    )

    print(
        f"Total runtime      : {format_time(total_time)}"
    )

    if processing_times:

        average_time = (
            sum(processing_times)
            / len(processing_times)
        )

        print(
            f"Average/image      : "
            f"{average_time:.2f} seconds"
        )

    print("======================================================")

    print(
        f"\nProcessed images:\n"
        f"{os.path.abspath(PROCESSED_FOLDER)}"
    )

    print(
        f"\nROI masks:\n"
        f"{os.path.abspath(MASK_FOLDER)}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    process_dataset()