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

# Smaller size ONLY for GrabCut segmentation.
# This makes ROI generation much faster.
SEGMENTATION_SIZE = (300, 300)

# Reduce GrabCut from 5 iterations to 2
GRABCUT_ITERATIONS = 2

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
    Create fruit ROI mask using GrabCut.

    For speed:
    1. Resize 600x600 image to 300x300.
    2. Run GrabCut on smaller image.
    3. Clean mask.
    4. Resize mask back to 600x600.
    """

    original_height, original_width = image.shape[:2]

    # --------------------------------------------------------
    # Resize temporarily for segmentation
    # --------------------------------------------------------

    small_image = cv2.resize(
        image,
        SEGMENTATION_SIZE,
        interpolation=cv2.INTER_AREA
    )

    height, width = small_image.shape[:2]

    # --------------------------------------------------------
    # Create GrabCut mask
    # --------------------------------------------------------

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

    # 5% margin
    margin_x = max(
        1,
        int(width * 0.05)
    )

    margin_y = max(
        1,
        int(height * 0.05)
    )

    rectangle = (
        margin_x,
        margin_y,
        width - (2 * margin_x),
        height - (2 * margin_y)
    )

    # --------------------------------------------------------
    # GrabCut
    # --------------------------------------------------------

    cv2.grabCut(
        small_image,
        mask,
        rectangle,
        background_model,
        foreground_model,
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_RECT
    )

    # --------------------------------------------------------
    # Convert GrabCut classes to binary
    #
    # Fruit      = 255
    # Background = 0
    # --------------------------------------------------------

    binary_mask = np.where(
        (mask == cv2.GC_FGD)
        | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Morphological cleaning
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Keep largest contour
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Resize mask back to original 600x600
    #
    # IMPORTANT:
    # INTER_NEAREST prevents creation of grey mask pixels.
    # --------------------------------------------------------

    binary_mask = cv2.resize(
        binary_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    return binary_mask


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