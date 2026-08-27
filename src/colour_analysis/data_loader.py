import cv2
import pandas as pd

from config import (
    PROCESSED_DIR,
    MASK_DIR,
    SUPPORTED_EXTENSIONS,
    SAMPLE_PER_FRUIT_CATEGORY,
    RANDOM_STATE,
)


# ============================================================
# FIND CORRESPONDING ROI MASK
# ============================================================

def find_mask(image_path):
    """
    Find the ROI mask corresponding to a processed image.

    Example:

    Processed image:
    Apple_Overripe_001.jpg

    ROI mask:
    Apple_Overripe_001_mask.png
    """

    # Get path relative to ProcessedImages
    #
    # Example:
    # Apple/Overripe/Apple_Overripe_001.jpg
    relative_path = image_path.relative_to(
        PROCESSED_DIR
    )

    # Corresponding ROI mask folder
    #
    # Example:
    # ROIMasks/Apple/Overripe/
    mask_folder = (
        MASK_DIR
        / relative_path.parent
    )

    # ========================================================
    # MAIN MASK NAMING FORMAT
    # ========================================================
    #
    # Processed:
    # Apple_Overripe_001.jpg
    #
    # Mask:
    # Apple_Overripe_001_mask.png
    #

    mask_stem = (
        f"{image_path.stem}_mask"
    )

    for extension in SUPPORTED_EXTENSIONS:

        mask_path = (
            mask_folder
            / f"{mask_stem}{extension}"
        )

        if mask_path.exists():
            return mask_path

    # ========================================================
    # FALLBACK
    # ========================================================
    # Also try exact same filename stem just in case
    # some masks do not contain "_mask".
    # ========================================================

    for extension in SUPPORTED_EXTENSIONS:

        mask_path = (
            mask_folder
            / f"{image_path.stem}{extension}"
        )

        if mask_path.exists():
            return mask_path

    return None


# ============================================================
# GET DATASET INFORMATION
# ============================================================

def get_metadata(image_path):
    """
    Extract fruit name and ripeness category
    from the folder structure.

    Example:

    ProcessedImages/
        Apple/
            Overripe/
                Apple_Overripe_001.jpg

    fruit    = Apple
    category = Overripe
    """

    relative_path = (
        image_path.relative_to(
            PROCESSED_DIR
        )
    )

    parts = relative_path.parts

    # Immediate parent folder
    # Example: Overripe
    category = (
        image_path.parent.name
    )

    # Folder before category
    # Example: Apple
    if len(parts) >= 3:

        fruit = parts[-3]

    else:

        fruit = "Unknown"

    return (
        fruit,
        category,
    )


# ============================================================
# GET ALL IMAGE + MASK PAIRS
# ============================================================

def get_dataset_records():
    """
    Scan all processed images and find
    their corresponding ROI masks.
    """

    # --------------------------------------------------------
    # Check folders
    # --------------------------------------------------------

    if not PROCESSED_DIR.exists():

        raise FileNotFoundError(
            f"\nProcessedImages folder "
            f"not found:\n"
            f"{PROCESSED_DIR}"
        )

    if not MASK_DIR.exists():

        raise FileNotFoundError(
            f"\nROIMasks folder "
            f"not found:\n"
            f"{MASK_DIR}"
        )

    # --------------------------------------------------------
    # Find processed images
    # --------------------------------------------------------

    image_paths = sorted([
        path
        for path
        in PROCESSED_DIR.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    print(
        f"\nProcessed images found: "
        f"{len(image_paths)}"
    )

    records = []

    missing_masks = 0

    # --------------------------------------------------------
    # Match image with mask
    # --------------------------------------------------------

    for image_path in image_paths:

        mask_path = find_mask(
            image_path
        )

        if mask_path is None:

            missing_masks += 1

            continue

        fruit, category = (
            get_metadata(
                image_path
            )
        )

        records.append({
            "image_path":
                image_path,

            "mask_path":
                mask_path,

            "fruit":
                fruit,

            "category":
                category,
        })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Valid image-mask pairs: "
        f"{len(records)}"
    )

    print(
        f"Missing masks: "
        f"{missing_masks}"
    )

    return records


# ============================================================
# CREATE REPRESENTATIVE SAMPLE
# ============================================================

def create_representative_sample(
    records
):
    """
    Select a representative balanced sample
    for RGB vs HSV vs Lab comparison.

    SAMPLE_PER_FRUIT_CATEGORY controls how many images
    are selected from each ripeness category.
    """

    df = pd.DataFrame(
        records
    )

    if df.empty:

        raise RuntimeError(
            "No valid image-mask pairs "
            "were found."
        )

    # --------------------------------------------------------
    # Show available images
    # --------------------------------------------------------

    print(
        "\nAvailable images "
        "per category:"
    )

    print(
        df[
            "category"
        ]
        .value_counts()
        .sort_index()
    )

    sampled_groups = []

    # --------------------------------------------------------
    # Sample same number from each class
    # --------------------------------------------------------

    for (
        category,
        group,
    ) in df.groupby(
        "category"
    ):

        sample_size = min(
            SAMPLE_PER_FRUIT_CATEGORY,
            len(group),
        )

        sampled = group.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )

        sampled_groups.append(
            sampled
        )

    # --------------------------------------------------------
    # Combine sampled classes
    # --------------------------------------------------------

    sample_df = pd.concat(
        sampled_groups,
        ignore_index=True,
    )

    # Shuffle sample
    sample_df = sample_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Show sample distribution
    # --------------------------------------------------------

    print(
        "\nRepresentative sample:"
    )

    print(
        sample_df[
            "category"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nTotal comparison images: "
        f"{len(sample_df)}"
    )

    return sample_df


# ============================================================
# LOAD IMAGE AND ROI MASK
# ============================================================

def load_image_mask(
    image_path,
    mask_path,
):
    """
    Load processed image and corresponding
    binary ROI mask.
    """

    # --------------------------------------------------------
    # Load processed image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    # --------------------------------------------------------
    # Load ROI mask as grayscale
    # --------------------------------------------------------

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise ValueError(
            f"Cannot read image: "
            f"{image_path}"
        )

    if mask is None:

        raise ValueError(
            f"Cannot read mask: "
            f"{mask_path}"
        )

    # --------------------------------------------------------
    # Make sure mask size matches image
    # --------------------------------------------------------

    if (
        mask.shape[:2]
        !=
        image.shape[:2]
    ):

        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=
                cv2.INTER_NEAREST,
        )

    # --------------------------------------------------------
    # Convert mask into strict binary:
    #
    # background = 0
    # fruit      = 255
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return (
        image,
        mask,
    )