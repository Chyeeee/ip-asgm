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
    Match processed image with corresponding ROI mask.

    Example:
    Apple_Overripe_001.jpg
    ->
    Apple_Overripe_001_mask.png
    """

    relative_path = image_path.relative_to(
        PROCESSED_DIR
    )

    mask_folder = (
        MASK_DIR
        / relative_path.parent
    )

    # Main mask format
    mask_stem = f"{image_path.stem}_mask"

    for extension in SUPPORTED_EXTENSIONS:

        mask_path = (
            mask_folder
            / f"{mask_stem}{extension}"
        )

        if mask_path.exists():
            return mask_path

    # Fallback: same stem without "_mask"
    for extension in SUPPORTED_EXTENSIONS:

        mask_path = (
            mask_folder
            / f"{image_path.stem}{extension}"
        )

        if mask_path.exists():
            return mask_path

    return None


# ============================================================
# GET FRUIT + CATEGORY
# ============================================================

def get_metadata(image_path):
    """
    Example:

    ProcessedImages/
        Apple/
            Overripe/
                Apple_Overripe_001.jpg

    fruit    = Apple
    category = Overripe
    """

    relative_path = image_path.relative_to(
        PROCESSED_DIR
    )

    parts = relative_path.parts

    category = image_path.parent.name

    if len(parts) >= 3:
        fruit = parts[-3]
    else:
        fruit = "Unknown"

    return fruit, category


# ============================================================
# GET COMPLETE DATASET
# ============================================================

def get_dataset_records():
    """
    Scan all processed images and match them
    with their ROI masks.
    """

    if not PROCESSED_DIR.exists():

        raise FileNotFoundError(
            f"\nProcessedImages folder not found:\n"
            f"{PROCESSED_DIR}"
        )

    if not MASK_DIR.exists():

        raise FileNotFoundError(
            f"\nROIMasks folder not found:\n"
            f"{MASK_DIR}"
        )

    image_paths = sorted([
        path
        for path in PROCESSED_DIR.rglob("*")
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

    for image_path in image_paths:

        mask_path = find_mask(
            image_path
        )

        if mask_path is None:

            missing_masks += 1
            continue

        fruit, category = get_metadata(
            image_path
        )

        records.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "fruit": fruit,
            "category": category,
        })

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
# CREATE BALANCED PER-FRUIT SAMPLE
# ============================================================

def create_per_fruit_sample(records):
    """
    Create a representative sample separately
    for every fruit + category.

    Example:

    Apple:
        Overripe = 20
        Ripe     = 20
        Rotten   = 20
        Unripe   = 20

    Guava:
        Class_A  = 20
        Class_B  = 20
        Defect   = 20
    """

    df = pd.DataFrame(
        records
    )

    if df.empty:

        raise RuntimeError(
            "No valid image-mask pairs found."
        )

    print("\nDataset distribution:")
    print("-" * 70)

    distribution = (
        df.groupby(
            ["fruit", "category"]
        )
        .size()
        .reset_index(
            name="Count"
        )
    )

    print(
        distribution.to_string(
            index=False
        )
    )

    sampled_groups = []

    for (
        fruit,
        category,
    ), group in df.groupby(
        ["fruit", "category"]
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

    sample_df = pd.concat(
        sampled_groups,
        ignore_index=True,
    )

    sample_df = sample_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    print("\n")
    print("=" * 70)
    print("REPRESENTATIVE SAMPLE")
    print("=" * 70)

    sample_distribution = (
        sample_df.groupby(
            ["fruit", "category"]
        )
        .size()
        .reset_index(
            name="Sample"
        )
    )

    print(
        sample_distribution.to_string(
            index=False
        )
    )

    print(
        f"\nTotal comparison images: "
        f"{len(sample_df)}"
    )

    return sample_df


# ============================================================
# LOAD IMAGE + ROI MASK
# ============================================================

def load_image_mask(
    image_path,
    mask_path,
):
    """
    Load processed image and its ROI mask.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

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

    # Make sure mask size matches image
    if mask.shape[:2] != image.shape[:2]:

        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    # Convert ROI mask into strict binary mask
    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return image, mask