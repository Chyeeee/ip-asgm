"""
Abnormality-Weighted Damage Percentage (AWDP)
=============================================

Member 4 enhancement.

Purpose:
    Improve damage quantification by preventing every Otsu-detected pixel
    from contributing equally to the final damage percentage.

Comparison:
    1. Raw Otsu + Morphology damage %
    2. AWDP damage %
    3. Ground-truth damage %

Metrics:
    MAE  - Mean Absolute Error
    RMSE - Root Mean Squared Error

The best configuration is selected using lowest MAE.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
    / "ground_truth"
)

ROI_ROOT = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "MedianFinal"
    / "ROIMasks"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "MedianFinal"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
    / "awdp"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# WEIGHT CONFIGURATIONS
# ============================================================

# Intensity + Colour + Texture = 1.0

CONFIGURATIONS = [
    {
        "name": "AWDP_A",
        "intensity": 0.50,
        "colour": 0.30,
        "texture": 0.20,
    },
    {
        "name": "AWDP_B",
        "intensity": 0.40,
        "colour": 0.40,
        "texture": 0.20,
    },
    {
        "name": "AWDP_C",
        "intensity": 0.40,
        "colour": 0.30,
        "texture": 0.30,
    },
    {
        "name": "AWDP_D",
        "intensity": 0.33,
        "colour": 0.34,
        "texture": 0.33,
    },
    {
        "name": "AWDP_E",
        "intensity": 0.30,
        "colour": 0.50,
        "texture": 0.20,
    },
]


# ============================================================
# FILE DISCOVERY
# ============================================================

def normalize_stem(name):
    """
    Normalize filenames so files such as:

        Apple_Overripe_084.jpg
        Apple_Overripe_084_mask.png
        Apple_Overripe_084_roi.png
        Apple_Overripe_084_median.png

    can still be matched.
    """

    name = Path(name).stem.lower()

    removable_suffixes = [
        "_gt",
        "_roi",
        "_mask",
        "_roi_mask",
        "_roimask",
        "_fruit_mask",
        "_median",
        "_filtered",
        "_processed",
        "_preprocessed",
    ]

    changed = True

    while changed:
        changed = False

        for suffix in removable_suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                changed = True

    return name


def find_file_by_stem(root, stem):
    """
    Find a corresponding file using flexible filename matching.
    """

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    target = normalize_stem(stem)

    # --------------------------------------------------------
    # Pass 1: normalized exact match
    # --------------------------------------------------------

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        candidate = normalize_stem(path.name)

        if candidate == target:
            return path

    # --------------------------------------------------------
    # Pass 2: target contained in filename
    # --------------------------------------------------------

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        candidate = path.stem.lower()

        if target in candidate:
            return path

    return None


def discover_samples():
    """
    Discover and match:

        Ground-truth blemish mask
        ROI fruit mask
        Median-filtered image

    Flexible filename matching is used because the ROI/image
    files may contain additional suffixes.
    """

    samples = []

    gt_files = sorted(
        GT_ROOT.rglob("*_gt.png")
    )

    print("=" * 70)
    print("AWDP SAMPLE DISCOVERY")
    print("=" * 70)

    print(
        f"Ground-truth masks found: {len(gt_files)}"
    )

    for gt_path in gt_files:

        base_stem = gt_path.stem.replace(
            "_gt",
            "",
        )

        parts = base_stem.split("_")

        if len(parts) < 3:
            print(
                f"[WARNING] Invalid filename: {base_stem}"
            )
            continue

        fruit = parts[0]

        category = "_".join(
            parts[1:-1]
        )

        # ====================================================
        # FIND ROI MASK
        # ====================================================

        roi_path = find_file_by_stem(
            ROI_ROOT,
            base_stem,
        )

        if roi_path is None:

            print(
                f"[WARNING] ROI not found: {base_stem}"
            )

            continue

        # ====================================================
        # FIND PROCESSED IMAGE
        # ====================================================

        image_path = None

        target = normalize_stem(
            base_stem
        )

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }

        # First attempt: normalized exact match
        for candidate in IMAGE_ROOT.rglob("*"):

            if not candidate.is_file():
                continue

            if (
                candidate.suffix.lower()
                not in valid_extensions
            ):
                continue

            # IMPORTANT:
            # Do not treat an ROI mask as the actual image.
            if "ROIMasks" in candidate.parts:
                continue

            if normalize_stem(
                candidate.name
            ) == target:

                image_path = candidate
                break

        # Second attempt: flexible contains match
        if image_path is None:

            for candidate in IMAGE_ROOT.rglob("*"):

                if not candidate.is_file():
                    continue

                if (
                    candidate.suffix.lower()
                    not in valid_extensions
                ):
                    continue

                if "ROIMasks" in candidate.parts:
                    continue

                if (
                    target
                    in candidate.stem.lower()
                ):

                    image_path = candidate
                    break

        if image_path is None:

            print(
                f"[WARNING] Image not found: {base_stem}"
            )

            continue

        # ====================================================
        # SUCCESS
        # ====================================================

        samples.append(
            {
                "fruit": fruit,
                "category": category,
                "stem": base_stem,
                "gt": gt_path,
                "roi": roi_path,
                "image": image_path,
            }
        )

        print(
            f"[MATCHED] {base_stem}"
        )

        print(
            f"          ROI   : {roi_path.name}"
        )

        print(
            f"          Image : {image_path.name}"
        )

    print()
    print(
        f"Valid matched samples: {len(samples)}"
    )

    print("=" * 70)

    return samples


# ============================================================
# IMAGE LOADING
# ============================================================

def load_binary_mask(path, target_shape=None):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise ValueError(
            f"Unable to read mask: {path}"
        )

    if (
        target_shape is not None
        and mask.shape != target_shape
    ):

        mask = cv2.resize(
            mask,
            (
                target_shape[1],
                target_shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    return (
        mask > 127
    ).astype(
        np.uint8
    )


# ============================================================
# OTSU + MORPHOLOGY
# ============================================================

def otsu_morphology_segmentation(
    image,
    roi_mask,
):
    """
    Reproduce Otsu + Morphology 7x7.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    roi_pixels = gray[
        roi_mask > 0
    ]

    if roi_pixels.size == 0:
        return np.zeros_like(
            roi_mask
        )

    # Use only ROI pixels to estimate Otsu threshold
    threshold, _ = cv2.threshold(
        roi_pixels.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    # Darker pixels treated as blemish candidates
    mask = (
        (gray <= threshold)
        & (roi_mask > 0)
    ).astype(
        np.uint8
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask[
        roi_mask == 0
    ] = 0

    return mask


# ============================================================
# INTENSITY ABNORMALITY
# ============================================================

def intensity_abnormality(
    image,
    roi_mask,
):
    """
    Dark pixels receive higher abnormality.

    Output:
        0.0 = normal
        1.0 = strongly abnormal
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(
        np.float32
    )

    roi_values = gray[
        roi_mask > 0
    ]

    if roi_values.size == 0:
        return np.zeros_like(
            gray,
            dtype=np.float32,
        )

    median_intensity = np.median(
        roi_values
    )

    score = (
        median_intensity - gray
    ) / max(
        median_intensity,
        1.0,
    )

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    score[
        roi_mask == 0
    ] = 0.0

    return score


# ============================================================
# COLOUR ABNORMALITY
# ============================================================

def colour_abnormality(
    image,
    roi_mask,
):
    """
    Estimate colour deviation from the dominant/median fruit colour
    using LAB colour space.

    Pixels further from the fruit's median LAB colour receive a
    higher abnormality score.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    roi_pixels = lab[
        roi_mask > 0
    ]

    if roi_pixels.size == 0:

        return np.zeros(
            roi_mask.shape,
            dtype=np.float32,
        )

    reference_colour = np.median(
        roi_pixels,
        axis=0,
    )

    difference = (
        lab
        - reference_colour
    )

    distance = np.sqrt(
        np.sum(
            difference ** 2,
            axis=2,
        )
    )

    roi_distance = distance[
        roi_mask > 0
    ]

    # Robust normalization
    scale = np.percentile(
        roi_distance,
        95,
    )

    if scale <= 0:
        scale = 1.0

    score = distance / scale

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    score[
        roi_mask == 0
    ] = 0.0

    return score


# ============================================================
# TEXTURE ABNORMALITY
# ============================================================

def texture_abnormality(
    image,
    roi_mask,
):
    """
    Local texture abnormality based on local standard deviation.

    High local variation receives a larger score.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(
        np.float32
    )

    window = 11

    mean = cv2.blur(
        gray,
        (window, window),
    )

    mean_square = cv2.blur(
        gray ** 2,
        (window, window),
    )

    variance = (
        mean_square
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        0,
    )

    std = np.sqrt(
        variance
    )

    roi_std = std[
        roi_mask > 0
    ]

    if roi_std.size == 0:

        return np.zeros_like(
            gray,
            dtype=np.float32,
        )

    scale = np.percentile(
        roi_std,
        95,
    )

    if scale <= 0:
        scale = 1.0

    score = std / scale

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    score[
        roi_mask == 0
    ] = 0.0

    return score


# ============================================================
# RAW DAMAGE
# ============================================================

def calculate_raw_damage(
    blemish_mask,
    roi_mask,
):

    fruit_area = np.count_nonzero(
        roi_mask
    )

    if fruit_area == 0:
        return 0.0

    blemish_area = np.count_nonzero(
        (blemish_mask > 0)
        & (roi_mask > 0)
    )

    return (
        blemish_area
        / fruit_area
    ) * 100.0


# ============================================================
# GT DAMAGE
# ============================================================

def calculate_gt_damage(
    gt_mask,
    roi_mask,
):

    fruit_area = np.count_nonzero(
        roi_mask
    )

    if fruit_area == 0:
        return 0.0

    gt_area = np.count_nonzero(
        (gt_mask > 0)
        & (roi_mask > 0)
    )

    return (
        gt_area
        / fruit_area
    ) * 100.0


# ============================================================
# AWDP
# ============================================================

def calculate_awdp(
    blemish_mask,
    roi_mask,
    intensity_score,
    colour_score,
    texture_score,
    config,
):
    """
    Abnormality-Weighted Damage Percentage.

    Confidence:

        C(x) =
            wI * Intensity(x)
          + wC * Colour(x)
          + wT * Texture(x)

    Damage:

        AWDP =
            sum(M(x) * C(x))
            -----------------
                fruit area

            * 100
    """

    confidence = (
        config["intensity"]
        * intensity_score

        + config["colour"]
        * colour_score

        + config["texture"]
        * texture_score
    )

    confidence = np.clip(
        confidence,
        0.0,
        1.0,
    )

    weighted_mask = (
        blemish_mask.astype(
            np.float32
        )
        * confidence
    )

    weighted_mask[
        roi_mask == 0
    ] = 0.0

    fruit_area = np.count_nonzero(
        roi_mask
    )

    if fruit_area == 0:
        return 0.0

    weighted_damage = (
        np.sum(
            weighted_mask
        )
        / fruit_area
    ) * 100.0

    return float(
        weighted_damage
    )


# ============================================================
# ERROR METRICS
# ============================================================

def calculate_mae(
    ground_truth,
    prediction,
):

    return float(
        np.mean(
            np.abs(
                np.asarray(ground_truth)
                - np.asarray(prediction)
            )
        )
    )


def calculate_rmse(
    ground_truth,
    prediction,
):

    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(ground_truth)
                    - np.asarray(prediction)
                ) ** 2
            )
        )
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    samples = discover_samples()

    if len(samples) == 0:

        raise RuntimeError(
            "No valid GT samples were found."
        )

    print()
    print("=" * 70)
    print(
        "ABNORMALITY-WEIGHTED DAMAGE PERCENTAGE"
    )
    print("=" * 70)

    print(
        f"Ground-truth images: {len(samples)}"
    )

    print("\nWeight configurations:")

    for config in CONFIGURATIONS:

        print(
            f"  {config['name']}: "
            f"I={config['intensity']:.2f}, "
            f"C={config['colour']:.2f}, "
            f"T={config['texture']:.2f}"
        )

    detailed_rows = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(samples)}] "
            f"{sample['fruit']} | "
            f"{sample['category']} | "
            f"{sample['stem']}"
        )

        image = cv2.imread(
            str(
                sample["image"]
            )
        )

        if image is None:

            print(
                "  ERROR: Unable to load image."
            )

            continue

        h, w = image.shape[:2]

        roi_mask = load_binary_mask(
            sample["roi"],
            (h, w),
        )

        gt_mask = load_binary_mask(
            sample["gt"],
            (h, w),
        )

        # ----------------------------------------------------
        # Baseline segmentation
        # ----------------------------------------------------

        blemish_mask = (
            otsu_morphology_segmentation(
                image,
                roi_mask,
            )
        )

        # ----------------------------------------------------
        # Raw damage
        # ----------------------------------------------------

        raw_damage = calculate_raw_damage(
            blemish_mask,
            roi_mask,
        )

        # ----------------------------------------------------
        # GT damage
        # ----------------------------------------------------

        gt_damage = calculate_gt_damage(
            gt_mask,
            roi_mask,
        )

        # ----------------------------------------------------
        # Abnormality cues
        # ----------------------------------------------------

        intensity_score = (
            intensity_abnormality(
                image,
                roi_mask,
            )
        )

        colour_score = (
            colour_abnormality(
                image,
                roi_mask,
            )
        )

        texture_score = (
            texture_abnormality(
                image,
                roi_mask,
            )
        )

        print(
            f"  GT Damage        : "
            f"{gt_damage:.2f}%"
        )

        print(
            f"  Raw Otsu Damage  : "
            f"{raw_damage:.2f}%"
        )

        # ----------------------------------------------------
        # AWDP configurations
        # ----------------------------------------------------

        for config in CONFIGURATIONS:

            awdp = calculate_awdp(
                blemish_mask,
                roi_mask,
                intensity_score,
                colour_score,
                texture_score,
                config,
            )

            absolute_error = abs(
                awdp
                - gt_damage
            )

            detailed_rows.append(
                {
                    "fruit":
                        sample["fruit"],

                    "category":
                        sample["category"],

                    "image":
                        sample["stem"],

                    "method":
                        config["name"],

                    "gt_damage_percentage":
                        gt_damage,

                    "raw_damage_percentage":
                        raw_damage,

                    "predicted_damage_percentage":
                        awdp,

                    "absolute_error":
                        absolute_error,

                    "intensity_weight":
                        config["intensity"],

                    "colour_weight":
                        config["colour"],

                    "texture_weight":
                        config["texture"],
                }
            )

            print(
                f"  {config['name']:<10}: "
                f"{awdp:6.2f}% "
                f"| Error={absolute_error:6.2f}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    df = pd.DataFrame(
        detailed_rows
    )

    if df.empty:

        raise RuntimeError(
            "No AWDP results were generated."
        )

    # --------------------------------------------------------
    # Raw baseline
    # --------------------------------------------------------

    baseline = (
        df
        .drop_duplicates(
            subset=[
                "fruit",
                "category",
                "image",
            ]
        )
    )

    raw_mae = calculate_mae(
        baseline[
            "gt_damage_percentage"
        ],
        baseline[
            "raw_damage_percentage"
        ],
    )

    raw_rmse = calculate_rmse(
        baseline[
            "gt_damage_percentage"
        ],
        baseline[
            "raw_damage_percentage"
        ],
    )

    # --------------------------------------------------------
    # AWDP summary
    # --------------------------------------------------------

    summary_rows = []

    for config in CONFIGURATIONS:

        config_df = df[
            df["method"]
            == config["name"]
        ]

        mae = calculate_mae(
            config_df[
                "gt_damage_percentage"
            ],
            config_df[
                "predicted_damage_percentage"
            ],
        )

        rmse = calculate_rmse(
            config_df[
                "gt_damage_percentage"
            ],
            config_df[
                "predicted_damage_percentage"
            ],
        )

        summary_rows.append(
            {
                "method":
                    config["name"],

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "intensity_weight":
                    config["intensity"],

                "colour_weight":
                    config["colour"],

                "texture_weight":
                    config["texture"],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        "MAE"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "AWDP DAMAGE QUANTIFICATION SUMMARY"
    )
    print("=" * 70)

    print()
    print("Raw Otsu + Morphology")

    print(
        f"  MAE  : {raw_mae:.4f} percentage points"
    )

    print(
        f"  RMSE : {raw_rmse:.4f} percentage points"
    )

    for _, row in summary_df.iterrows():

        print()
        print(row["method"])

        print(
            f"  MAE  : "
            f"{row['MAE']:.4f} percentage points"
        )

        print(
            f"  RMSE : "
            f"{row['RMSE']:.4f} percentage points"
        )

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = summary_df.iloc[0]

    improvement = (
        raw_mae
        - best["MAE"]
    )

    relative_improvement = (
        improvement
        / raw_mae
        * 100
        if raw_mae > 0
        else 0
    )

    print()
    print("=" * 70)
    print("BEST AWDP CONFIGURATION")
    print("=" * 70)

    print(
        f"Method            : "
        f"{best['method']}"
    )

    print(
        f"Intensity Weight  : "
        f"{best['intensity_weight']:.2f}"
    )

    print(
        f"Colour Weight     : "
        f"{best['colour_weight']:.2f}"
    )

    print(
        f"Texture Weight    : "
        f"{best['texture_weight']:.2f}"
    )

    print(
        f"AWDP MAE          : "
        f"{best['MAE']:.4f}"
    )

    print(
        f"Raw Damage MAE    : "
        f"{raw_mae:.4f}"
    )

    print(
        f"MAE Improvement   : "
        f"{improvement:+.4f} percentage points"
    )

    print(
        f"Relative Change   : "
        f"{relative_improvement:+.2f}%"
    )

    if best["MAE"] < raw_mae:

        print(
            "\nRESULT: AWDP IMPROVED damage quantification."
        )

    else:

        print(
            "\nRESULT: AWDP DID NOT improve damage quantification."
        )

    print("=" * 70)

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_DIR
        / "awdp_detailed_results.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "awdp_summary.csv",
        index=False,
    )

    baseline_summary = pd.DataFrame(
        [
            {
                "method":
                    "Raw Otsu + Morphology",

                "MAE":
                    raw_mae,

                "RMSE":
                    raw_rmse,
            }
        ]
    )

    baseline_summary.to_csv(
        OUTPUT_DIR
        / "raw_damage_summary.csv",
        index=False,
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiment()