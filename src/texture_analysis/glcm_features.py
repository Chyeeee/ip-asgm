import cv2
import numpy as np
import pandas as pd

from .config import (
    GLCM_ANGLES_DEGREES,
    GLCM_DISTANCE,
    GLCM_LEVELS,
    GLCM_PROPERTIES,
)
from .data_loader import ImageRecord
from .preprocessing import preprocess_image


# ============================================================
# GRAYSCALE QUANTIZATION
# ============================================================

def quantize_image(
    gray: np.ndarray,
    levels: int,
) -> np.ndarray:
    """
    Reduce grayscale values from 0-255 to a smaller number of levels.

    Example:
    256 grayscale levels -> 32 grayscale levels.

    This reduces GLCM computation while preserving important
    texture information.
    """

    quantized = (
        gray.astype(np.float32)
        * levels
        / 256.0
    )

    quantized = np.clip(
        quantized,
        0,
        levels - 1,
    )

    return quantized.astype(
        np.uint8
    )


# ============================================================
# ANGLE TO PIXEL OFFSET
# ============================================================

def _offset_from_angle(
    distance: int,
    angle_degrees: float,
) -> tuple[int, int]:
    """
    Convert GLCM angle into row and column offsets.

    The assignment uses:
    0 degrees
    45 degrees
    90 degrees
    135 degrees
    """

    radians = np.deg2rad(
        angle_degrees
    )

    dx = int(
        round(
            np.cos(radians)
            * distance
        )
    )

    dy = int(
        round(
            -np.sin(radians)
            * distance
        )
    )

    return dy, dx


# ============================================================
# MASKED GLCM
# ============================================================

def masked_glcm(
    quantized: np.ndarray,
    mask: np.ndarray,
    levels: int,
    distance: int,
    angle_degrees: float,
) -> np.ndarray:
    """
    Construct a GLCM using only banana pixels.

    A pixel pair is counted only when BOTH:
    - reference pixel is inside banana ROI
    - neighbouring pixel is inside banana ROI

    This prevents the white background from affecting the
    texture features.
    """

    dy, dx = _offset_from_angle(
        distance,
        angle_degrees,
    )

    height, width = (
        quantized.shape
    )

    y_start = max(
        0,
        -dy,
    )

    y_end = min(
        height,
        height - dy,
    )

    x_start = max(
        0,
        -dx,
    )

    x_end = min(
        width,
        width - dx,
    )

    source = quantized[
        y_start:y_end,
        x_start:x_end,
    ]

    neighbour = quantized[
        y_start + dy:y_end + dy,
        x_start + dx:x_end + dx,
    ]

    source_mask = (
        mask[
            y_start:y_end,
            x_start:x_end,
        ]
        > 0
    )

    neighbour_mask = (
        mask[
            y_start + dy:y_end + dy,
            x_start + dx:x_end + dx,
        ]
        > 0
    )

    # Both pixels must belong to banana region.
    valid = (
        source_mask
        & neighbour_mask
    )

    source_values = source[
        valid
    ].astype(
        np.int32
    )

    neighbour_values = neighbour[
        valid
    ].astype(
        np.int32
    )

    matrix = np.zeros(
        (
            levels,
            levels,
        ),
        dtype=np.float64,
    )

    # Count gray-level pairs.
    np.add.at(
        matrix,
        (
            source_values,
            neighbour_values,
        ),
        1,
    )

    # Make matrix symmetric.
    matrix = (
        matrix
        + matrix.T
    )

    # Normalize.
    total = matrix.sum()

    if total > 0:
        matrix /= total

    return matrix


# ============================================================
# GLCM FEATURE CALCULATION
# ============================================================

def calculate_glcm_properties(
    matrix: np.ndarray,
) -> dict[str, float]:
    """
    Calculate six GLCM texture features:

    1. Contrast
    2. Dissimilarity
    3. Homogeneity
    4. Energy
    5. Correlation
    6. ASM
    """

    levels = matrix.shape[0]

    i, j = np.indices(
        (
            levels,
            levels,
        )
    )

    difference = (
        i - j
    )

    # --------------------------------------------------------
    # Contrast
    # Higher value = stronger local intensity differences
    # --------------------------------------------------------

    contrast = float(
        np.sum(
            matrix
            * difference**2
        )
    )

    # --------------------------------------------------------
    # Dissimilarity
    # Higher value = neighbouring pixels are more different
    # --------------------------------------------------------

    dissimilarity = float(
        np.sum(
            matrix
            * np.abs(
                difference
            )
        )
    )

    # --------------------------------------------------------
    # Homogeneity
    # Higher value = smoother / more uniform texture
    # --------------------------------------------------------

    homogeneity = float(
        np.sum(
            matrix
            / (
                1.0
                + difference**2
            )
        )
    )

    # --------------------------------------------------------
    # ASM
    # Measures texture uniformity
    # --------------------------------------------------------

    asm = float(
        np.sum(
            matrix**2
        )
    )

    # --------------------------------------------------------
    # Energy
    # Square root of ASM
    # --------------------------------------------------------

    energy = float(
        np.sqrt(
            asm
        )
    )

    # --------------------------------------------------------
    # Correlation
    # Measures relationship between neighbouring gray values
    # --------------------------------------------------------

    pi = matrix.sum(
        axis=1
    )

    pj = matrix.sum(
        axis=0
    )

    values = np.arange(
        levels
    )

    mean_i = np.sum(
        values
        * pi
    )

    mean_j = np.sum(
        values
        * pj
    )

    std_i = np.sqrt(
        np.sum(
            (
                values
                - mean_i
            )**2
            * pi
        )
    )

    std_j = np.sqrt(
        np.sum(
            (
                values
                - mean_j
            )**2
            * pj
        )
    )

    if (
        std_i > 0
        and std_j > 0
    ):

        correlation = float(
            np.sum(
                (
                    (i - mean_i)
                    * (j - mean_j)
                    * matrix
                )
            )
            / (
                std_i
                * std_j
            )
        )

    else:
        correlation = 1.0

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "ASM": asm,
    }


# ============================================================
# GLOBAL GLCM FEATURE EXTRACTION
# ============================================================

def extract_glcm_from_image(
    gray: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract GLCM features from one banana image.

    GLCM is calculated in four directions:
    0, 45, 90 and 135 degrees.

    The mean feature value across all four directions is used.

    Returns:
    - six GLCM features
    - averaged GLCM matrix for visualization
    """

    quantized = quantize_image(
        gray,
        GLCM_LEVELS,
    )

    directional_matrices = []
    directional_properties = []

    for angle in (
        GLCM_ANGLES_DEGREES
    ):

        matrix = masked_glcm(
            quantized=quantized,
            mask=mask,
            levels=GLCM_LEVELS,
            distance=GLCM_DISTANCE,
            angle_degrees=angle,
        )

        directional_matrices.append(
            matrix
        )

        properties = (
            calculate_glcm_properties(
                matrix
            )
        )

        directional_properties.append(
            properties
        )

    features = []

    for property_name in (
        GLCM_PROPERTIES
    ):

        values = [
            properties[
                property_name
            ]
            for properties
            in directional_properties
        ]

        mean_value = np.mean(
            values
        )

        features.append(
            float(
                mean_value
            )
        )

    average_matrix = np.mean(
        directional_matrices,
        axis=0,
    )

    return (
        np.asarray(
            features,
            dtype=np.float64,
        ),
        average_matrix,
    )


# ============================================================
# LOCAL GLCM TEXTURE MAP
# ============================================================

def create_local_glcm_contrast_map(
    gray: np.ndarray,
    mask: np.ndarray,
    window_size: int = 21,
    step: int = 4,
) -> np.ndarray:
    """
    Create a Local GLCM Contrast Map.

    A small window moves across the banana image.

    For every local banana region:
    1. Calculate GLCM.
    2. Extract local contrast.
    3. Store the contrast value.

    The result forms a texture map.

    Brighter regions:
        Higher local contrast / stronger texture variation.

    Darker regions:
        Smoother / more uniform texture.

    This map is mainly used for VISUALIZATION.
    The classification model still uses the six global GLCM features.
    """

    height, width = (
        gray.shape
    )

    half_window = (
        window_size // 2
    )

    y_positions = list(
        range(
            half_window,
            height - half_window,
            step,
        )
    )

    x_positions = list(
        range(
            half_window,
            width - half_window,
            step,
        )
    )

    # Small map containing local contrast values.
    local_map = np.zeros(
        (
            len(y_positions),
            len(x_positions),
        ),
        dtype=np.float32,
    )

    valid_map = np.zeros(
        local_map.shape,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Move window across banana surface
    # --------------------------------------------------------

    for row_index, y in enumerate(
        y_positions
    ):

        for column_index, x in enumerate(
            x_positions
        ):

            # Center pixel must belong to banana.
            if mask[y, x] == 0:
                continue

            y1 = (
                y - half_window
            )

            y2 = (
                y
                + half_window
                + 1
            )

            x1 = (
                x - half_window
            )

            x2 = (
                x
                + half_window
                + 1
            )

            gray_patch = gray[
                y1:y2,
                x1:x2,
            ]

            mask_patch = mask[
                y1:y2,
                x1:x2,
            ]

            # ------------------------------------------------
            # Require most of the window to contain banana.
            # ------------------------------------------------

            banana_ratio = (
                np.count_nonzero(
                    mask_patch
                )
                / mask_patch.size
            )

            if banana_ratio < 0.60:
                continue

            quantized_patch = (
                quantize_image(
                    gray_patch,
                    GLCM_LEVELS,
                )
            )

            local_contrasts = []

            # Calculate local contrast in four directions.
            for angle in (
                GLCM_ANGLES_DEGREES
            ):

                matrix = masked_glcm(
                    quantized=quantized_patch,
                    mask=mask_patch,
                    levels=GLCM_LEVELS,
                    distance=GLCM_DISTANCE,
                    angle_degrees=angle,
                )

                properties = (
                    calculate_glcm_properties(
                        matrix
                    )
                )

                local_contrasts.append(
                    properties[
                        "contrast"
                    ]
                )

            local_map[
                row_index,
                column_index,
            ] = np.mean(
                local_contrasts
            )

            valid_map[
                row_index,
                column_index,
            ] = 1

    # ========================================================
    # NORMALIZE MAP FOR DISPLAY
    # ========================================================

    valid_values = local_map[
        valid_map > 0
    ]

    if len(
        valid_values
    ) > 0:

        # Percentiles prevent one extreme value from
        # dominating the entire visualization.

        minimum = np.percentile(
            valid_values,
            5,
        )

        maximum = np.percentile(
            valid_values,
            95,
        )

        if maximum > minimum:

            local_map = (
                local_map
                - minimum
            ) / (
                maximum
                - minimum
            )

            local_map = np.clip(
                local_map,
                0,
                1,
            )

    # Resize the small local map back to 256 x 256.
    local_map = cv2.resize(
        local_map,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    # Hide background.
    local_map[
        mask == 0
    ] = np.nan

    return local_map


# ============================================================
# EXTRACT GLCM FEATURES FOR FULL DATASET
# ============================================================

def extract_glcm_dataset(
    records: list[ImageRecord],
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the six global GLCM features from every banana image.
    """

    feature_rows = []
    csv_rows = []

    print(
        "\nExtracting masked GLCM features..."
    )

    for index, record in enumerate(
        records,
        start=1,
    ):

        processed = preprocess_image(
            record.path
        )

        features, _ = (
            extract_glcm_from_image(
                processed.gray,
                processed.mask,
            )
        )

        feature_rows.append(
            features
        )

        row = {
            "image_path": str(
                record.path
            ),
            "label": record.label,
        }

        for name, value in zip(
            GLCM_PROPERTIES,
            features,
        ):

            row[
                name
            ] = value

        csv_rows.append(
            row
        )

        if (
            index % 100 == 0
            or index == len(
                records
            )
        ):

            print(
                f"  GLCM: "
                f"{index}/"
                f"{len(records)} images"
            )

    return (
        np.vstack(
            feature_rows
        ),
        pd.DataFrame(
            csv_rows
        ),
    )