import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

from .config import (
    LBP_METHOD,
    LBP_POINTS,
    LBP_RADIUS,
)
from .data_loader import ImageRecord
from .preprocessing import preprocess_image


def extract_lbp_from_image(
    gray: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate LBP across the ROI, but build the histogram using only
    pixels inside the banana mask.
    """
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD,
    )

    number_of_bins = (
        LBP_POINTS + 2
    )

    banana_lbp_values = lbp[
        mask > 0
    ]

    histogram, _ = np.histogram(
        banana_lbp_values,
        bins=np.arange(
            0,
            number_of_bins + 1,
        ),
        range=(
            0,
            number_of_bins,
        ),
    )

    histogram = histogram.astype(
        np.float64
    )

    histogram /= (
        histogram.sum()
        + 1e-12
    )

    return histogram, lbp


def extract_lbp_dataset(
    records: list[ImageRecord],
) -> tuple[np.ndarray, pd.DataFrame]:
    feature_rows = []
    csv_rows = []

    print(
        "\nExtracting masked LBP features..."
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        processed = preprocess_image(
            record.path
        )

        features, _ = extract_lbp_from_image(
            processed.gray,
            processed.mask,
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

        for bin_index, value in enumerate(
            features
        ):
            row[
                f"lbp_bin_{bin_index}"
            ] = value

        csv_rows.append(
            row
        )

        if (
            index % 100 == 0
            or index == len(records)
        ):
            print(
                f"  LBP : "
                f"{index}/{len(records)} images"
            )

    return (
        np.vstack(
            feature_rows
        ),
        pd.DataFrame(
            csv_rows
        ),
    )
