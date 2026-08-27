import cv2
import numpy as np

from config import HIST_BINS


# ============================================================
# COLOUR CONVERSION
# ============================================================

def convert_colour_space(
    image,
    colour_space,
):

    if colour_space == "RGB":

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    if colour_space == "HSV":

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

    if colour_space == "Lab":

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

    raise ValueError(
        f"Unsupported colour space: "
        f"{colour_space}"
    )


# ============================================================
# BASIC FEATURES
# ============================================================

def extract_basic_features(
    image,
    mask,
    colour_space,
):
    """
    RGB:
        R mean
        G mean
        B mean

    HSV:
        H mean
        S mean
        V mean

    Lab:
        L mean
        a mean
        b mean
    """

    converted = convert_colour_space(
        image,
        colour_space,
    )

    roi_pixels = converted[
        mask > 0
    ]

    if len(roi_pixels) == 0:
        return None

    means = np.mean(
        roi_pixels,
        axis=0,
    )

    return means.astype(
        np.float64
    )


# ============================================================
# ENHANCED FEATURES
# ============================================================

def extract_enhanced_features(
    image,
    mask,
    colour_space,
):
    """
    Enhanced method:

    Mean
    +
    Standard deviation
    +
    Normalized histogram
    """

    converted = convert_colour_space(
        image,
        colour_space,
    )

    roi_pixels = converted[
        mask > 0
    ]

    if len(roi_pixels) == 0:
        return None

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    means = np.mean(
        roi_pixels,
        axis=0,
    )

    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    stds = np.std(
        roi_pixels,
        axis=0,
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    histogram_features = []

    for channel in range(3):

        # OpenCV Hue uses range 0-179
        if (
            colour_space == "HSV"
            and channel == 0
        ):

            channel_range = (
                0,
                180,
            )

        else:

            channel_range = (
                0,
                256,
            )

        histogram, _ = np.histogram(
            roi_pixels[:, channel],
            bins=HIST_BINS,
            range=channel_range,
        )

        histogram = histogram.astype(
            np.float64
        )

        # Normalize histogram
        histogram /= (
            np.sum(histogram)
            + 1e-10
        )

        histogram_features.extend(
            histogram.tolist()
        )

    return np.concatenate([
        means,
        stds,
        np.asarray(
            histogram_features,
            dtype=np.float64,
        ),
    ])


# ============================================================
# BASIC FEATURE NAMES
# ============================================================

def get_basic_feature_names(
    colour_space,
):

    if colour_space == "RGB":

        return [
            "R_mean",
            "G_mean",
            "B_mean",
        ]

    if colour_space == "HSV":

        return [
            "H_mean",
            "S_mean",
            "V_mean",
        ]

    if colour_space == "Lab":

        return [
            "L_mean",
            "a_mean",
            "b_mean",
        ]

    raise ValueError(
        f"Unsupported colour space: "
        f"{colour_space}"
    )


# ============================================================
# ENHANCED FEATURE NAMES
# ============================================================

def get_enhanced_feature_names(
    colour_space,
):

    if colour_space == "RGB":

        channels = [
            "R",
            "G",
            "B",
        ]

    elif colour_space == "HSV":

        channels = [
            "H",
            "S",
            "V",
        ]

    elif colour_space == "Lab":

        channels = [
            "L",
            "a",
            "b",
        ]

    else:

        raise ValueError(
            f"Unsupported colour space: "
            f"{colour_space}"
        )

    feature_names = []

    # Mean
    for channel in channels:

        feature_names.append(
            f"{channel}_mean"
        )

    # Standard deviation
    for channel in channels:

        feature_names.append(
            f"{channel}_std"
        )

    # Histogram
    for channel in channels:

        for bin_number in range(
            1,
            HIST_BINS + 1,
        ):

            feature_names.append(
                f"{channel}_hist_"
                f"{bin_number}"
            )

    return feature_names