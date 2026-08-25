from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    IMAGE_SIZE,
    MIN_COMPONENT_AREA_RATIO,
    MIN_SATURATION,
    ROI_PADDING,
)


@dataclass
class PreprocessedImage:
    original: np.ndarray
    roi_bgr: np.ndarray
    gray: np.ndarray
    mask: np.ndarray


def _build_foreground_mask(image: np.ndarray) -> np.ndarray:
    """
    Detect the banana region from the light/white background.

    The dataset has yellow/brown bananas on a mostly white background.
    Saturation is therefore used to separate the fruit from the
    low-saturation background.

    After thresholding:
    - small noisy regions are removed;
    - meaningful banana components are retained;
    - external contours are filled so dark bruises/defects inside the
      banana remain part of the banana mask.
    """
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]

    foreground = (
        saturation >= MIN_SATURATION
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    # Remove small isolated noise.
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # Join small gaps within the banana area.
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # Keep only sufficiently large connected components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )

    min_area = int(
        image.shape[0]
        * image.shape[1]
        * MIN_COMPONENT_AREA_RATIO
    )

    cleaned = np.zeros_like(
        foreground
    )

    for component_id in range(1, count):
        area = stats[
            component_id,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:
            cleaned[
                labels == component_id
            ] = 255

    # If segmentation unexpectedly fails, keep the thresholded mask.
    if cv2.countNonZero(cleaned) == 0:
        cleaned = foreground

    # Fill the external banana shapes.
    # This is important because dark bruises may have low saturation,
    # but they are still part of the banana surface.
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filled = np.zeros_like(
        cleaned
    )

    cv2.drawContours(
        filled,
        contours,
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return filled


def _crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop around all detected banana components with a small padding.
    """
    points = cv2.findNonZero(
        mask
    )

    if points is None:
        return (
            image.copy(),
            np.full(
                image.shape[:2],
                255,
                dtype=np.uint8,
            ),
        )

    x, y, width, height = cv2.boundingRect(
        points
    )

    x1 = max(
        0,
        x - ROI_PADDING,
    )
    y1 = max(
        0,
        y - ROI_PADDING,
    )

    x2 = min(
        image.shape[1],
        x + width + ROI_PADDING,
    )
    y2 = min(
        image.shape[0],
        y + height + ROI_PADDING,
    )

    return (
        image[
            y1:y2,
            x1:x2,
        ],
        mask[
            y1:y2,
            x1:x2,
        ],
    )


def preprocess_image(
    image_path: Path,
) -> PreprocessedImage:
    """
    Texture-analysis preprocessing:

    1. Read original image.
    2. Detect banana foreground.
    3. Crop to the banana ROI.
    4. Resize ROI and mask to 256 x 256.
    5. Convert ROI to grayscale.

    The mask is retained so GLCM and LBP can ignore background pixels.
    """
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    original = image.copy()

    mask = _build_foreground_mask(
        image
    )

    roi_bgr, roi_mask = _crop_to_mask(
        image,
        mask,
    )

    roi_bgr = cv2.resize(
        roi_bgr,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    roi_mask = cv2.resize(
        roi_mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )

    roi_mask = (
        roi_mask > 0
    ).astype(np.uint8) * 255

    gray = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    return PreprocessedImage(
        original=original,
        roi_bgr=roi_bgr,
        gray=gray,
        mask=roi_mask,
    )


def create_masked_roi_visual(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Show only the detected banana ROI on a white background.
    """
    output = np.full_like(
        roi_bgr,
        255,
    )

    banana_pixels = (
        mask > 0
    )

    output[
        banana_pixels
    ] = roi_bgr[
        banana_pixels
    ]

    return output


def create_masked_gray_visual(
    gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Show grayscale banana pixels while keeping background white.
    """
    output = np.full_like(
        gray,
        255,
    )

    banana_pixels = (
        mask > 0
    )

    output[
        banana_pixels
    ] = gray[
        banana_pixels
    ]

    return output
