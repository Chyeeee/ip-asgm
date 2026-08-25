import cv2
import numpy as np


def morphological_enhancement(mask, kernel_size=3):
    """
    Enhance a binary blemish mask using
    morphological opening and closing.

    Opening:
    Removes small isolated noise.

    Closing:
    Fills small holes and connects nearby
    blemish regions.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    # Remove small noise
    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    enhanced = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    return enhanced