import cv2

from otsu import otsu_segmentation
from colour_segmentation import colour_segmentation
from morphology import morphological_enhancement


def hybrid_segmentation(image, banana_mask):

    # Otsu candidates
    otsu_mask = otsu_segmentation(
        image,
        banana_mask
    )

    # Colour candidates
    colour_mask = colour_segmentation(
        image,
        banana_mask
    )

    # Keep pixels detected by BOTH methods
    combined = cv2.bitwise_and(
        otsu_mask,
        colour_mask
    )

    # Clean result
    enhanced = morphological_enhancement(
        combined,
        kernel_size=5
    )

    # Restrict final result to fruit
    enhanced = cv2.bitwise_and(
        enhanced,
        banana_mask
    )

    return enhanced