import cv2
import numpy as np


def colour_segmentation(image, banana_mask):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_blemish = np.array([0, 120, 15])
    upper_blemish = np.array([19, 255, 145])

    colour_mask = cv2.inRange(
        hsv,
        lower_blemish,
        upper_blemish
    )

    blemish_mask = cv2.bitwise_and(
        colour_mask,
        banana_mask
    )

    return blemish_mask