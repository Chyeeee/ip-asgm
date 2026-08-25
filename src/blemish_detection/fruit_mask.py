import cv2
import numpy as np


def create_banana_mask(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_banana = np.array([0, 30, 15])
    upper_banana = np.array([50, 255, 255])

    banana_mask = cv2.inRange(
        hsv,
        lower_banana,
        upper_banana
    )

    return banana_mask