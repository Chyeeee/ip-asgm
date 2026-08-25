import cv2


def adaptive_segmentation(image, banana_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    blemish_mask = cv2.bitwise_and(
        adaptive,
        banana_mask
    )

    return blemish_mask