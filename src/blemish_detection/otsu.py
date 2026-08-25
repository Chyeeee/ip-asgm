import cv2


def otsu_segmentation(image, banana_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    blemish_mask = cv2.bitwise_and(
        otsu,
        banana_mask
    )

    return blemish_mask