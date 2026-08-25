import cv2
import numpy as np
from pathlib import Path

from fruit_mask import create_banana_mask
from otsu import otsu_segmentation
from morphology import morphological_enhancement
from colour_segmentation import colour_segmentation


PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_folder = (
    PROJECT_ROOT
    / "data"
    / "quality"
    / "defect"
)

image_files = (
    list(image_folder.glob("*.jpg"))
    + list(image_folder.glob("*.jpeg"))
    + list(image_folder.glob("*.png"))
)

if not image_files:
    print("No images found.")
    exit()


def calculate_damage_percentage(image):

    # 1. Detect visible banana surface
    banana_mask = create_banana_mask(image)

    # 2. Detect blemishes using the method with
    #    lowest percentage-estimation MAE
    blemish_mask = colour_segmentation(
        image,
        banana_mask
    )

    # Ensure detected blemishes remain inside fruit ROI
    blemish_mask = cv2.bitwise_and(
        blemish_mask,
        banana_mask
    )

    # 3. Count visible fruit pixels
    fruit_pixels = cv2.countNonZero(
        banana_mask
    )

    # 4. Count detected blemish pixels
    blemish_pixels = cv2.countNonZero(
        blemish_mask
    )

    # 5. Calculate visible blemish percentage
    if fruit_pixels == 0:
        blemish_percentage = 0.0
    else:
        blemish_percentage = (
            blemish_pixels / fruit_pixels
        ) * 100

    return (
        blemish_percentage,
        fruit_pixels,
        blemish_pixels,
        banana_mask,
        blemish_mask
    )


# ---------------------------------------
# Test using first image
# ---------------------------------------

image_path = image_files[0]

image = cv2.imread(
    str(image_path)
)

if image is None:
    print("Could not load image.")
    exit()

image = cv2.resize(
    image,
    (600, 600)
)


(
    damage_percentage,
    fruit_pixels,
    blemish_pixels,
    banana_mask,
    blemish_mask
) = calculate_damage_percentage(image)


print("\n====================================")
print("BLEMISH AREA ANALYSIS")
print("====================================")

print("Image:", image_path.name)
print("Fruit surface pixels:", fruit_pixels)
print("Blemish pixels:", blemish_pixels)

print(
    f"Blemish percentage: "
    f"{damage_percentage:.2f}%"
)


# ---------------------------------------
# Highlight detected blemishes
# ---------------------------------------

visualisation = image.copy()

visualisation[
    blemish_mask > 0
] = [0, 0, 255]


# ---------------------------------------
# Display
# ---------------------------------------

cv2.imshow(
    "Original Image",
    image
)

cv2.imshow(
    "Fruit Surface Mask",
    banana_mask
)

cv2.imshow(
    "Final Blemish Mask",
    blemish_mask
)

cv2.imshow(
    "Detected Blemishes",
    visualisation
)

cv2.waitKey(0)
cv2.destroyAllWindows()