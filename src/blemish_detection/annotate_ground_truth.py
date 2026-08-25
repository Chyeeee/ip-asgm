import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_folder = PROJECT_ROOT / "data" / "quality" / "defect"
mask_folder = PROJECT_ROOT / "data" / "quality" / "ground_truth"

mask_folder.mkdir(parents=True, exist_ok=True)

image_files = (
    list(image_folder.glob("*.jpg"))
    + list(image_folder.glob("*.jpeg"))
    + list(image_folder.glob("*.png"))
)

if not image_files:
    print("No images found.")
    exit()

current_index = 0
drawing = False
brush_size = 10


def load_image(index):
    image_path = image_files[index]

    image = cv2.imread(str(image_path))

    if image is None:
        return None, None, None

    image = cv2.resize(image, (600, 600))

    mask = np.zeros(
        (image.shape[0], image.shape[1]),
        dtype=np.uint8
    )

    return image_path, image, mask


image_path, image, mask = load_image(current_index)

display = image.copy()


def draw_mask(event, x, y, flags, param):
    global drawing, display, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(
            mask,
            (x, y),
            brush_size,
            255,
            -1
        )

        display = image.copy()

        overlay = np.zeros_like(display)

        overlay[mask > 0] = [0, 0, 255]

        display = cv2.addWeighted(
            display,
            1.0,
            overlay,
            0.5,
            0
        )

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


cv2.namedWindow("Annotate Blemishes")
cv2.setMouseCallback(
    "Annotate Blemishes",
    draw_mask
)

print("Controls:")
print("Left mouse drag = paint blemish")
print("S = save mask")
print("R = reset current mask")
print("N = next image")
print("Q = quit")

while True:

    cv2.imshow("Annotate Blemishes", display)
    cv2.imshow("Ground Truth Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        output_name = image_path.stem + "_mask.png"
        output_path = mask_folder / output_name

        cv2.imwrite(str(output_path), mask)

        print("Saved:", output_path)

    elif key == ord("r"):

        mask[:] = 0
        display = image.copy()

        print("Mask reset.")

    elif key == ord("n"):

        current_index += 1

        if current_index >= len(image_files):
            print("No more images.")
            break

        image_path, image, mask = load_image(current_index)

        display = image.copy()

        print("Current image:", image_path.name)

    elif key == ord("q"):
        break

cv2.destroyAllWindows()