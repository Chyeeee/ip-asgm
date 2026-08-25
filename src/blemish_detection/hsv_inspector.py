import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

defect_folder = PROJECT_ROOT / "data" / "quality" / "defect"

image_files = (
    list(defect_folder.glob("*.jpg"))
    + list(defect_folder.glob("*.jpeg"))
    + list(defect_folder.glob("*.png"))
)

if not image_files:
    print("No images found.")
    exit()

image = cv2.imread(str(image_files[0]))
image = cv2.resize(image, (600, 600))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def show_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        bgr_value = image[y, x]
        hsv_value = hsv[y, x]

        print(
            f"Position ({x}, {y}) | "
            f"BGR: {bgr_value} | "
            f"HSV: {hsv_value}"
        )

        display = image.copy()

        cv2.circle(
            display,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.imshow("HSV Inspector", display)


cv2.imshow("HSV Inspector", image)

cv2.setMouseCallback(
    "HSV Inspector",
    show_pixel
)

print("Click different areas of the banana.")
print("Record HSV values for:")
print("1. Healthy yellow peel")
print("2. Light brown blemish")
print("3. Dark brown blemish")
print("Press Q to quit.")

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()