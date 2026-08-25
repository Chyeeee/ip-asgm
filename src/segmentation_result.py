from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def create_banana_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect white/light background
    lower_white = np.array([0, 0, 170], dtype=np.uint8)
    upper_white = np.array([179, 70, 255], dtype=np.uint8)
    background_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert mask: banana = white, background = black
    banana_mask = cv2.bitwise_not(background_mask)

    # Remove noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    banana_mask = cv2.morphologyEx(
        banana_mask, cv2.MORPH_OPEN, kernel
    )
    banana_mask = cv2.morphologyEx(
        banana_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # Keep largest object only
    contours, _ = cv2.findContours(
        banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(banana_mask)
        cv2.drawContours(
            clean_mask, [largest_contour], -1, 255, cv2.FILLED
        )
        banana_mask = clean_mask

    return banana_mask

# Use one Green banana as example
green_folder = DATASET_DIR / "Green"

image_files = sorted([
    file for file in green_folder.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

if not image_files:
    raise FileNotFoundError("No image found in Green folder.")

image_path = image_files[0]
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Unable to read image.")

mask = create_banana_mask(image)

# Keep only banana pixels
segmented = cv2.bitwise_and(
    image, image, mask=mask
)

# OpenCV BGR -> RGB for matplotlib
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

# Create one combined figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Banana Mask")
axes[1].axis("off")

axes[2].imshow(segmented_rgb)
axes[2].set_title("Segmented Banana")
axes[2].axis("off")

plt.tight_layout()

output_path = RESULTS_DIR / "segmentation_result.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Image used:", image_path.name)
print("Segmentation result saved to:")
print(output_path)

plt.show()