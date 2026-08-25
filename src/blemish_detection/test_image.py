import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

defect_folder = PROJECT_ROOT / "data" / "quality" / "defect"

image_files = list(defect_folder.glob("*.jpg")) + list(defect_folder.glob("*.png"))

if not image_files:
    print("No images found in:", defect_folder)
    exit()

image_path = image_files[0]

print("Using image:", image_path.name)

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image could not be loaded.")
    exit()

print("Image loaded successfully!")
print("Image shape:", image.shape)

cv2.imshow("Original Banana", image)

cv2.waitKey(0)
cv2.destroyAllWindows()