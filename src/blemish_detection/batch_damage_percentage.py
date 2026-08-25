import cv2
import csv
from pathlib import Path

from fruit_mask import create_banana_mask
from colour_segmentation import colour_segmentation


PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_folder = (
    PROJECT_ROOT
    / "data"
    / "quality"
    / "defect"
)

results_folder = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = (
    results_folder
    / "blemish_percentage_results.csv"
)


image_files = (
    list(image_folder.glob("*.jpg"))
    + list(image_folder.glob("*.jpeg"))
    + list(image_folder.glob("*.png"))
)

if not image_files:
    print("No images found.")
    exit()


results = []


for image_path in image_files:

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load:", image_path.name)
        continue

    image = cv2.resize(
        image,
        (600, 600)
    )

    # Create banana ROI
    banana_mask = create_banana_mask(
        image
    )

    # Colour-based blemish detection
    blemish_mask = colour_segmentation(
        image,
        banana_mask
    )

    # Ensure blemishes remain inside fruit ROI
    blemish_mask = cv2.bitwise_and(
        blemish_mask,
        banana_mask
    )

    # Count pixels
    fruit_pixels = cv2.countNonZero(
        banana_mask
    )

    blemish_pixels = cv2.countNonZero(
        blemish_mask
    )

    if fruit_pixels == 0:
        blemish_percentage = 0.0
    else:
        blemish_percentage = (
            blemish_pixels / fruit_pixels
        ) * 100

    results.append({
        "Image": image_path.name,
        "Fruit Pixels": fruit_pixels,
        "Blemish Pixels": blemish_pixels,
        "Blemish Percentage": blemish_percentage
    })

    print(
        f"{image_path.name}: "
        f"{blemish_percentage:.2f}%"
    )


# Save CSV
with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "Image",
        "Fruit Pixels",
        "Blemish Pixels",
        "Blemish Percentage"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in results:

        writer.writerow({
            "Image": result["Image"],
            "Fruit Pixels": result["Fruit Pixels"],
            "Blemish Pixels": result["Blemish Pixels"],
            "Blemish Percentage":
                f"{result['Blemish Percentage']:.2f}"
        })


print("\n====================================")
print("Batch processing completed!")
print("Results saved to:")
print(csv_path)
print("====================================")