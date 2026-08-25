import cv2
import csv
from pathlib import Path

from fruit_mask import create_banana_mask
from otsu import otsu_segmentation
from colour_segmentation import colour_segmentation
from morphology import morphological_enhancement
from hybrid import hybrid_segmentation


PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_folder = PROJECT_ROOT / "data" / "quality" / "defect"
gt_folder = PROJECT_ROOT / "data" / "quality" / "ground_truth"

results_folder = PROJECT_ROOT / "results" / "blemish_detection"
results_folder.mkdir(parents=True, exist_ok=True)

csv_path = results_folder / "damage_percentage_comparison.csv"


image_files = (
    list(image_folder.glob("*.jpg"))
    + list(image_folder.glob("*.jpeg"))
    + list(image_folder.glob("*.png"))
)


methods = [
    "Otsu",
    "Otsu + Morphology",
    "Colour-Based",
    "Hybrid"
]

absolute_errors = {method: [] for method in methods}
rows = []


for image_path in image_files:

    gt_path = gt_folder / f"{image_path.stem}_mask.png"

    if not gt_path.exists():
        continue

    image = cv2.imread(str(image_path))

    ground_truth = cv2.imread(
        str(gt_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or ground_truth is None:
        continue

    image = cv2.resize(image, (600, 600))

    ground_truth = cv2.resize(
        ground_truth,
        (600, 600),
        interpolation=cv2.INTER_NEAREST
    )

    # ---------------------------------
    # Fruit surface
    # ---------------------------------

    banana_mask = create_banana_mask(image)

    fruit_pixels = cv2.countNonZero(banana_mask)

    if fruit_pixels == 0:
        continue

    # Ground truth must stay inside fruit ROI
    ground_truth = cv2.bitwise_and(
        ground_truth,
        banana_mask
    )

    actual_pixels = cv2.countNonZero(ground_truth)

    actual_percentage = (
        actual_pixels / fruit_pixels
    ) * 100


    # ---------------------------------
    # Generate masks
    # ---------------------------------

    otsu_mask = otsu_segmentation(
        image,
        banana_mask
    )

    morphology_mask = morphological_enhancement(
        otsu_mask,
        kernel_size=5
    )

    morphology_mask = cv2.bitwise_and(
        morphology_mask,
        banana_mask
    )

    colour_mask = colour_segmentation(
        image,
        banana_mask
    )

    hybrid_mask = hybrid_segmentation(
        image,
        banana_mask
    )


    masks = {
        "Otsu": otsu_mask,
        "Otsu + Morphology": morphology_mask,
        "Colour-Based": colour_mask,
        "Hybrid": hybrid_mask
    }


    # ---------------------------------
    # Compare percentage
    # ---------------------------------

    for method, mask in masks.items():

        predicted_pixels = cv2.countNonZero(mask)

        predicted_percentage = (
            predicted_pixels / fruit_pixels
        ) * 100

        error = abs(
            predicted_percentage
            - actual_percentage
        )

        absolute_errors[method].append(error)

        rows.append({
            "Image": image_path.name,
            "Method": method,
            "Ground Truth (%)": actual_percentage,
            "Predicted (%)": predicted_percentage,
            "Absolute Error": error
        })


# ---------------------------------
# Calculate MAE
# ---------------------------------

print("\n======================================")
print("DAMAGE PERCENTAGE COMPARISON")
print("======================================")

for method in methods:

    errors = absolute_errors[method]

    if errors:
        mae = sum(errors) / len(errors)
    else:
        mae = 0

    print(f"\n{method}")
    print(
        f"Mean Absolute Error: "
        f"{mae:.2f} percentage points"
    )


# ---------------------------------
# Save CSV
# ---------------------------------

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "Image",
        "Method",
        "Ground Truth (%)",
        "Predicted (%)",
        "Absolute Error"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in rows:

        writer.writerow({
            "Image": row["Image"],
            "Method": row["Method"],
            "Ground Truth (%)":
                f"{row['Ground Truth (%)']:.2f}",
            "Predicted (%)":
                f"{row['Predicted (%)']:.2f}",
            "Absolute Error":
                f"{row['Absolute Error']:.2f}"
        })


print("\nResults saved to:")
print(csv_path)