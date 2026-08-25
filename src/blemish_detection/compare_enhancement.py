import cv2
import csv
from pathlib import Path

from fruit_mask import create_banana_mask
from otsu import otsu_segmentation
from morphology import morphological_enhancement
from evaluation import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_folder = (
    PROJECT_ROOT
    / "data"
    / "quality"
    / "defect"
)

ground_truth_folder = (
    PROJECT_ROOT
    / "data"
    / "quality"
    / "ground_truth"
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
    / "otsu_enhancement_comparison.csv"
)


# -----------------------------------------
# Get images
# -----------------------------------------

image_files = (
    list(image_folder.glob("*.jpg"))
    + list(image_folder.glob("*.jpeg"))
    + list(image_folder.glob("*.png"))
)


# -----------------------------------------
# Methods
# -----------------------------------------

methods = [
    "Original Otsu",
    "Otsu + Morphology 3x3",
    "Otsu + Morphology 5x5",
    "Otsu + Morphology 7x7"
]

totals = {}

counts = {}

for method in methods:

    totals[method] = {
        "IoU": 0,
        "Dice": 0,
        "Precision": 0,
        "Recall": 0
    }

    counts[method] = 0


all_results = []


# -----------------------------------------
# Evaluate images
# -----------------------------------------

for image_path in image_files:

    gt_path = (
        ground_truth_folder
        / f"{image_path.stem}_mask.png"
    )

    if not gt_path.exists():
        continue

    image = cv2.imread(
        str(image_path)
    )

    ground_truth = cv2.imread(
        str(gt_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or ground_truth is None:
        continue

    image = cv2.resize(
        image,
        (600, 600)
    )

    ground_truth = cv2.resize(
        ground_truth,
        (600, 600),
        interpolation=cv2.INTER_NEAREST
    )

    banana_mask = create_banana_mask(
        image
    )

    # Original Otsu
    otsu_mask = otsu_segmentation(
        image,
        banana_mask
    )

    masks = {
        "Original Otsu": otsu_mask,

        "Otsu + Morphology 3x3":
            morphological_enhancement(
                otsu_mask,
                3
            ),

        "Otsu + Morphology 5x5":
            morphological_enhancement(
                otsu_mask,
                5
            ),

        "Otsu + Morphology 7x7":
            morphological_enhancement(
                otsu_mask,
                7
            )
    }

    # -----------------------------------------
    # Calculate metrics
    # -----------------------------------------

    for method, mask in masks.items():

        metrics = calculate_metrics(
            mask,
            ground_truth
        )

        all_results.append({
            "Image": image_path.name,
            "Method": method,
            **metrics
        })

        for metric in totals[method]:

            totals[method][metric] += (
                metrics[metric]
            )

        counts[method] += 1


# -----------------------------------------
# Calculate averages
# -----------------------------------------

average_results = {}

print("\n======================================")
print("OTSU ENHANCEMENT COMPARISON")
print("======================================")

for method in methods:

    average_results[method] = {}

    print(f"\n{method}")

    for metric in totals[method]:

        average = (
            totals[method][metric]
            / counts[method]
        )

        average_results[method][metric] = (
            average
        )

        print(
            f"{metric}: {average:.4f}"
        )


# -----------------------------------------
# Save CSV
# -----------------------------------------

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "Image",
        "Method",
        "IoU",
        "Dice",
        "Precision",
        "Recall"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in all_results:

        writer.writerow({
            "Image": result["Image"],
            "Method": result["Method"],
            "IoU": f"{result['IoU']:.4f}",
            "Dice": f"{result['Dice']:.4f}",
            "Precision": f"{result['Precision']:.4f}",
            "Recall": f"{result['Recall']:.4f}"
        })

    writer.writerow({})

    for method, metrics in average_results.items():

        writer.writerow({
            "Image": "AVERAGE",
            "Method": method,
            "IoU": f"{metrics['IoU']:.4f}",
            "Dice": f"{metrics['Dice']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}"
        })


print("\nResults saved to:")
print(csv_path)