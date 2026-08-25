import cv2
import csv
from pathlib import Path

from fruit_mask import create_banana_mask
from otsu import otsu_segmentation
from morphology import morphological_enhancement
from hybrid import hybrid_segmentation
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
    / "hybrid_comparison.csv"
)


# --------------------------------------------------
# Get images
# --------------------------------------------------

image_files = (
    list(image_folder.glob("*.jpg"))
    + list(image_folder.glob("*.jpeg"))
    + list(image_folder.glob("*.png"))
)

if not image_files:
    print("No images found.")
    exit()


# --------------------------------------------------
# Methods
# --------------------------------------------------

methods = [
    "Original Otsu",
    "Otsu + Morphology 5x5",
    "Hybrid"
]

totals = {
    method: {
        "IoU": 0,
        "Dice": 0,
        "Precision": 0,
        "Recall": 0
    }
    for method in methods
}

counts = {
    method: 0
    for method in methods
}

all_results = []


# --------------------------------------------------
# Evaluate all annotated images
# --------------------------------------------------

for image_path in image_files:

    ground_truth_path = (
        ground_truth_folder
        / f"{image_path.stem}_mask.png"
    )

    # Skip images without ground truth
    if not ground_truth_path.exists():
        continue

    image = cv2.imread(
        str(image_path)
    )

    ground_truth = cv2.imread(
        str(ground_truth_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or ground_truth is None:
        continue

    # Same size used for annotation
    image = cv2.resize(
        image,
        (600, 600)
    )

    ground_truth = cv2.resize(
        ground_truth,
        (600, 600),
        interpolation=cv2.INTER_NEAREST
    )


    # --------------------------------------------------
    # Fruit ROI
    # --------------------------------------------------

    banana_mask = create_banana_mask(
        image
    )


    # --------------------------------------------------
    # Original Otsu
    # --------------------------------------------------

    otsu_mask = otsu_segmentation(
        image,
        banana_mask
    )


    # --------------------------------------------------
    # Otsu + 5x5 morphology
    # --------------------------------------------------

    morphology_mask = morphological_enhancement(
        otsu_mask,
        kernel_size=5
    )

    morphology_mask = cv2.bitwise_and(
        morphology_mask,
        banana_mask
    )


    # --------------------------------------------------
    # Hybrid
    # --------------------------------------------------

    hybrid_mask = hybrid_segmentation(
        image,
        banana_mask
    )


    masks = {
        "Original Otsu": otsu_mask,
        "Otsu + Morphology 5x5": morphology_mask,
        "Hybrid": hybrid_mask
    }


    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    for method, predicted_mask in masks.items():

        metrics = calculate_metrics(
            predicted_mask,
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


# --------------------------------------------------
# Check results
# --------------------------------------------------

if not all_results:
    print("No annotated images found.")
    exit()


# --------------------------------------------------
# Calculate average
# --------------------------------------------------

average_results = {}

print("\n======================================")
print("HYBRID ENHANCEMENT COMPARISON")
print("======================================")

for method in methods:

    average_results[method] = {}

    print(f"\n{method}")

    for metric in totals[method]:

        if counts[method] > 0:

            average = (
                totals[method][metric]
                / counts[method]
            )

        else:
            average = 0

        average_results[method][metric] = (
            average
        )

        print(
            f"{metric}: {average:.4f}"
        )


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

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

    # Individual results
    for result in all_results:

        writer.writerow({
            "Image": result["Image"],
            "Method": result["Method"],
            "IoU": f"{result['IoU']:.4f}",
            "Dice": f"{result['Dice']:.4f}",
            "Precision": f"{result['Precision']:.4f}",
            "Recall": f"{result['Recall']:.4f}"
        })

    # Blank row
    writer.writerow({})

    # Average results
    for method, metrics in average_results.items():

        writer.writerow({
            "Image": "AVERAGE",
            "Method": method,
            "IoU": f"{metrics['IoU']:.4f}",
            "Dice": f"{metrics['Dice']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}"
        })


print("\n======================================")
print("Results saved successfully!")
print("CSV:", csv_path)
print("======================================")