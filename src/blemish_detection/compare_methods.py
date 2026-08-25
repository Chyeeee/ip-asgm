import cv2
import csv
from pathlib import Path

from fruit_mask import create_banana_mask
from otsu import otsu_segmentation
from adaptive import adaptive_segmentation
from colour_segmentation import colour_segmentation
from evaluation import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_folder = PROJECT_ROOT / "data" / "quality" / "defect"
ground_truth_folder = PROJECT_ROOT / "data" / "quality" / "ground_truth"

results_folder = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
)

results_folder.mkdir(parents=True, exist_ok=True)

csv_path = results_folder / "baseline_comparison.csv"


# --------------------------------------------------
# Get all defect images
# --------------------------------------------------

image_files = (
    list(image_folder.glob("*.jpg"))
    + list(image_folder.glob("*.jpeg"))
    + list(image_folder.glob("*.png"))
)

if not image_files:
    print("No defect images found.")
    exit()


# --------------------------------------------------
# Store results
# --------------------------------------------------

all_results = []

method_totals = {
    "Otsu": {
        "IoU": 0,
        "Dice": 0,
        "Precision": 0,
        "Recall": 0
    },

    "Adaptive": {
        "IoU": 0,
        "Dice": 0,
        "Precision": 0,
        "Recall": 0
    },

    "Colour-Based": {
        "IoU": 0,
        "Dice": 0,
        "Precision": 0,
        "Recall": 0
    }
}

method_counts = {
    "Otsu": 0,
    "Adaptive": 0,
    "Colour-Based": 0
}


# --------------------------------------------------
# Evaluate annotated images
# --------------------------------------------------

for image_path in image_files:

    ground_truth_path = (
        ground_truth_folder
        / f"{image_path.stem}_mask.png"
    )

    # Skip images without ground truth
    if not ground_truth_path.exists():
        continue

    image = cv2.imread(str(image_path))

    ground_truth = cv2.imread(
        str(ground_truth_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or ground_truth is None:
        print("Could not load:", image_path.name)
        continue

    # Same size used during annotation
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
    # Banana ROI
    # --------------------------------------------------

    banana_mask = create_banana_mask(image)

    # --------------------------------------------------
    # Run segmentation methods
    # --------------------------------------------------

    masks = {
        "Otsu": otsu_segmentation(
            image,
            banana_mask
        ),

        "Adaptive": adaptive_segmentation(
            image,
            banana_mask
        ),

        "Colour-Based": colour_segmentation(
            image,
            banana_mask
        )
    }

    # --------------------------------------------------
    # Evaluate each method
    # --------------------------------------------------

    print("\n================================")
    print("Image:", image_path.name)
    print("================================")

    for method_name, predicted_mask in masks.items():

        metrics = calculate_metrics(
            predicted_mask,
            ground_truth
        )

        print(f"\n{method_name}")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        # Save individual result
        all_results.append({
            "Image": image_path.name,
            "Method": method_name,
            "IoU": metrics["IoU"],
            "Dice": metrics["Dice"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"]
        })

        # Add to totals
        for metric_name in method_totals[method_name]:
            method_totals[method_name][metric_name] += (
                metrics[metric_name]
            )

        method_counts[method_name] += 1


# --------------------------------------------------
# Make sure at least one image was evaluated
# --------------------------------------------------

if not all_results:
    print("\nNo annotated images were found.")
    print(
        "Make sure your masks are inside:",
        ground_truth_folder
    )
    exit()


# --------------------------------------------------
# Calculate averages
# --------------------------------------------------

average_results = {}

for method_name, totals in method_totals.items():

    count = method_counts[method_name]

    average_results[method_name] = {}

    for metric_name, total in totals.items():

        if count > 0:
            average = total / count
        else:
            average = 0

        average_results[method_name][metric_name] = average


# --------------------------------------------------
# Print average results
# --------------------------------------------------

print("\n")
print("========================================")
print("AVERAGE RESULTS")
print("========================================")

for method_name, metrics in average_results.items():

    print(f"\n{method_name}")

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


# --------------------------------------------------
# Save results to CSV
# --------------------------------------------------

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "Image",
        "Method",
        "IoU",
        "Dice",
        "Precision",
        "Recall"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    # Individual image results
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
    for method_name, metrics in average_results.items():

        writer.writerow({
            "Image": "AVERAGE",
            "Method": method_name,
            "IoU": f"{metrics['IoU']:.4f}",
            "Dice": f"{metrics['Dice']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}"
        })


print("\n========================================")
print("Results saved successfully!")
print("CSV:", csv_path)
print("========================================")