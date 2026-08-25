import cv2
import csv
import numpy as np
from pathlib import Path

from fruit_mask import create_banana_mask
from colour_segmentation import colour_segmentation


PROJECT_ROOT = Path(__file__).resolve().parents[2]

quality_folder = (
    PROJECT_ROOT
    / "data"
    / "quality"
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
    / "quality_class_comparison.csv"
)


# --------------------------------------------------
# Dataset classes
# Change folder names here if necessary
# --------------------------------------------------

classes = {
    "Class A": quality_folder / "class_a",
    "Class B": quality_folder / "class_b",
    "Defect": quality_folder / "defect"
}


all_results = []

class_percentages = {
    "Class A": [],
    "Class B": [],
    "Defect": []
}


# --------------------------------------------------
# Process each class
# --------------------------------------------------

for class_name, folder in classes.items():

    print("\n====================================")
    print("Processing:", class_name)
    print("====================================")

    if not folder.exists():
        print("Folder not found:", folder)
        continue

    image_files = (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
    )

    print("Images found:", len(image_files))

    for image_path in image_files:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                "Could not load:",
                image_path.name
            )
            continue

        image = cv2.resize(
            image,
            (600, 600)
        )

        # ----------------------------------------
        # Step 1: Fruit ROI
        # ----------------------------------------

        banana_mask = create_banana_mask(
            image
        )

        # ----------------------------------------
        # Step 2: Colour-based blemish detection
        # ----------------------------------------

        blemish_mask = colour_segmentation(
            image,
            banana_mask
        )

        blemish_mask = cv2.bitwise_and(
            blemish_mask,
            banana_mask
        )

        # ----------------------------------------
        # Step 3: Count pixels
        # ----------------------------------------

        fruit_pixels = cv2.countNonZero(
            banana_mask
        )

        blemish_pixels = cv2.countNonZero(
            blemish_mask
        )

        if fruit_pixels == 0:
            continue

        # ----------------------------------------
        # Step 4: Blemish %
        # ----------------------------------------

        blemish_percentage = (
            blemish_pixels / fruit_pixels
        ) * 100

        class_percentages[
            class_name
        ].append(
            blemish_percentage
        )

        all_results.append({
            "Image": image_path.name,
            "Class": class_name,
            "Fruit Pixels": fruit_pixels,
            "Blemish Pixels": blemish_pixels,
            "Blemish Percentage":
                blemish_percentage
        })


# --------------------------------------------------
# Display summary
# --------------------------------------------------

print("\n")
print("==========================================")
print("QUALITY CLASS COMPARISON")
print("==========================================")

summary_results = []

for class_name, percentages in class_percentages.items():

    if not percentages:
        print(
            f"\n{class_name}: No results"
        )
        continue

    percentages = np.array(
        percentages
    )

    mean_value = np.mean(percentages)
    median_value = np.median(percentages)
    std_value = np.std(percentages)
    minimum = np.min(percentages)
    maximum = np.max(percentages)

    summary_results.append({
        "Class": class_name,
        "Count": len(percentages),
        "Mean": mean_value,
        "Median": median_value,
        "Std": std_value,
        "Min": minimum,
        "Max": maximum
    })

    print(f"\n{class_name}")
    print(
        f"Number of images: "
        f"{len(percentages)}"
    )
    print(
        f"Mean blemish %: "
        f"{mean_value:.2f}%"
    )
    print(
        f"Median blemish %: "
        f"{median_value:.2f}%"
    )
    print(
        f"Standard deviation: "
        f"{std_value:.2f}"
    )
    print(
        f"Minimum: "
        f"{minimum:.2f}%"
    )
    print(
        f"Maximum: "
        f"{maximum:.2f}%"
    )


# --------------------------------------------------
# Save individual results
# --------------------------------------------------

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "Image",
        "Class",
        "Fruit Pixels",
        "Blemish Pixels",
        "Blemish Percentage"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in all_results:

        writer.writerow({
            "Image":
                result["Image"],

            "Class":
                result["Class"],

            "Fruit Pixels":
                result["Fruit Pixels"],

            "Blemish Pixels":
                result["Blemish Pixels"],

            "Blemish Percentage":
                f"{result['Blemish Percentage']:.2f}"
        })


print("\n==========================================")
print("Results saved successfully!")
print("CSV:", csv_path)
print("==========================================")