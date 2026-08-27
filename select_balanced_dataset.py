import os
import random
import shutil
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

# Change these if your folder names are different
DATASET1_FOLDER = "data/Dataset1"
DATASET2_FOLDER = "data/Dataset2"

OUTPUT_FOLDER = "BalancedDataset"

IMAGES_PER_CLASS = 100

# Fixed seed = same random images every time you run the script
RANDOM_SEED = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)

random.seed(RANDOM_SEED)


# ============================================================
# DATASET 1 CLASS NAMES
# ============================================================
# IMPORTANT:
# These must follow the SAME ORDER as the "names:" section
# inside your data.yaml.
#
# If your data.yaml order is different, replace this list
# with the exact order from data.yaml.
# ============================================================

CLASS_NAMES = [
    "Apple_Overripe",
    "Apple_Ripe",
    "Apple_Rotten",
    "Apple_Unripe",

    "Banana_Overripe",
    "Banana_Ripe",
    "Banana_Rotten",
    "Banana_Unripe",

    "Grape_Overripe",
    "Grape_Ripe",
    "Grape_Rotten",
    "Grape_Unripe",

    "Mango_Overripe",
    "Mango_Ripe",
    "Mango_Rotten",
    "Mango_Unripe",

    "Melon_Overripe",
    "Melon_Ripe",
    "Melon_Rotten",
    "Melon_Unripe",

    "Orange_Overripe",
    "Orange_Ripe",
    "Orange_Rotten",
    "Orange_Unripe",

    "Peach_Overripe",
    "Peach_Ripe",
    "Peach_Rotten",
    "Peach_Unripe",

    "Pear_Overripe",
    "Pear_Ripe",
    "Pear_Rotten",
    "Pear_Unripe"
]


# ============================================================
# HELPER: CREATE CLEAN OUTPUT FOLDER
# ============================================================

def prepare_output_folder():

    if os.path.exists(OUTPUT_FOLDER):

        print(
            f"\nWARNING: {OUTPUT_FOLDER} already exists."
        )

        answer = input(
            "Delete it and create a new balanced dataset? (y/n): "
        )

        if answer.lower() != "y":

            print("Selection cancelled.")
            exit()

        shutil.rmtree(OUTPUT_FOLDER)

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )


# ============================================================
# HELPER: SPLIT CLASS NAME
# ============================================================

def split_class_name(class_name):

    parts = class_name.split("_", 1)

    if len(parts) != 2:
        return class_name, "Unknown"

    fruit = parts[0]
    category = parts[1]

    return fruit, category


# ============================================================
# HELPER: FIND LABEL FILE
# ============================================================

def get_label_path(image_path):

    # Roboflow normally stores:
    #
    # train/images/example.jpg
    # train/labels/example.txt

    image_directory = os.path.dirname(
        image_path
    )

    filename = os.path.basename(
        image_path
    )

    filename_without_extension = os.path.splitext(
        filename
    )[0]

    # Replace final "images" folder with "labels"
    parent_directory = os.path.dirname(
        image_directory
    )

    label_directory = os.path.join(
        parent_directory,
        "labels"
    )

    label_path = os.path.join(
        label_directory,
        filename_without_extension + ".txt"
    )

    return label_path


# ============================================================
# 1. SCAN DATASET 1
# ============================================================

def scan_dataset1():

    print("\n======================================================")
    print("SCANNING DATASET 1")
    print("======================================================")

    class_images = defaultdict(list)

    total_images = 0
    no_label = 0
    multiple_classes = 0
    invalid_labels = 0

    # Dataset 1 normally has:
    # train/images
    # valid/images
    # test/images

    for split in ["train", "valid", "test"]:

        image_folder = os.path.join(
            DATASET1_FOLDER,
            split,
            "images"
        )

        if not os.path.exists(image_folder):

            print(
                f"Split not found: {image_folder}"
            )

            continue

        print(
            f"\nReading: {image_folder}"
        )

        files = sorted(
            os.listdir(image_folder)
        )

        for filename in files:

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            image_path = os.path.join(
                image_folder,
                filename
            )

            total_images += 1

            label_path = get_label_path(
                image_path
            )

            if not os.path.exists(label_path):

                no_label += 1
                continue

            class_ids = set()

            try:

                with open(
                    label_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    for line in file:

                        line = line.strip()

                        if not line:
                            continue

                        parts = line.split()

                        if len(parts) < 1:
                            continue

                        class_id = int(
                            parts[0]
                        )

                        if (
                            0 <= class_id
                            < len(CLASS_NAMES)
                        ):

                            class_ids.add(
                                class_id
                            )

                        else:

                            invalid_labels += 1

            except Exception as error:

                print(
                    f"Could not read {label_path}: {error}"
                )

                continue

            if len(class_ids) == 0:
                continue

            # ------------------------------------------------
            # IMPORTANT:
            # Only use images containing ONE class.
            #
            # This avoids putting an image containing
            # e.g. Apple Ripe + Apple Rotten into only one
            # classification folder.
            # ------------------------------------------------

            if len(class_ids) > 1:

                multiple_classes += 1
                continue

            class_id = next(
                iter(class_ids)
            )

            class_name = CLASS_NAMES[
                class_id
            ]

            class_images[
                class_name
            ].append(
                image_path
            )

    print("\n------------------------------------------------------")
    print("DATASET 1 SCAN COMPLETED")
    print("------------------------------------------------------")

    print(
        f"Images scanned             : {total_images}"
    )

    print(
        f"Images without label       : {no_label}"
    )

    print(
        f"Multi-class images skipped : {multiple_classes}"
    )

    print(
        f"Invalid label entries      : {invalid_labels}"
    )

    print("\nAvailable images per class:\n")

    for class_name in CLASS_NAMES:

        count = len(
            class_images[class_name]
        )

        print(
            f"{class_name:<25} : {count}"
        )

    return class_images


# ============================================================
# 2. SELECT DATASET 1 IMAGES
# ============================================================

def select_dataset1(class_images):

    print("\n======================================================")
    print("SELECTING DATASET 1")
    print("======================================================\n")

    total_selected = 0

    for class_name in CLASS_NAMES:

        available_images = class_images[
            class_name
        ]

        fruit, category = split_class_name(
            class_name
        )

        output_directory = os.path.join(
            OUTPUT_FOLDER,
            fruit,
            category
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        available_count = len(
            available_images
        )

        if available_count == 0:

            print(
                f"WARNING: {class_name} has 0 images."
            )

            continue

        number_to_select = min(
            IMAGES_PER_CLASS,
            available_count
        )

        if available_count < IMAGES_PER_CLASS:

            print(
                f"WARNING: {class_name} only has "
                f"{available_count} images."
            )

        selected_images = random.sample(
            available_images,
            number_to_select
        )

        for index, source_path in enumerate(
            selected_images,
            start=1
        ):

            extension = os.path.splitext(
                source_path
            )[1].lower()

            new_filename = (
                f"{fruit}_{category}_{index:03d}"
                f"{extension}"
            )

            destination_path = os.path.join(
                output_directory,
                new_filename
            )

            shutil.copy2(
                source_path,
                destination_path
            )

        total_selected += number_to_select

        print(
            f"{fruit:<10} "
            f"{category:<12} "
            f"{number_to_select:>3} images"
        )

    return total_selected


# ============================================================
# 3. FIND GUAVA CATEGORY FOLDERS
# ============================================================

def find_guava_folders():

    print("\n======================================================")
    print("SCANNING DATASET 2 - GUAVA")
    print("======================================================\n")

    detected_folders = {}

    # Names that may appear in Dataset 2
    category_keywords = {
        "Class_A": [
            "class a",
            "class_a",
            "classa",
            "a class"
        ],

        "Class_B": [
            "class b",
            "class_b",
            "classb",
            "b class"
        ],

        "Defect": [
            "defect",
            "defective",
            "defect class",
            "defect_class"
        ]
    }

    for root, folders, files in os.walk(
        DATASET2_FOLDER
    ):

        image_count = sum(
            1
            for filename in files
            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            )
        )

        if image_count == 0:
            continue

        folder_name = os.path.basename(
            root
        ).lower()

        normalized_name = (
            folder_name
            .replace("-", " ")
            .replace("_", " ")
            .strip()
        )

        for category, keywords in (
            category_keywords.items()
        ):

            for keyword in keywords:

                normalized_keyword = (
                    keyword
                    .replace("_", " ")
                    .replace("-", " ")
                    .strip()
                )

                if (
                    normalized_name
                    == normalized_keyword
                ):

                    detected_folders[
                        category
                    ] = root

                    print(
                        f"{category:<10}: "
                        f"{root} "
                        f"({image_count} images)"
                    )

                    break

    return detected_folders


# ============================================================
# 4. SELECT GUAVA IMAGES
# ============================================================

def select_guava():

    guava_folders = find_guava_folders()

    expected_categories = [
        "Class_A",
        "Class_B",
        "Defect"
    ]

    total_selected = 0

    print("\n======================================================")
    print("SELECTING GUAVA IMAGES")
    print("======================================================\n")

    for category in expected_categories:

        if category not in guava_folders:

            print(
                f"WARNING: Could not automatically find "
                f"Guava {category} folder."
            )

            continue

        source_folder = guava_folders[
            category
        ]

        images = []

        for filename in os.listdir(
            source_folder
        ):

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                images.append(
                    os.path.join(
                        source_folder,
                        filename
                    )
                )

        images.sort()

        available_count = len(
            images
        )

        number_to_select = min(
            IMAGES_PER_CLASS,
            available_count
        )

        if available_count < IMAGES_PER_CLASS:

            print(
                f"WARNING: Guava {category} only has "
                f"{available_count} images."
            )

        selected_images = random.sample(
            images,
            number_to_select
        )

        output_directory = os.path.join(
            OUTPUT_FOLDER,
            "Guava",
            category
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        for index, source_path in enumerate(
            selected_images,
            start=1
        ):

            extension = os.path.splitext(
                source_path
            )[1].lower()

            new_filename = (
                f"Guava_{category}_{index:03d}"
                f"{extension}"
            )

            destination_path = os.path.join(
                output_directory,
                new_filename
            )

            shutil.copy2(
                source_path,
                destination_path
            )

        total_selected += number_to_select

        print(
            f"Guava      "
            f"{category:<12} "
            f"{number_to_select:>3} images"
        )

    return total_selected


# ============================================================
# 5. VERIFY FINAL BALANCED DATASET
# ============================================================

def verify_dataset():

    print("\n======================================================")
    print("FINAL DATASET VERIFICATION")
    print("======================================================\n")

    grand_total = 0

    for fruit in sorted(
        os.listdir(OUTPUT_FOLDER)
    ):

        fruit_path = os.path.join(
            OUTPUT_FOLDER,
            fruit
        )

        if not os.path.isdir(
            fruit_path
        ):
            continue

        print(f"\n{fruit}")

        for category in sorted(
            os.listdir(fruit_path)
        ):

            category_path = os.path.join(
                fruit_path,
                category
            )

            if not os.path.isdir(
                category_path
            ):
                continue

            count = sum(
                1
                for filename in os.listdir(
                    category_path
                )
                if filename.lower().endswith(
                    IMAGE_EXTENSIONS
                )
            )

            grand_total += count

            status = (
                "OK"
                if count == IMAGES_PER_CLASS
                else "CHECK"
            )

            print(
                f"    {category:<15} "
                f"{count:>3} "
                f"[{status}]"
            )

    print("\n======================================================")
    print(
        f"TOTAL SELECTED IMAGES: {grand_total}"
    )

    print(
        f"EXPECTED MAXIMUM     : 3500"
    )

    print("======================================================")

    return grand_total


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n======================================================")
    print("       BALANCED FRUIT DATASET SELECTION")
    print("======================================================")

    print(
        f"Target: {IMAGES_PER_CLASS} images "
        f"per fruit/category"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print("======================================================")

    # Create empty BalancedDataset
    prepare_output_folder()

    # Dataset 1
    dataset1_images = scan_dataset1()

    dataset1_total = select_dataset1(
        dataset1_images
    )

    # Dataset 2
    dataset2_total = select_guava()

    # Verify
    final_total = verify_dataset()

    print("\n======================================================")
    print("SELECTION COMPLETED")
    print("======================================================")

    print(
        f"Dataset 1 selected : {dataset1_total}"
    )

    print(
        f"Dataset 2 selected : {dataset2_total}"
    )

    print(
        f"Final total        : {final_total}"
    )

    print(
        f"\nBalanced dataset saved to:\n"
        f"{os.path.abspath(OUTPUT_FOLDER)}"
    )

    print("======================================================\n")


if __name__ == "__main__":
    main()