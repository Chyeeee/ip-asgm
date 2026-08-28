"""
======================================================================
END-TO-END FRUIT QUALITY ASSESSMENT SYSTEM EVALUATION
======================================================================

Evaluates the COMPLETE prediction pipeline:

    Image
      -> Preprocessing
      -> ROI extraction
      -> Colour + Texture + Shape features
      -> Fruit classification
      -> Ripeness / Guava quality classification
      -> AWDP surface damage

The script evaluates:
1. Fruit classification accuracy
2. Ripeness classification accuracy
3. Guava quality classification accuracy
4. Combined fruit + condition accuracy
5. Average prediction confidence
6. Per-fruit performance
7. Per-category performance
8. Confusion matrices
9. Low-confidence predictions
10. Failed images

IMPORTANT:
This evaluates the integrated system, rather than directly calling
the trained classifier on an existing feature CSV.
======================================================================
"""

import sys
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_DIR = PROJECT_ROOT / "src" / "system"

if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))


# ======================================================================
# IMPORT EXISTING PIPELINE
# ======================================================================

from prediction_pipeline import (
    load_models,
    predict_image,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

# Number of images tested for EACH fruit/category combination.
SAMPLES_PER_CATEGORY = 10

# Reproducible sampling
RANDOM_SEED = 42

# Same confidence threshold used by video system
FRUIT_CONFIDENCE_THRESHOLD = 40.0

# Output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "system"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# DATASET LOCATION
# ======================================================================

# We first try the processed images because these are known to exist
# in your project.
PROCESSED_ROOT = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "MedianFinal"
    / "ProcessedImages"
)


# ======================================================================
# SUPPORTED LABELS
# ======================================================================

STANDARD_FRUITS = [
    "Apple",
    "Banana",
    "Grape",
    "Mango",
    "Melon",
    "Orange",
    "Peach",
    "Pear",
]

STANDARD_CATEGORIES = [
    "Unripe",
    "Ripe",
    "Overripe",
    "Rotten",
]

GUAVA_CATEGORIES = [
    "Class_A",
    "Class_B",
    "Defect",
]

ALL_FRUITS = [
    "Apple",
    "Banana",
    "Grape",
    "Guava",
    "Mango",
    "Melon",
    "Orange",
    "Peach",
    "Pear",
]


# ======================================================================
# PRINT SECTION
# ======================================================================

def print_section(title):

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ======================================================================
# FIND IMAGE FILES
# ======================================================================

def find_images(folder):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    images = []

    if not folder.exists():
        return images

    for path in folder.iterdir():

        if (
            path.is_file()
            and path.suffix.lower() in extensions
        ):
            images.append(path)

    return sorted(images)


# ======================================================================
# DISCOVER DATASET
# ======================================================================

def discover_dataset():

    print_section(
        "DATASET DISCOVERY"
    )

    print(
        f"Processed image root:\n"
        f"{PROCESSED_ROOT}\n"
    )

    if not PROCESSED_ROOT.exists():

        raise FileNotFoundError(
            "ProcessedImages folder was not found:\n"
            f"{PROCESSED_ROOT}"
        )

    samples = []

    # ------------------------------------------------------------------
    # STANDARD FRUITS
    # ------------------------------------------------------------------

    for fruit in STANDARD_FRUITS:

        for category in STANDARD_CATEGORIES:

            folder = (
                PROCESSED_ROOT
                / fruit
                / category
            )

            images = find_images(
                folder
            )

            if not images:

                print(
                    f"[WARNING] No images: "
                    f"{fruit}/{category}"
                )

                continue

            for image_path in images:

                samples.append(
                    {
                        "fruit": fruit,
                        "category": category,
                        "image_path": image_path,
                    }
                )

    # ------------------------------------------------------------------
    # GUAVA
    # ------------------------------------------------------------------

    for category in GUAVA_CATEGORIES:

        folder = (
            PROCESSED_ROOT
            / "Guava"
            / category
        )

        images = find_images(
            folder
        )

        if not images:

            print(
                f"[WARNING] No images: "
                f"Guava/{category}"
            )

            continue

        for image_path in images:

            samples.append(
                {
                    "fruit": "Guava",
                    "category": category,
                    "image_path": image_path,
                }
            )

    print(
        f"\nTotal available images: "
        f"{len(samples)}"
    )

    return samples


# ======================================================================
# SAMPLE DATASET
# ======================================================================

def sample_dataset(
    samples,
    samples_per_category
):

    random.seed(
        RANDOM_SEED
    )

    grouped = defaultdict(list)

    for sample in samples:

        key = (
            sample["fruit"],
            sample["category"],
        )

        grouped[key].append(
            sample
        )

    selected = []

    print_section(
        "EVALUATION SAMPLE SELECTION"
    )

    for key in sorted(grouped):

        fruit, category = key

        group = grouped[key]

        number_to_select = min(
            samples_per_category,
            len(group)
        )

        chosen = random.sample(
            group,
            number_to_select
        )

        selected.extend(
            chosen
        )

        print(
            f"{fruit:<10} "
            f"{category:<12} "
            f"{number_to_select:>3} / "
            f"{len(group):>3}"
        )

    random.shuffle(
        selected
    )

    print(
        f"\nTotal evaluation images: "
        f"{len(selected)}"
    )

    return selected


# ======================================================================
# NORMALIZE CONDITION LABEL
# ======================================================================

def normalize_label(value):

    if value is None:
        return ""

    return str(value).strip()


# ======================================================================
# EVALUATE ONE IMAGE
# ======================================================================

def evaluate_one(
    sample,
    models
):

    actual_fruit = (
        sample["fruit"]
    )

    actual_category = (
        sample["category"]
    )

    image_path = (
        sample["image_path"]
    )

    start = time.perf_counter()

    # IMPORTANT:
    # save_results=False prevents hundreds of masks/images
    # being written during evaluation.

    result = predict_image(
        image_path,
        models=models,
        save_results=False
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    predicted_fruit = normalize_label(
        result["fruit"]
    )

    fruit_confidence = float(
        result["fruit_confidence"]
    )

    predicted_condition = normalize_label(
        result["condition"]
    )

    condition_confidence = float(
        result["condition_confidence"]
    )

    condition_type = normalize_label(
        result["condition_type"]
    )

    damage_percentage = float(
        result["damage_percentage"]
    )

    damage_level = normalize_label(
        result["damage_level"]
    )

    raw_damage_percentage = float(
        result.get(
            "raw_damage_percentage",
            np.nan
        )
    )

    # ------------------------------------------------------------------
    # CORRECTNESS
    # ------------------------------------------------------------------

    fruit_correct = (
        predicted_fruit.lower()
        == actual_fruit.lower()
    )

    condition_correct = (
        predicted_condition.lower()
        == actual_category.lower()
    )

    overall_correct = (
        fruit_correct
        and condition_correct
    )

    low_confidence = (
        fruit_confidence
        < FRUIT_CONFIDENCE_THRESHOLD
    )

    return {
        "image":
            image_path.name,

        "image_path":
            str(image_path),

        "actual_fruit":
            actual_fruit,

        "predicted_fruit":
            predicted_fruit,

        "fruit_confidence":
            fruit_confidence,

        "fruit_correct":
            fruit_correct,

        "low_fruit_confidence":
            low_confidence,

        "actual_category":
            actual_category,

        "predicted_condition":
            predicted_condition,

        "condition_type":
            condition_type,

        "condition_confidence":
            condition_confidence,

        "condition_correct":
            condition_correct,

        "overall_correct":
            overall_correct,

        "raw_damage_percentage":
            raw_damage_percentage,

        "damage_percentage":
            damage_percentage,

        "damage_level":
            damage_level,

        "processing_time_seconds":
            elapsed,
    }


# ======================================================================
# CALCULATE CLASSIFICATION METRICS
# ======================================================================

def calculate_metrics(
    actual,
    predicted
):

    accuracy = accuracy_score(
        actual,
        predicted
    )

    (
        precision,
        recall,
        f1,
        _
    ) = precision_recall_fscore_support(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ======================================================================
# SAVE CONFUSION MATRIX
# ======================================================================

def save_confusion_matrix(
    actual,
    predicted,
    labels,
    filename
):

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual_{x}"
            for x in labels
        ],
        columns=[
            f"Predicted_{x}"
            for x in labels
        ]
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    matrix_df.to_csv(
        output_path
    )

    return output_path


# ======================================================================
# PRINT FRUIT PERFORMANCE
# ======================================================================

def print_fruit_performance(
    df
):

    print_section(
        "FRUIT CLASSIFICATION PERFORMANCE"
    )

    metrics = calculate_metrics(
        df["actual_fruit"],
        df["predicted_fruit"]
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"\nAverage Fruit Confidence : "
        f"{df['fruit_confidence'].mean():.2f}%"
    )

    print(
        f"Low-confidence predictions (<"
        f"{FRUIT_CONFIDENCE_THRESHOLD:.0f}%): "
        f"{df['low_fruit_confidence'].sum()} / "
        f"{len(df)}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            df["actual_fruit"],
            df["predicted_fruit"],
            zero_division=0,
            digits=4
        )
    )

    return metrics


# ======================================================================
# CONDITION PERFORMANCE
# ======================================================================

def print_condition_performance(
    df
):

    # ------------------------------------------------------------------
    # STANDARD RIPENESS
    # ------------------------------------------------------------------

    standard_df = df[
        df["actual_fruit"]
        != "Guava"
    ].copy()

    print_section(
        "RIPENESS CLASSIFICATION PERFORMANCE"
    )

    if len(standard_df) > 0:

        metrics = calculate_metrics(
            standard_df[
                "actual_category"
            ],
            standard_df[
                "predicted_condition"
            ]
        )

        print(
            f"Images    : "
            f"{len(standard_df)}"
        )

        print(
            f"Accuracy  : "
            f"{metrics['accuracy']:.4f} "
            f"({metrics['accuracy'] * 100:.2f}%)"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"\nAverage Ripeness Confidence : "
            f"{standard_df['condition_confidence'].mean():.2f}%"
        )

    else:

        metrics = None

        print(
            "No standard ripeness images "
            "were evaluated."
        )

    # ------------------------------------------------------------------
    # GUAVA
    # ------------------------------------------------------------------

    guava_df = df[
        df["actual_fruit"]
        == "Guava"
    ].copy()

    print_section(
        "GUAVA QUALITY CLASSIFICATION PERFORMANCE"
    )

    if len(guava_df) > 0:

        guava_metrics = calculate_metrics(
            guava_df[
                "actual_category"
            ],
            guava_df[
                "predicted_condition"
            ]
        )

        print(
            f"Images    : "
            f"{len(guava_df)}"
        )

        print(
            f"Accuracy  : "
            f"{guava_metrics['accuracy']:.4f} "
            f"({guava_metrics['accuracy'] * 100:.2f}%)"
        )

        print(
            f"Precision : "
            f"{guava_metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{guava_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{guava_metrics['f1']:.4f}"
        )

        print(
            f"\nAverage Quality Confidence : "
            f"{guava_df['condition_confidence'].mean():.2f}%"
        )

    else:

        guava_metrics = None

        print(
            "No Guava images were evaluated."
        )

    return (
        metrics,
        guava_metrics
    )


# ======================================================================
# PER-FRUIT SUMMARY
# ======================================================================

def create_per_fruit_summary(
    df
):

    rows = []

    for fruit in sorted(
        df["actual_fruit"].unique()
    ):

        subset = df[
            df["actual_fruit"]
            == fruit
        ]

        fruit_accuracy = (
            subset[
                "fruit_correct"
            ].mean()
        )

        condition_accuracy = (
            subset[
                "condition_correct"
            ].mean()
        )

        overall_accuracy = (
            subset[
                "overall_correct"
            ].mean()
        )

        rows.append(
            {
                "fruit":
                    fruit,

                "samples":
                    len(subset),

                "fruit_accuracy":
                    fruit_accuracy,

                "condition_accuracy":
                    condition_accuracy,

                "overall_accuracy":
                    overall_accuracy,

                "avg_fruit_confidence":
                    subset[
                        "fruit_confidence"
                    ].mean(),

                "avg_condition_confidence":
                    subset[
                        "condition_confidence"
                    ].mean(),

                "avg_surface_damage":
                    subset[
                        "damage_percentage"
                    ].mean(),

                "avg_processing_time_seconds":
                    subset[
                        "processing_time_seconds"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# PER-CATEGORY SUMMARY
# ======================================================================

def create_per_category_summary(
    df
):

    rows = []

    grouped = df.groupby(
        [
            "actual_fruit",
            "actual_category"
        ]
    )

    for (
        fruit,
        category
    ), subset in grouped:

        rows.append(
            {
                "fruit":
                    fruit,

                "category":
                    category,

                "samples":
                    len(subset),

                "fruit_accuracy":
                    subset[
                        "fruit_correct"
                    ].mean(),

                "condition_accuracy":
                    subset[
                        "condition_correct"
                    ].mean(),

                "overall_accuracy":
                    subset[
                        "overall_correct"
                    ].mean(),

                "avg_fruit_confidence":
                    subset[
                        "fruit_confidence"
                    ].mean(),

                "avg_condition_confidence":
                    subset[
                        "condition_confidence"
                    ].mean(),

                "avg_damage_percentage":
                    subset[
                        "damage_percentage"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# PRINT SUMMARY
# ======================================================================

def print_summary(
    df,
    per_fruit
):

    print_section(
        "END-TO-END SYSTEM SUMMARY"
    )

    total = len(df)

    fruit_correct = int(
        df["fruit_correct"].sum()
    )

    condition_correct = int(
        df["condition_correct"].sum()
    )

    overall_correct = int(
        df["overall_correct"].sum()
    )

    print(
        f"Images evaluated           : "
        f"{total}"
    )

    print(
        f"Fruit correct              : "
        f"{fruit_correct}/{total} "
        f"({fruit_correct / total * 100:.2f}%)"
    )

    print(
        f"Condition correct          : "
        f"{condition_correct}/{total} "
        f"({condition_correct / total * 100:.2f}%)"
    )

    print(
        f"Fruit + Condition correct  : "
        f"{overall_correct}/{total} "
        f"({overall_correct / total * 100:.2f}%)"
    )

    print(
        f"\nAverage fruit confidence   : "
        f"{df['fruit_confidence'].mean():.2f}%"
    )

    print(
        f"Average condition confidence: "
        f"{df['condition_confidence'].mean():.2f}%"
    )

    print(
        f"Average surface damage     : "
        f"{df['damage_percentage'].mean():.2f}%"
    )

    print(
        f"Average processing time    : "
        f"{df['processing_time_seconds'].mean():.4f} sec/image"
    )

    print(
        "\nPER-FRUIT SUMMARY"
    )

    display = per_fruit.copy()

    percentage_columns = [
        "fruit_accuracy",
        "condition_accuracy",
        "overall_accuracy",
    ]

    for column in percentage_columns:

        display[column] = (
            display[column]
            * 100
        ).round(2)

    display[
        "avg_fruit_confidence"
    ] = display[
        "avg_fruit_confidence"
    ].round(2)

    display[
        "avg_condition_confidence"
    ] = display[
        "avg_condition_confidence"
    ].round(2)

    display[
        "avg_surface_damage"
    ] = display[
        "avg_surface_damage"
    ].round(2)

    display[
        "avg_processing_time_seconds"
    ] = display[
        "avg_processing_time_seconds"
    ].round(4)

    print(
        display.to_string(
            index=False
        )
    )


# ======================================================================
# MAIN EVALUATION
# ======================================================================

def run_evaluation():

    print_section(
        "END-TO-END FRUIT QUALITY ASSESSMENT EVALUATION"
    )

    print(
        f"Samples/category : "
        f"{SAMPLES_PER_CATEGORY}"
    )

    print(
        f"Random seed      : "
        f"{RANDOM_SEED}"
    )

    print(
        f"Confidence limit : "
        f"{FRUIT_CONFIDENCE_THRESHOLD:.1f}%"
    )

    # ------------------------------------------------------------------
    # DISCOVER + SAMPLE
    # ------------------------------------------------------------------

    all_samples = (
        discover_dataset()
    )

    selected_samples = (
        sample_dataset(
            all_samples,
            SAMPLES_PER_CATEGORY
        )
    )

    if not selected_samples:

        raise RuntimeError(
            "No evaluation images "
            "were selected."
        )

    # ------------------------------------------------------------------
    # LOAD MODELS ONCE
    # ------------------------------------------------------------------

    print_section(
        "LOADING MODELS"
    )

    models = load_models()

    print(
        "All models loaded successfully."
    )

    # ------------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------------

    print_section(
        "RUNNING END-TO-END EVALUATION"
    )

    results = []

    failures = []

    total = len(
        selected_samples
    )

    evaluation_start = (
        time.perf_counter()
    )

    for index, sample in enumerate(
        selected_samples,
        start=1
    ):

        try:

            result = evaluate_one(
                sample,
                models
            )

            results.append(
                result
            )

            fruit_status = (
                "OK"
                if result["fruit_correct"]
                else "WRONG"
            )

            condition_status = (
                "OK"
                if result["condition_correct"]
                else "WRONG"
            )

            print(
                f"[{index:03d}/{total:03d}] "
                f"{sample['image_path'].name:<30} "
                f"Fruit={result['predicted_fruit']:<8} "
                f"{fruit_status:<5} | "
                f"Condition={result['predicted_condition']:<10} "
                f"{condition_status:<5}"
            )

        except Exception as error:

            failures.append(
                {
                    "image":
                        sample[
                            "image_path"
                        ].name,

                    "image_path":
                        str(
                            sample[
                                "image_path"
                            ]
                        ),

                    "actual_fruit":
                        sample["fruit"],

                    "actual_category":
                        sample["category"],

                    "error_type":
                        type(error).__name__,

                    "error":
                        str(error),
                }
            )

            print(
                f"[{index:03d}/{total:03d}] "
                f"FAILED: "
                f"{sample['image_path'].name} "
                f"-> {error}"
            )

    total_time = (
        time.perf_counter()
        - evaluation_start
    )

    if not results:

        raise RuntimeError(
            "Every evaluation image failed."
        )

    # ------------------------------------------------------------------
    # DATAFRAME
    # ------------------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    fruit_metrics = (
        print_fruit_performance(
            df
        )
    )

    (
        ripeness_metrics,
        guava_metrics
    ) = print_condition_performance(
        df
    )

    # ------------------------------------------------------------------
    # SUMMARIES
    # ------------------------------------------------------------------

    per_fruit = (
        create_per_fruit_summary(
            df
        )
    )

    per_category = (
        create_per_category_summary(
            df
        )
    )

    print_summary(
        df,
        per_fruit
    )

    # ------------------------------------------------------------------
    # SAVE DETAILED RESULTS
    # ------------------------------------------------------------------

    detailed_path = (
        OUTPUT_DIR
        / "end_to_end_results.csv"
    )

    df.to_csv(
        detailed_path,
        index=False
    )

    per_fruit_path = (
        OUTPUT_DIR
        / "per_fruit_summary.csv"
    )

    per_fruit.to_csv(
        per_fruit_path,
        index=False
    )

    per_category_path = (
        OUTPUT_DIR
        / "per_category_summary.csv"
    )

    per_category.to_csv(
        per_category_path,
        index=False
    )

    # ------------------------------------------------------------------
    # WRONG PREDICTIONS
    # ------------------------------------------------------------------

    errors_df = df[
        ~df["overall_correct"]
    ].copy()

    errors_path = (
        OUTPUT_DIR
        / "incorrect_predictions.csv"
    )

    errors_df.to_csv(
        errors_path,
        index=False
    )

    # ------------------------------------------------------------------
    # LOW CONFIDENCE
    # ------------------------------------------------------------------

    low_confidence_df = df[
        df[
            "low_fruit_confidence"
        ]
    ].copy()

    low_confidence_path = (
        OUTPUT_DIR
        / "low_confidence_predictions.csv"
    )

    low_confidence_df.to_csv(
        low_confidence_path,
        index=False
    )

    # ------------------------------------------------------------------
    # FAILURES
    # ------------------------------------------------------------------

    failures_path = (
        OUTPUT_DIR
        / "failed_images.csv"
    )

    pd.DataFrame(
        failures
    ).to_csv(
        failures_path,
        index=False
    )

    # ------------------------------------------------------------------
    # FRUIT CONFUSION MATRIX
    # ------------------------------------------------------------------

    fruit_labels = sorted(
        set(
            df["actual_fruit"].tolist()
            + df["predicted_fruit"].tolist()
        )
    )

    fruit_cm_path = (
        save_confusion_matrix(
            df["actual_fruit"],
            df["predicted_fruit"],
            fruit_labels,
            "fruit_confusion_matrix.csv"
        )
    )

    # ------------------------------------------------------------------
    # RIPENESS CONFUSION MATRIX
    # ------------------------------------------------------------------

    standard_df = df[
        df["actual_fruit"]
        != "Guava"
    ]

    if len(standard_df) > 0:

        ripeness_labels = sorted(
            set(
                standard_df[
                    "actual_category"
                ].tolist()
                +
                standard_df[
                    "predicted_condition"
                ].tolist()
            )
        )

        save_confusion_matrix(
            standard_df[
                "actual_category"
            ],
            standard_df[
                "predicted_condition"
            ],
            ripeness_labels,
            "ripeness_confusion_matrix.csv"
        )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print_section(
        "EVALUATION COMPLETED"
    )

    print(
        f"Successfully evaluated : "
        f"{len(df)}"
    )

    print(
        f"Failed images          : "
        f"{len(failures)}"
    )

    print(
        f"Total evaluation time  : "
        f"{total_time:.2f} seconds"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFiles:"
    )

    print(
        "  end_to_end_results.csv"
    )

    print(
        "  per_fruit_summary.csv"
    )

    print(
        "  per_category_summary.csv"
    )

    print(
        "  incorrect_predictions.csv"
    )

    print(
        "  low_confidence_predictions.csv"
    )

    print(
        "  failed_images.csv"
    )

    print(
        "  fruit_confusion_matrix.csv"
    )

    print(
        "  ripeness_confusion_matrix.csv"
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    run_evaluation()