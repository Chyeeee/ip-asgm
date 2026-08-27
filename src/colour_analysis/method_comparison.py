from time import perf_counter

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

from data_loader import (
    get_dataset_records,
    create_per_fruit_sample,
    load_image_mask,
)

from colour_features import (
    extract_basic_features,
    get_basic_feature_names,
)

from evaluation import (
    evaluate_features,
)


# ============================================================
# OUTPUT FILES
# ============================================================

SAMPLE_FILE = (
    OUTPUT_DIR
    / "comparison_sample.csv"
)

PER_FRUIT_RESULT_FILE = (
    OUTPUT_DIR
    / "per_fruit_method_comparison.csv"
)

OVERALL_RESULT_FILE = (
    OUTPUT_DIR
    / "method_comparison.csv"
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_colour_features(
    fruit_df,
    colour_space,
):

    features = []
    labels = []
    metadata = []
    processing_times = []

    total = len(fruit_df)

    for position, (_, row) in enumerate(
        fruit_df.iterrows(),
        start=1,
    ):

        try:

            image, mask = load_image_mask(
                row["image_path"],
                row["mask_path"],
            )

            start = perf_counter()

            feature = extract_basic_features(
                image,
                mask,
                colour_space,
            )

            elapsed = (
                perf_counter()
                - start
            )

            if feature is None:

                print(
                    f"Empty ROI skipped: "
                    f"{row['image_path']}"
                )

                continue

            features.append(
                feature
            )

            labels.append(
                row["category"]
            )

            processing_times.append(
                elapsed
            )

            metadata.append({
                "fruit":
                    row["fruit"],

                "category":
                    row["category"],

                "image":
                    row[
                        "image_path"
                    ].name,
            })

        except Exception as error:

            print(
                f"\nError processing "
                f"{row['image_path']}:\n"
                f"{error}"
            )

        if (
            position % 20 == 0
            or
            position == total
        ):

            print(
                f"  {colour_space}: "
                f"{position}/{total}"
            )

    if not features:

        raise RuntimeError(
            f"No valid {colour_space} "
            f"features extracted."
        )

    average_time_ms = (
        np.mean(
            processing_times
        )
        * 1000
    )

    return (
        np.asarray(features),
        np.asarray(labels),
        metadata,
        average_time_ms,
    )


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

def create_feature_dataframe(
    features,
    metadata,
    colour_space,
):

    feature_names = (
        get_basic_feature_names(
            colour_space
        )
    )

    metadata_df = pd.DataFrame(
        metadata
    )

    feature_df = pd.DataFrame(
        features,
        columns=feature_names,
    )

    return pd.concat(
        [
            metadata_df,
            feature_df,
        ],
        axis=1,
    )


# ============================================================
# DISPLAY FRUIT RESULT
# ============================================================

def display_fruit_result(
    fruit,
    fruit_results,
):

    result_df = pd.DataFrame(
        fruit_results
    )

    result_df = result_df.sort_values(
        by=[
            "F1_Score",
            "Accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    )

    display_df = result_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
    ]:

        display_df[column] *= 100

    print("\n")
    print("-" * 90)
    print(
        f"{fruit.upper()} RESULTS"
    )
    print("-" * 90)

    print(
        display_df[
            [
                "Method",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "Fisher_Separability",
                "Avg_Processing_Time_ms",
            ]
        ].to_string(
            index=False
        )
    )

    best = result_df.iloc[0]

    print(
        f"\nBest for {fruit}: "
        f"{best['Method']} "
        f"(Macro F1 = "
        f"{best['F1_Score'] * 100:.2f}%)"
    )


# ============================================================
# OVERALL AVERAGE
# ============================================================

def calculate_overall_results(
    per_fruit_df,
):

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "Fisher_Separability",
        "Avg_Processing_Time_ms",
    ]

    overall_df = (
        per_fruit_df.groupby(
            "Method"
        )[metric_columns]
        .mean()
        .reset_index()
    )

    overall_df.insert(
        1,
        "Number_of_Features",
        3,
    )

    overall_df.insert(
        2,
        "Fruit_Count",
        per_fruit_df[
            "Fruit"
        ].nunique(),
    )

    overall_df = (
        overall_df.sort_values(
            by=[
                "F1_Score",
                "Accuracy",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return overall_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "MEMBER 2 - COLOUR ANALYSIS"
    )
    print(
        "STAGE 1: PER-FRUIT "
        "RGB vs HSV vs Lab"
    )
    print("=" * 75)

    records = get_dataset_records()

    if not records:

        raise RuntimeError(
            "No valid image-mask pairs found."
        )

    sample_df = create_per_fruit_sample(
        records
    )

    # Save exact sample used
    sample_save = sample_df.copy()

    sample_save[
        "image_path"
    ] = sample_save[
        "image_path"
    ].astype(str)

    sample_save[
        "mask_path"
    ] = sample_save[
        "mask_path"
    ].astype(str)

    sample_save.to_csv(
        SAMPLE_FILE,
        index=False,
    )

    print(
        f"\nSaved sample:\n"
        f"{SAMPLE_FILE}"
    )

    fruits = sorted(
        sample_df[
            "fruit"
        ].unique()
    )

    print(
        f"\nFruits to evaluate: "
        f"{len(fruits)}"
    )

    print(
        ", ".join(fruits)
    )

    all_results = []

    feature_frames = {
        "RGB": [],
        "HSV": [],
        "Lab": [],
    }

    # ========================================================
    # EACH FRUIT
    # ========================================================

    for fruit in fruits:

        print("\n")
        print("=" * 75)

        print(
            f"TESTING FRUIT: "
            f"{fruit}"
        )

        print("=" * 75)

        fruit_df = (
            sample_df[
                sample_df["fruit"]
                == fruit
            ]
            .copy()
            .reset_index(drop=True)
        )

        print("\nCategories:")

        for category, count in (
            fruit_df[
                "category"
            ]
            .value_counts()
            .sort_index()
            .items()
        ):

            print(
                f"  {category}: "
                f"{count}"
            )

        fruit_results = []

        for colour_space in [
            "RGB",
            "HSV",
            "Lab",
        ]:

            print(
                f"\nTesting "
                f"{colour_space}..."
            )

            (
                features,
                labels,
                metadata,
                average_time_ms,
            ) = extract_colour_features(
                fruit_df,
                colour_space,
            )

            feature_frames[
                colour_space
            ].append(
                create_feature_dataframe(
                    features,
                    metadata,
                    colour_space,
                )
            )

            result = evaluate_features(
                features,
                labels,
                colour_space,
                average_time_ms,
            )

            result["Fruit"] = fruit

            result[
                "Number_of_Classes"
            ] = len(
                np.unique(labels)
            )

            result[
                "Number_of_Images"
            ] = len(labels)

            fruit_results.append(
                result
            )

            all_results.append(
                result
            )

        display_fruit_result(
            fruit,
            fruit_results,
        )

    # ========================================================
    # SAVE BASIC FEATURE FILES
    # ========================================================

    print("\n")
    print("=" * 75)
    print(
        "SAVING BASIC FEATURE FILES"
    )
    print("=" * 75)

    for colour_space in [
        "RGB",
        "HSV",
        "Lab",
    ]:

        combined_df = pd.concat(
            feature_frames[
                colour_space
            ],
            ignore_index=True,
        )

        output_file = (
            OUTPUT_DIR
            / (
                f"{colour_space.lower()}"
                "_basic_features.csv"
            )
        )

        combined_df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"Saved: "
            f"{output_file.name}"
        )

    # ========================================================
    # PER-FRUIT RESULTS
    # ========================================================

    per_fruit_df = pd.DataFrame(
        all_results
    )

    per_fruit_df = per_fruit_df[
        [
            "Fruit",
            "Method",
            "Number_of_Classes",
            "Number_of_Images",
            "Number_of_Features",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "Fisher_Separability",
            "Avg_Processing_Time_ms",
        ]
    ]

    per_fruit_df.to_csv(
        PER_FRUIT_RESULT_FILE,
        index=False,
    )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    overall_df = (
        calculate_overall_results(
            per_fruit_df
        )
    )

    overall_df.to_csv(
        OVERALL_RESULT_FILE,
        index=False,
    )

    display_df = overall_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
    ]:

        display_df[column] *= 100

    print("\n")
    print("=" * 100)

    print(
        "OVERALL COLOUR SPACE COMPARISON"
    )

    print(
        "Average performance "
        "across all fruits"
    )

    print("=" * 100)

    print(
        display_df.to_string(
            index=False
        )
    )

    best = overall_df.iloc[0]

    print("\n")
    print("=" * 75)

    print(
        f"BEST OVERALL COLOUR SPACE: "
        f"{best['Method']}"
    )

    print(
        f"Average Accuracy: "
        f"{best['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Average Macro F1: "
        f"{best['F1_Score'] * 100:.2f}%"
    )

    print(
        f"Average Fisher Separability: "
        f"{best['Fisher_Separability']:.4f}"
    )

    print("=" * 75)

    print(
        "\nStage 1 completed."
    )


if __name__ == "__main__":
    main()