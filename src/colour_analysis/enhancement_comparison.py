from time import perf_counter

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

from data_loader import (
    load_image_mask,
)

from colour_features import (
    extract_basic_features,
    extract_enhanced_features,
)

from evaluation import (
    evaluate_features,
)


# ============================================================
# INPUT / OUTPUT FILES
# ============================================================

METHOD_COMPARISON_FILE = (
    OUTPUT_DIR
    / "method_comparison.csv"
)

SAMPLE_FILE = (
    OUTPUT_DIR
    / "comparison_sample.csv"
)

PER_FRUIT_OUTPUT_FILE = (
    OUTPUT_DIR
    / "per_fruit_enhancement_comparison.csv"
)

OVERALL_OUTPUT_FILE = (
    OUTPUT_DIR
    / "enhanced_method_comparison.csv"
)

FINAL_SELECTION_FILE = (
    OUTPUT_DIR
    / "final_method_selection.csv"
)


# ============================================================
# GET STAGE 1 WINNER
# ============================================================

def get_best_colour_space():

    if not METHOD_COMPARISON_FILE.exists():

        raise FileNotFoundError(
            "\nmethod_comparison.csv "
            "not found.\n"
            "Run method_comparison.py first."
        )

    results = pd.read_csv(
        METHOD_COMPARISON_FILE
    )

    results = results.sort_values(
        by=[
            "F1_Score",
            "Accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return results.iloc[0][
        "Method"
    ]


# ============================================================
# EXTRACT BASIC / ENHANCED DATASET
# ============================================================

def extract_version_features(
    fruit_df,
    colour_space,
    version,
):

    features = []
    labels = []
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

            if version == "Basic":

                feature = (
                    extract_basic_features(
                        image,
                        mask,
                        colour_space,
                    )
                )

            else:

                feature = (
                    extract_enhanced_features(
                        image,
                        mask,
                        colour_space,
                    )
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
                f"  {version}: "
                f"{position}/{total}"
            )

    if not features:

        raise RuntimeError(
            f"No valid {version} "
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
        average_time_ms,
    )


# ============================================================
# DISPLAY PER-FRUIT RESULT
# ============================================================

def display_fruit_result(
    fruit,
    results,
):

    result_df = pd.DataFrame(
        results
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
    print("-" * 95)

    print(
        f"{fruit.upper()} "
        f"BASIC VS ENHANCED"
    )

    print("-" * 95)

    print(
        display_df[
            [
                "Version",
                "Number_of_Features",
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
        f"{best['Version']} "
        f"(Macro F1 = "
        f"{best['F1_Score'] * 100:.2f}%)"
    )


# ============================================================
# CALCULATE OVERALL AVERAGE
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

    rows = []

    for version in [
        "Basic",
        "Enhanced",
    ]:

        version_df = (
            per_fruit_df[
                per_fruit_df[
                    "Version"
                ]
                == version
            ]
        )

        row = {
            "Version":
                version,

            "Colour_Space":
                version_df.iloc[0][
                    "Colour_Space"
                ],

            "Number_of_Features":
                int(
                    version_df[
                        "Number_of_Features"
                    ].iloc[0]
                ),

            "Fruit_Count":
                version_df[
                    "Fruit"
                ].nunique(),
        }

        for metric in metric_columns:

            row[metric] = (
                version_df[
                    metric
                ].mean()
            )

        rows.append(
            row
        )

    overall_df = pd.DataFrame(
        rows
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
        "STAGE 2: BASIC VS "
        "ENHANCED METHOD"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Stage 1 winner
    # --------------------------------------------------------

    colour_space = (
        get_best_colour_space()
    )

    print(
        f"\nStage 1 winner: "
        f"{colour_space}"
    )

    # --------------------------------------------------------
    # Load same sample as Stage 1
    # --------------------------------------------------------

    if not SAMPLE_FILE.exists():

        raise FileNotFoundError(
            "\ncomparison_sample.csv "
            "not found."
        )

    sample_df = pd.read_csv(
        SAMPLE_FILE
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

    # ========================================================
    # PER FRUIT
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
                sample_df[
                    "fruit"
                ]
                == fruit
            ]
            .copy()
            .reset_index(drop=True)
        )

        fruit_results = []

        for version in [
            "Basic",
            "Enhanced",
        ]:

            print(
                f"\nTesting "
                f"{version} "
                f"{colour_space}..."
            )

            (
                features,
                labels,
                average_time_ms,
            ) = extract_version_features(
                fruit_df,
                colour_space,
                version,
            )

            method_name = (
                f"{version} "
                f"{colour_space}"
            )

            result = evaluate_features(
                features,
                labels,
                method_name,
                average_time_ms,
            )

            result["Fruit"] = fruit

            result[
                "Colour_Space"
            ] = colour_space

            result[
                "Version"
            ] = version

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
    # SAVE PER-FRUIT RESULTS
    # ========================================================

    per_fruit_df = pd.DataFrame(
        all_results
    )

    per_fruit_df = per_fruit_df[
        [
            "Fruit",
            "Colour_Space",
            "Version",
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
        PER_FRUIT_OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # OVERALL AVERAGE
    # ========================================================

    overall_df = (
        calculate_overall_results(
            per_fruit_df
        )
    )

    overall_df.to_csv(
        OVERALL_OUTPUT_FILE,
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
    print("=" * 105)

    print(
        f"OVERALL BASIC VS ENHANCED "
        f"{colour_space}"
    )

    print(
        "Average performance "
        "across all fruits"
    )

    print("=" * 105)

    print(
        display_df.to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL SELECTION
    # ========================================================

    best = overall_df.iloc[0]

    final_version = best[
        "Version"
    ]

    final_method = (
        f"{final_version} "
        f"{colour_space}"
    )

    basic_row = (
        overall_df[
            overall_df[
                "Version"
            ]
            == "Basic"
        ]
        .iloc[0]
    )

    enhanced_row = (
        overall_df[
            overall_df[
                "Version"
            ]
            == "Enhanced"
        ]
        .iloc[0]
    )

    f1_change = (
        enhanced_row[
            "F1_Score"
        ]
        -
        basic_row[
            "F1_Score"
        ]
    ) * 100

    accuracy_change = (
        enhanced_row[
            "Accuracy"
        ]
        -
        basic_row[
            "Accuracy"
        ]
    ) * 100

    print("\n")
    print("=" * 75)

    print(
        f"FINAL SELECTED METHOD: "
        f"{final_method}"
    )

    print(
        f"Basic Average F1: "
        f"{basic_row['F1_Score'] * 100:.2f}%"
    )

    print(
        f"Enhanced Average F1: "
        f"{enhanced_row['F1_Score'] * 100:.2f}%"
    )

    print(
        f"F1 Change: "
        f"{f1_change:+.2f} "
        f"percentage points"
    )

    print(
        f"Accuracy Change: "
        f"{accuracy_change:+.2f} "
        f"percentage points"
    )

    print("=" * 75)

    # ========================================================
    # SAVE FINAL SELECTION
    # ========================================================

    selection_df = pd.DataFrame([
        {
            "Colour_Space":
                colour_space,

            "Feature_Version":
                final_version,

            "Final_Method":
                final_method,

            "Basic_Avg_Accuracy":
                basic_row[
                    "Accuracy"
                ],

            "Enhanced_Avg_Accuracy":
                enhanced_row[
                    "Accuracy"
                ],

            "Basic_Avg_F1":
                basic_row[
                    "F1_Score"
                ],

            "Enhanced_Avg_F1":
                enhanced_row[
                    "F1_Score"
                ],

            "F1_Change_Percentage_Points":
                f1_change,

            "Accuracy_Change_Percentage_Points":
                accuracy_change,
        }
    ])

    selection_df.to_csv(
        FINAL_SELECTION_FILE,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        PER_FRUIT_OUTPUT_FILE
    )

    print(
        OVERALL_OUTPUT_FILE
    )

    print(
        FINAL_SELECTION_FILE
    )

    print(
        "\nStage 2 completed."
    )


if __name__ == "__main__":
    main()