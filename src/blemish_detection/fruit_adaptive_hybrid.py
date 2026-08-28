"""
Fruit-Adaptive Hybrid (FAH) Segmentation Evaluation
====================================================

Purpose:
    Evaluate whether fruit-specific method selection improves blemish
    segmentation compared with using one global segmentation method.

Important:
    This script DOES NOT select the best method using the test image itself.

    For every held-out image:
        1. Remove that image.
        2. Use the other annotated images of the SAME fruit.
        3. Find the segmentation method with the highest mean IoU.
        4. Use that selected method's result for the held-out image.
        5. Repeat for all 27 images.

This provides a leave-one-out fruit-adaptive evaluation.

Input:
    results/blemish_detection/fabr_enhancement/
    fabr_detailed_results.csv

Output:
    results/blemish_detection/fruit_adaptive_hybrid/
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
    / "fabr_enhancement"
    / "fabr_detailed_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "blemish_detection"
    / "fruit_adaptive_hybrid"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

METRICS = [
    "IoU",
    "Dice",
    "Precision",
    "Recall",
]

BASELINE_METHOD = "Otsu + Morphology 7x7"

ORIGINAL_OTSU = "Original Otsu"

FABR_METHODS = [
    "FABR_A",
    "FABR_B",
    "FABR_C",
    "FABR_D",
    "FABR_E",
]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df):
    """
    Check that all columns required by the FAH experiment exist.
    """

    required_columns = {
        "fruit",
        "category",
        "image",
        "method",
        "IoU",
        "Dice",
        "Precision",
        "Recall",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing)
            )
            + "\n\nAvailable columns:\n"
            + str(df.columns.tolist())
        )

    print("Dataset validation: PASSED")


# ============================================================
# DATASET INFORMATION
# ============================================================

def print_dataset_information(df):

    print()
    print("=" * 72)
    print("DATASET INFORMATION")
    print("=" * 72)

    images = (
        df[
            [
                "fruit",
                "category",
                "image",
            ]
        ]
        .drop_duplicates()
    )

    print(
        f"Total result rows   : {len(df)}"
    )

    print(
        f"Unique GT images    : {len(images)}"
    )

    print(
        f"Fruit types         : {df['fruit'].nunique()}"
    )

    print(
        f"Methods             : {df['method'].nunique()}"
    )

    print("\nMethods found:")

    for method in sorted(
        df["method"].unique()
    ):
        print(
            f"  - {method}"
        )

    print("\nImages per fruit:")

    fruit_counts = (
        images
        .groupby("fruit")
        .size()
        .sort_index()
    )

    for fruit, count in fruit_counts.items():

        print(
            f"  {fruit:<10}: {count}"
        )


# ============================================================
# METHOD SUMMARY
# ============================================================

def calculate_global_method_summary(df):
    """
    Calculate average performance of every global method.
    """

    summary = (
        df
        .groupby("method")[METRICS]
        .mean()
        .reset_index()
    )

    summary["Overall_Score"] = (
        summary[METRICS]
        .mean(axis=1)
    )

    summary = summary.sort_values(
        "Overall_Score",
        ascending=False,
    )

    return summary


# ============================================================
# FRUIT-SPECIFIC METHOD SUMMARY
# ============================================================

def calculate_fruit_method_summary(df):
    """
    Show how each method performs for each fruit.

    This is descriptive only.

    IMPORTANT:
    These values are NOT directly used to evaluate the same images.
    The actual FAH evaluation uses leave-one-out selection below.
    """

    summary = (
        df
        .groupby(
            [
                "fruit",
                "method",
            ]
        )[METRICS]
        .mean()
        .reset_index()
    )

    summary["Overall_Score"] = (
        summary[METRICS]
        .mean(axis=1)
    )

    return summary


# ============================================================
# LEAVE-ONE-OUT METHOD SELECTION
# ============================================================

def select_method_leave_one_out(
    df,
    fruit,
    held_out_image,
):
    """
    Select the best method for one held-out image.

    Selection uses:
        - same fruit
        - all OTHER images
        - highest mean IoU

    The held-out image is NEVER used to select its own method.
    """

    training_data = df[
        (df["fruit"] == fruit)
        & (df["image"] != held_out_image)
    ].copy()

    if training_data.empty:
        raise ValueError(
            f"No training samples available for {fruit} "
            f"after holding out {held_out_image}."
        )

    method_performance = (
        training_data
        .groupby("method")[METRICS]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------------
    # Primary criterion:
    # highest Mean IoU
    #
    # Tie-breaking:
    # 1. Dice
    # 2. Precision
    # 3. Recall
    # --------------------------------------------------------

    method_performance = (
        method_performance
        .sort_values(
            by=[
                "IoU",
                "Dice",
                "Precision",
                "Recall",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    best = method_performance.iloc[0]

    return (
        best["method"],
        float(best["IoU"]),
        float(best["Dice"]),
        float(best["Precision"]),
        float(best["Recall"]),
    )


# ============================================================
# RUN FRUIT-ADAPTIVE HYBRID
# ============================================================

def run_fah(df):
    """
    Perform leave-one-out Fruit-Adaptive Hybrid evaluation.
    """

    images = (
        df[
            [
                "fruit",
                "category",
                "image",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "fruit",
                "category",
                "image",
            ]
        )
        .reset_index(drop=True)
    )

    results = []

    print()
    print("=" * 72)
    print("LEAVE-ONE-OUT FRUIT-ADAPTIVE METHOD SELECTION")
    print("=" * 72)

    print(
        f"Images to evaluate: {len(images)}"
    )

    print()

    for index, image_row in images.iterrows():

        fruit = image_row["fruit"]
        category = image_row["category"]
        image_name = image_row["image"]

        (
            selected_method,
            selection_iou,
            selection_dice,
            selection_precision,
            selection_recall,
        ) = select_method_leave_one_out(
            df,
            fruit,
            image_name,
        )

        # ----------------------------------------------------
        # Retrieve held-out performance for selected method
        # ----------------------------------------------------

        held_out = df[
            (df["fruit"] == fruit)
            & (df["image"] == image_name)
            & (df["method"] == selected_method)
        ]

        if held_out.empty:

            print(
                f"[WARNING] No result for "
                f"{image_name} using {selected_method}"
            )

            continue

        held_out = held_out.iloc[0]

        results.append(
            {
                "fruit": fruit,
                "category": category,
                "image": image_name,

                "selected_method": selected_method,

                # Performance used during selection
                # DOES NOT include current image.
                "selection_mean_IoU": selection_iou,
                "selection_mean_Dice": selection_dice,
                "selection_mean_Precision":
                    selection_precision,
                "selection_mean_Recall":
                    selection_recall,

                # Actual held-out performance
                "IoU": float(
                    held_out["IoU"]
                ),
                "Dice": float(
                    held_out["Dice"]
                ),
                "Precision": float(
                    held_out["Precision"]
                ),
                "Recall": float(
                    held_out["Recall"]
                ),
            }
        )

        print(
            f"[{index + 1:02d}/{len(images)}] "
            f"{fruit:<8} | "
            f"{category:<10} | "
            f"{selected_method:<24} | "
            f"IoU={held_out['IoU']:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# FAH SUMMARY
# ============================================================

def calculate_fah_summary(fah_df):

    if fah_df.empty:
        raise ValueError(
            "FAH produced no results."
        )

    means = (
        fah_df[METRICS]
        .mean()
    )

    summary = {
        "method":
            "Fruit-Adaptive Hybrid (LOO)",

        "IoU":
            float(means["IoU"]),

        "Dice":
            float(means["Dice"]),

        "Precision":
            float(means["Precision"]),

        "Recall":
            float(means["Recall"]),
    }

    summary["Overall_Score"] = np.mean(
        [
            summary["IoU"],
            summary["Dice"],
            summary["Precision"],
            summary["Recall"],
        ]
    )

    return summary


# ============================================================
# METHOD SELECTION FREQUENCY
# ============================================================

def calculate_selection_frequency(fah_df):

    frequency = (
        fah_df
        .groupby(
            [
                "fruit",
                "selected_method",
            ]
        )
        .size()
        .reset_index(
            name="selection_count"
        )
    )

    return frequency


# ============================================================
# COMPARE FAH AGAINST GLOBAL METHODS
# ============================================================

def build_final_comparison(
    global_summary,
    fah_summary,
):

    fah_row = pd.DataFrame(
        [fah_summary]
    )

    comparison = pd.concat(
        [
            global_summary,
            fah_row,
        ],
        ignore_index=True,
    )

    comparison = comparison[
        [
            "method",
            "IoU",
            "Dice",
            "Precision",
            "Recall",
            "Overall_Score",
        ]
    ]

    comparison = comparison.sort_values(
        "Overall_Score",
        ascending=False,
    ).reset_index(drop=True)

    return comparison


# ============================================================
# BASELINE IMPROVEMENT
# ============================================================

def calculate_improvement(
    comparison,
    fah_summary,
):

    baseline = comparison[
        comparison["method"]
        == BASELINE_METHOD
    ]

    if baseline.empty:

        print()
        print(
            "WARNING: Baseline method not found:"
        )

        print(
            f"  {BASELINE_METHOD}"
        )

        return None

    baseline = baseline.iloc[0]

    improvement = {}

    for metric in (
        METRICS
        + ["Overall_Score"]
    ):

        improvement[metric] = (
            fah_summary[metric]
            - float(
                baseline[metric]
            )
        )

    return improvement


# ============================================================
# PER-FRUIT FAH PERFORMANCE
# ============================================================

def calculate_per_fruit_fah(fah_df):

    summary = (
        fah_df
        .groupby("fruit")[METRICS]
        .mean()
        .reset_index()
    )

    summary["Overall_Score"] = (
        summary[METRICS]
        .mean(axis=1)
    )

    return summary


# ============================================================
# PRINT GLOBAL METHODS
# ============================================================

def print_global_summary(summary):

    print()
    print("=" * 72)
    print("GLOBAL METHOD PERFORMANCE")
    print("=" * 72)

    for _, row in summary.iterrows():

        print()
        print(row["method"])

        print(
            f"  Mean IoU       : "
            f"{row['IoU']:.4f}"
        )

        print(
            f"  Mean Dice      : "
            f"{row['Dice']:.4f}"
        )

        print(
            f"  Mean Precision : "
            f"{row['Precision']:.4f}"
        )

        print(
            f"  Mean Recall    : "
            f"{row['Recall']:.4f}"
        )

        print(
            f"  Overall Score  : "
            f"{row['Overall_Score']:.4f}"
        )


# ============================================================
# PRINT FAH SUMMARY
# ============================================================

def print_fah_summary(summary):

    print()
    print("=" * 72)
    print("FRUIT-ADAPTIVE HYBRID PERFORMANCE")
    print("=" * 72)

    print()
    print(
        f"Mean IoU       : "
        f"{summary['IoU']:.4f}"
    )

    print(
        f"Mean Dice      : "
        f"{summary['Dice']:.4f}"
    )

    print(
        f"Mean Precision : "
        f"{summary['Precision']:.4f}"
    )

    print(
        f"Mean Recall    : "
        f"{summary['Recall']:.4f}"
    )

    print(
        f"Overall Score  : "
        f"{summary['Overall_Score']:.4f}"
    )


# ============================================================
# PRINT METHOD SELECTIONS
# ============================================================

def print_selection_frequency(frequency):

    print()
    print("=" * 72)
    print("FRUIT-SPECIFIC METHOD SELECTION FREQUENCY")
    print("=" * 72)

    current_fruit = None

    for _, row in frequency.iterrows():

        fruit = row["fruit"]

        if fruit != current_fruit:

            print()
            print(f"{fruit}")

            current_fruit = fruit

        print(
            f"  {row['selected_method']:<25}"
            f"{int(row['selection_count'])} time(s)"
        )


# ============================================================
# PRINT PER-FRUIT PERFORMANCE
# ============================================================

def print_per_fruit_summary(summary):

    print()
    print("=" * 72)
    print("FAH PERFORMANCE BY FRUIT")
    print("=" * 72)

    for _, row in summary.iterrows():

        print()
        print(row["fruit"])

        print(
            f"  IoU       : "
            f"{row['IoU']:.4f}"
        )

        print(
            f"  Dice      : "
            f"{row['Dice']:.4f}"
        )

        print(
            f"  Precision : "
            f"{row['Precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{row['Recall']:.4f}"
        )

        print(
            f"  Score     : "
            f"{row['Overall_Score']:.4f}"
        )


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

def print_final_comparison(
    comparison,
    improvement,
):

    print()
    print("=" * 72)
    print("FINAL METHOD COMPARISON")
    print("=" * 72)

    print()

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    if improvement is not None:

        print()
        print("=" * 72)
        print(
            "FAH CHANGE VS OTSU + MORPHOLOGY 7x7"
        )
        print("=" * 72)

        print()

        for metric in (
            METRICS
            + ["Overall_Score"]
        ):

            value = improvement[metric]

            sign = "+" if value >= 0 else ""

            print(
                f"{metric:<15}: "
                f"{sign}{value:.4f}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    global_summary,
    fruit_method_summary,
    fah_df,
    fah_summary,
    selection_frequency,
    per_fruit_fah,
    comparison,
):

    global_summary.to_csv(
        OUTPUT_DIR
        / "global_method_summary.csv",
        index=False,
    )

    fruit_method_summary.to_csv(
        OUTPUT_DIR
        / "fruit_method_summary.csv",
        index=False,
    )

    fah_df.to_csv(
        OUTPUT_DIR
        / "fah_leave_one_out_results.csv",
        index=False,
    )

    pd.DataFrame(
        [fah_summary]
    ).to_csv(
        OUTPUT_DIR
        / "fah_summary.csv",
        index=False,
    )

    selection_frequency.to_csv(
        OUTPUT_DIR
        / "method_selection_frequency.csv",
        index=False,
    )

    per_fruit_fah.to_csv(
        OUTPUT_DIR
        / "fah_per_fruit_summary.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "final_method_comparison.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "MEMBER 4 - FRUIT-ADAPTIVE HYBRID SEGMENTATION"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            "\nFABR detailed results were not found:\n"
            f"{INPUT_CSV}\n\n"
            "Run fabr_enhancement.py first."
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    validate_dataframe(
        df
    )

    print_dataset_information(
        df
    )

    # --------------------------------------------------------
    # Global performance
    # --------------------------------------------------------

    global_summary = (
        calculate_global_method_summary(
            df
        )
    )

    print_global_summary(
        global_summary
    )

    # --------------------------------------------------------
    # Descriptive fruit-method performance
    # --------------------------------------------------------

    fruit_method_summary = (
        calculate_fruit_method_summary(
            df
        )
    )

    # --------------------------------------------------------
    # Leave-one-out FAH
    # --------------------------------------------------------

    fah_df = run_fah(
        df
    )

    # --------------------------------------------------------
    # FAH summary
    # --------------------------------------------------------

    fah_summary = (
        calculate_fah_summary(
            fah_df
        )
    )

    print_fah_summary(
        fah_summary
    )

    # --------------------------------------------------------
    # Selection frequency
    # --------------------------------------------------------

    selection_frequency = (
        calculate_selection_frequency(
            fah_df
        )
    )

    print_selection_frequency(
        selection_frequency
    )

    # --------------------------------------------------------
    # Per-fruit FAH
    # --------------------------------------------------------

    per_fruit_fah = (
        calculate_per_fruit_fah(
            fah_df
        )
    )

    print_per_fruit_summary(
        per_fruit_fah
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    comparison = (
        build_final_comparison(
            global_summary,
            fah_summary,
        )
    )

    improvement = (
        calculate_improvement(
            comparison,
            fah_summary,
        )
    )

    print_final_comparison(
        comparison,
        improvement,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        global_summary,
        fruit_method_summary,
        fah_df,
        fah_summary,
        selection_frequency,
        per_fruit_fah,
        comparison,
    )

    print()
    print("=" * 72)
    print("EXPERIMENT COMPLETED")
    print("=" * 72)

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Main comparison:"
    )

    print(
        OUTPUT_DIR
        / "final_method_comparison.csv"
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()