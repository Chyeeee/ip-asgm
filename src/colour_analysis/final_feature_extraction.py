import pandas as pd

from config import (
    OUTPUT_DIR,
    PROCESSED_DIR,
)

from data_loader import (
    get_dataset_records,
    load_image_mask,
)

from colour_features import (
    extract_basic_features,
    extract_enhanced_features,
    get_basic_feature_names,
    get_enhanced_feature_names,
)


# ============================================================
# INPUT / OUTPUT FILES
# ============================================================

FINAL_SELECTION_FILE = (
    OUTPUT_DIR
    / "final_method_selection.csv"
)

FINAL_FEATURE_FILE = (
    OUTPUT_DIR
    / "colour_features.csv"
)

SKIPPED_FILE = (
    OUTPUT_DIR
    / "skipped_images.csv"
)


# ============================================================
# LOAD FINAL SELECTED METHOD
# ============================================================

def load_final_selection():

    if not FINAL_SELECTION_FILE.exists():

        raise FileNotFoundError(
            "\nfinal_method_selection.csv "
            "not found.\n"
            "Run enhancement_comparison.py "
            "first."
        )

    selection_df = pd.read_csv(
        FINAL_SELECTION_FILE
    )

    if selection_df.empty:

        raise RuntimeError(
            "final_method_selection.csv "
            "is empty."
        )

    selection = (
        selection_df.iloc[0]
    )

    colour_space = (
        selection[
            "Colour_Space"
        ]
    )

    feature_version = (
        selection[
            "Feature_Version"
        ]
    )

    use_enhanced = (
        feature_version
        == "Enhanced"
    )

    return (
        colour_space,
        feature_version,
        use_enhanced,
    )


# ============================================================
# GET FINAL FEATURE NAMES
# ============================================================

def get_selected_feature_names(
    colour_space,
    use_enhanced,
):

    if use_enhanced:

        return (
            get_enhanced_feature_names(
                colour_space
            )
        )

    return get_basic_feature_names(
        colour_space
    )


# ============================================================
# EXTRACT SELECTED FEATURES
# ============================================================

def extract_selected_features(
    image,
    mask,
    colour_space,
    use_enhanced,
):

    if use_enhanced:

        return (
            extract_enhanced_features(
                image,
                mask,
                colour_space,
            )
        )

    return extract_basic_features(
        image,
        mask,
        colour_space,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "MEMBER 2 - COLOUR ANALYSIS"
    )

    print(
        "STAGE 3: FINAL FEATURE EXTRACTION"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Read Stage 2 final selection
    # --------------------------------------------------------

    (
        colour_space,
        feature_version,
        use_enhanced,
    ) = load_final_selection()

    final_method = (
        f"{feature_version} "
        f"{colour_space}"
    )

    print(
        f"\nSelected colour space: "
        f"{colour_space}"
    )

    print(
        f"Selected feature version: "
        f"{feature_version}"
    )

    print(
        f"Final selected method: "
        f"{final_method}"
    )

    # --------------------------------------------------------
    # Load COMPLETE dataset
    # --------------------------------------------------------

    records = get_dataset_records()

    if not records:

        raise RuntimeError(
            "No valid image-mask "
            "pairs found."
        )

    feature_names = (
        get_selected_feature_names(
            colour_space,
            use_enhanced,
        )
    )

    print(
        f"\nNumber of final "
        f"colour features: "
        f"{len(feature_names)}"
    )

    print(
        f"Total images to process: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    final_records = []

    skipped_records = []

    total = len(records)

    for index, record in enumerate(
        records,
        start=1,
    ):

        image_path = (
            record[
                "image_path"
            ]
        )

        mask_path = (
            record[
                "mask_path"
            ]
        )

        try:

            image, mask = load_image_mask(
                image_path,
                mask_path,
            )

            features = (
                extract_selected_features(
                    image,
                    mask,
                    colour_space,
                    use_enhanced,
                )
            )

            # Empty ROI
            if features is None:

                skipped_records.append({
                    "fruit":
                        record[
                            "fruit"
                        ],

                    "category":
                        record[
                            "category"
                        ],

                    "image":
                        image_path.name,

                    "relative_path":
                        image_path
                        .relative_to(
                            PROCESSED_DIR
                        )
                        .as_posix(),

                    "reason":
                        "Empty ROI mask",
                })

                continue

            row = {
                "fruit":
                    record[
                        "fruit"
                    ],

                "category":
                    record[
                        "category"
                    ],

                "image":
                    image_path.name,

                "relative_path":
                    image_path
                    .relative_to(
                        PROCESSED_DIR
                    )
                    .as_posix(),

                "colour_space":
                    colour_space,

                "feature_version":
                    feature_version,
            }

            for (
                feature_name,
                feature_value,
            ) in zip(
                feature_names,
                features,
            ):

                row[
                    feature_name
                ] = float(
                    feature_value
                )

            final_records.append(
                row
            )

        except Exception as error:

            skipped_records.append({
                "fruit":
                    record[
                        "fruit"
                    ],

                "category":
                    record[
                        "category"
                    ],

                "image":
                    image_path.name,

                "relative_path":
                    image_path
                    .relative_to(
                        PROCESSED_DIR
                    )
                    .as_posix(),

                "reason":
                    str(error),
            })

        if (
            index % 100 == 0
            or
            index == total
        ):

            print(
                f"Processed: "
                f"{index}/{total}"
            )

    # ========================================================
    # SAVE FINAL FEATURE CSV
    # ========================================================

    if not final_records:

        raise RuntimeError(
            "No final colour features "
            "were extracted."
        )

    final_df = pd.DataFrame(
        final_records
    )

    final_df.to_csv(
        FINAL_FEATURE_FILE,
        index=False,
    )

    # ========================================================
    # SAVE SKIPPED IMAGE REPORT
    # ========================================================

    skipped_df = pd.DataFrame(
        skipped_records,
        columns=[
            "fruit",
            "category",
            "image",
            "relative_path",
            "reason",
        ],
    )

    skipped_df.to_csv(
        SKIPPED_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 75)

    print(
        "FINAL COLOUR FEATURE "
        "EXTRACTION COMPLETED"
    )

    print("=" * 75)

    print(
        f"Final method: "
        f"{final_method}"
    )

    print(
        f"Features per image: "
        f"{len(feature_names)}"
    )

    print(
        f"Successfully processed: "
        f"{len(final_df)}"
    )

    print(
        f"Skipped images: "
        f"{len(skipped_df)}"
    )

    print(
        "\nImages per fruit:"
    )

    print(
        final_df[
            "fruit"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nImages per "
        "fruit/category:"
    )

    print(
        final_df.groupby(
            [
                "fruit",
                "category",
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nFinal feature file:"
    )

    print(
        FINAL_FEATURE_FILE
    )

    print(
        "\nSkipped image report:"
    )

    print(
        SKIPPED_FILE
    )

    print("=" * 75)


if __name__ == "__main__":
    main()