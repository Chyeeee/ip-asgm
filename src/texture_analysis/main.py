import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .comparison import (
    create_comparison_table,
    determine_best_technique,
)
from .config import (
    BANANA_DATASET_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from .data_loader import (
    load_dataset,
    print_dataset_summary,
)
from .evaluation import (
    evaluate_features,
    print_evaluation,
)
from .glcm_features import (
    extract_glcm_dataset,
)
from .lbp_features import (
    extract_lbp_dataset,
)
from .visualization import (
    save_confusion_matrix,
    save_performance_chart,
)


def main():
    print("=" * 60)
    print("TEXTURE-BASED SURFACE QUALITY ANALYSIS")
    print("Banana: Class_A vs Class_B vs Defect")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # STEP 1 & 2:
    # Dataset 2, banana Class_A / Class_B / Defect
    records = load_dataset(
        BANANA_DATASET_DIR
    )

    print_dataset_summary(
        records
    )

    labels = np.asarray(
        [
            record.label
            for record in records
        ]
    )

    # STEP 3:
    # Extract ROI-based GLCM and LBP features
    glcm_features, glcm_table = (
        extract_glcm_dataset(
            records
        )
    )

    lbp_features, lbp_table = (
        extract_lbp_dataset(
            records
        )
    )

    glcm_table.to_csv(
        OUTPUT_DIR
        / "glcm_features.csv",
        index=False,
    )

    lbp_table.to_csv(
        OUTPUT_DIR
        / "lbp_features.csv",
        index=False,
    )

    # Same train/test image indices for both feature techniques
    indices = np.arange(
        len(records)
    )

    train_indices, test_indices = (
        train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
    )

    y_train = labels[
        train_indices
    ]
    y_test = labels[
        test_indices
    ]

    # STEP 4:
    # Compare classification performance using same SVM
    glcm_result = evaluate_features(
        technique="GLCM",
        x_train=glcm_features[
            train_indices
        ],
        x_test=glcm_features[
            test_indices
        ],
        y_train=y_train,
        y_test=y_test,
    )

    lbp_result = evaluate_features(
        technique="LBP",
        x_train=lbp_features[
            train_indices
        ],
        x_test=lbp_features[
            test_indices
        ],
        y_train=y_train,
        y_test=y_test,
    )

    print_evaluation(
        glcm_result
    )
    print_evaluation(
        lbp_result
    )

    # STEP 5:
    # Determine best texture technique
    comparison = create_comparison_table(
        glcm_result,
        lbp_result,
    )

    best, reason = determine_best_technique(
        glcm_result,
        lbp_result,
    )

    print("\n" + "=" * 60)
    print("GLCM VS LBP COMPARISON")
    print("=" * 60)
    print(
        comparison.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("BEST TEXTURE TECHNIQUE")
    print("=" * 60)
    print(
        f"Best technique: {best}"
    )
    print(
        f"Reason: {reason}"
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "performance_comparison.csv",
        index=False,
    )

    save_confusion_matrix(
        glcm_result,
        OUTPUT_DIR,
    )

    save_confusion_matrix(
        lbp_result,
        OUTPUT_DIR,
    )

    save_performance_chart(
        comparison,
        OUTPUT_DIR,
    )

    joblib.dump(
        glcm_result.model,
        OUTPUT_DIR
        / "glcm_svm_model.joblib",
    )

    joblib.dump(
        lbp_result.model,
        OUTPUT_DIR
        / "lbp_svm_model.joblib",
    )

    print(
        "\nResults saved in:"
    )
    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
