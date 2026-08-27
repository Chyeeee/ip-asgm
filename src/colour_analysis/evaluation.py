import numpy as np

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.svm import (
    SVC,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import (
    TEST_SIZE,
    RANDOM_STATE,
)


# ============================================================
# FISHER CLASS SEPARABILITY
# ============================================================

def calculate_fisher_score(
    features,
    labels,
):
    """
    Calculate Fisher class separability.

    Higher value means stronger class separation.
    """

    features = np.asarray(
        features,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels
    )

    classes = np.unique(
        labels
    )

    feature_scores = []

    for feature_index in range(
        features.shape[1]
    ):

        values = features[
            :,
            feature_index
        ]

        overall_mean = np.mean(
            values
        )

        between_class = 0.0
        within_class = 0.0

        for category in classes:

            class_values = values[
                labels == category
            ]

            class_mean = np.mean(
                class_values
            )

            between_class += (
                len(class_values)
                *
                (
                    class_mean
                    - overall_mean
                ) ** 2
            )

            within_class += np.sum(
                (
                    class_values
                    - class_mean
                ) ** 2
            )

        fisher = (
            between_class
            /
            (
                within_class
                + 1e-10
            )
        )

        feature_scores.append(
            fisher
        )

    return float(
        np.mean(
            feature_scores
        )
    )


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_features(
    features,
    labels,
    method_name,
    processing_time_ms,
):
    """
    All experiments use the exact same:

        StandardScaler
        +
        RBF SVM

    so comparison remains fair.
    """

    features = np.asarray(
        features,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels
    )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    classifier = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
            ),
        ),
    ])

    classifier.fit(
        x_train,
        y_train,
    )

    predictions = classifier.predict(
        x_test
    )

    return {
        "Method":
            method_name,

        "Number_of_Features":
            features.shape[1],

        "Accuracy":
            accuracy_score(
                y_test,
                predictions,
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "F1_Score":
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "Fisher_Separability":
            calculate_fisher_score(
                features,
                labels,
            ),

        "Avg_Processing_Time_ms":
            processing_time_ms,
    }