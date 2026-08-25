from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .classifier import create_classifier
from .config import CLASS_NAMES


@dataclass
class EvaluationResult:
    technique: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray
    report: str
    y_true: np.ndarray
    y_pred: np.ndarray
    model: object


def evaluate_features(
    technique: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> EvaluationResult:
    model = create_classifier()

    model.fit(
        x_train,
        y_train,
    )

    predictions = model.predict(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_NAMES,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    return EvaluationResult(
        technique=technique,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        confusion_matrix=matrix,
        report=report,
        y_true=y_test,
        y_pred=predictions,
        model=model,
    )


def print_evaluation(
    result: EvaluationResult,
) -> None:
    print("\n" + "=" * 60)
    print(
        f"{result.technique} CLASSIFICATION PERFORMANCE"
    )
    print("=" * 60)

    print(
        f"Accuracy        : "
        f"{result.accuracy:.4f} "
        f"({result.accuracy * 100:.2f}%)"
    )
    print(
        f"Macro Precision : "
        f"{result.precision_macro:.4f}"
    )
    print(
        f"Macro Recall    : "
        f"{result.recall_macro:.4f}"
    )
    print(
        f"Macro F1-score  : "
        f"{result.f1_macro:.4f}"
    )

    print(
        "\nClassification Report:"
    )
    print(
        result.report
    )

    print(
        "Confusion Matrix:"
    )
    print(
        result.confusion_matrix
    )
