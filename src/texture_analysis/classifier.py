from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    SVM_C,
    SVM_KERNEL,
)


def create_classifier() -> Pipeline:
    """
    The exact same SVM pipeline is used for GLCM and LBP.
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel=SVM_KERNEL,
                    C=SVM_C,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]
    )
