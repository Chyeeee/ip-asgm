import pandas as pd

from .evaluation import EvaluationResult


def create_comparison_table(
    glcm_result: EvaluationResult,
    lbp_result: EvaluationResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Technique": "GLCM",
                "Accuracy": glcm_result.accuracy,
                "Macro Precision": glcm_result.precision_macro,
                "Macro Recall": glcm_result.recall_macro,
                "Macro F1-score": glcm_result.f1_macro,
            },
            {
                "Technique": "LBP",
                "Accuracy": lbp_result.accuracy,
                "Macro Precision": lbp_result.precision_macro,
                "Macro Recall": lbp_result.recall_macro,
                "Macro F1-score": lbp_result.f1_macro,
            },
        ]
    )


def determine_best_technique(
    glcm_result: EvaluationResult,
    lbp_result: EvaluationResult,
) -> tuple[str, str]:
    """
    Accuracy is the primary comparison measure.
    Macro F1-score is used as the tie-breaker.
    """
    if glcm_result.accuracy > lbp_result.accuracy:
        return (
            "GLCM",
            "GLCM achieved the higher classification accuracy.",
        )

    if lbp_result.accuracy > glcm_result.accuracy:
        return (
            "LBP",
            "LBP achieved the higher classification accuracy.",
        )

    if glcm_result.f1_macro > lbp_result.f1_macro:
        return (
            "GLCM",
            "Accuracy was equal, but GLCM achieved the higher "
            "Macro F1-score.",
        )

    if lbp_result.f1_macro > glcm_result.f1_macro:
        return (
            "LBP",
            "Accuracy was equal, but LBP achieved the higher "
            "Macro F1-score.",
        )

    return (
        "Tie",
        "Both techniques achieved equal accuracy and Macro F1-score.",
    )
