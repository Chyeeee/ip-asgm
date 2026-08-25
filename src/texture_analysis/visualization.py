from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASS_NAMES
from .evaluation import EvaluationResult


def save_confusion_matrix(
    result: EvaluationResult,
    output_dir: Path,
) -> None:
    matrix = result.confusion_matrix

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix
    )

    ax.set_title(
        f"{result.technique} Confusion Matrix"
    )
    ax.set_xlabel(
        "Predicted Class"
    )
    ax.set_ylabel(
        "Actual Class"
    )

    ticks = np.arange(
        len(CLASS_NAMES)
    )

    ax.set_xticks(
        ticks
    )
    ax.set_yticks(
        ticks
    )
    ax.set_xticklabels(
        CLASS_NAMES
    )
    ax.set_yticklabels(
        CLASS_NAMES
    )

    for row in range(
        len(CLASS_NAMES)
    ):
        for column in range(
            len(CLASS_NAMES)
        ):
            ax.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{result.technique.lower()}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_performance_chart(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-score",
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    glcm_scores = (
        comparison.iloc[0][metrics]
        .astype(float)
        .to_numpy()
    )

    lbp_scores = (
        comparison.iloc[1][metrics]
        .astype(float)
        .to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars_glcm = ax.bar(
        x - width / 2,
        glcm_scores,
        width,
        label="GLCM",
    )

    bars_lbp = ax.bar(
        x + width / 2,
        lbp_scores,
        width,
        label="LBP",
    )

    ax.set_title(
        "GLCM vs LBP Classification Performance"
    )
    ax.set_ylabel(
        "Score"
    )
    ax.set_ylim(
        0,
        1.05,
    )
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        metrics
    )
    ax.legend()

    for bars in (
        bars_glcm,
        bars_lbp,
    ):
        for bar in bars:
            score = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                score + 0.015,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "glcm_vs_lbp_performance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )
