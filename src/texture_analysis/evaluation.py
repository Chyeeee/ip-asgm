from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classifier import (
    fit_tuned_mahalanobis_models,
    predict_with_fruit_models,
    stratified_reference_split,
)
from .config import RANDOM_SEED, REFERENCE_RATIO
from .pairwise_texture import fit_pairwise_residual_models, predict_pairwise_residual_models
from .feature_fusion import (
    baseline_feature_sets,
    colour_feature_columns,
    proposed_texture_candidates,
)

PROPOSED_METHOD = "Proposed_PTD_Plus"
BASELINE_FUSION_METHOD = "Colour_Texture_Fusion"


def _classification_metrics(true: list[str], pred: list[str]) -> dict[str, float]:
    labels = sorted(set(true) | set(pred))
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true, pred))
        fp = sum(t != label and p == label for t, p in zip(true, pred))
        fn = sum(t == label and p != label for t, p in zip(true, pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    accuracy = sum(t == p for t, p in zip(true, pred)) / max(1, len(true))
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _composite_labels(df: pd.DataFrame, values: list[str]) -> list[str]:
    # Categories differ for some fruits (e.g. Guava), so evaluate fruit|category.
    return [f"{fruit}|{cat}" for fruit, cat in zip(df["fruit"].astype(str), values)]


def _prediction_frame(
    evaluation_df: pd.DataFrame,
    method: str,
    preds: list[str],
    dists: list[float],
) -> pd.DataFrame:
    frame = evaluation_df[
        ["fruit", "category", "image", "relative_path", "processed_path", "mask_path"]
    ].copy()
    frame["method"] = method
    frame["prediction"] = preds
    frame["distance"] = dists
    frame["correct"] = frame["category"].astype(str) == frame["prediction"].astype(str)
    return frame


def evaluate_methods(
    feature_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate stronger baselines plus proposed PTD+.

    Fairness rules:
      * identical final reference/evaluation split for every method
      * all three baselines use the same classical regularized Mahalanobis
        minimum-distance classifier
      * classifier covariance tuning uses only internal reference validation
      * proposed pairwise residual-texture tuning also uses only reference data
      * final evaluation samples are never used for tuning
      * Member 2 colour descriptors are reused unchanged

    Timing rule:
      * model_setup_time_ms = fitting/tuning/configuration
      * classification_time_ms = inference ONLY on the final evaluation set
    """
    ref_idx, eval_idx = stratified_reference_split(feature_df, REFERENCE_RATIO, RANDOM_SEED)
    reference_df = feature_df.loc[ref_idx].copy()
    evaluation_df = feature_df.loc[eval_idx].copy()

    all_results: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    per_fruit_rows: list[dict] = []
    baseline_config_frames: list[pd.DataFrame] = []
    phase1_models: dict[str, dict] = {}

    true_comp = _composite_labels(
        evaluation_df, evaluation_df["category"].astype(str).tolist()
    )

    # ------------------------------------------------------------------
    # Phase 1: stronger but fair baseline methods.
    # ------------------------------------------------------------------
    methods = baseline_feature_sets(feature_df)
    for method_idx, (method, cols) in enumerate(methods.items()):
        if not cols:
            raise ValueError(f"No features found for method {method}")

        setup_started = time.perf_counter()
        models, classifier_config = fit_tuned_mahalanobis_models(
            reference_df,
            cols,
            method_name=method,
            seed_offset=method_idx * 1000,
        )
        setup_elapsed = time.perf_counter() - setup_started
        baseline_config_frames.append(classifier_config)
        phase1_models[method] = models

        infer_started = time.perf_counter()
        preds, dists = predict_with_fruit_models(models, evaluation_df)
        infer_elapsed = time.perf_counter() - infer_started

        pred_comp = _composite_labels(evaluation_df, preds)
        metrics = _classification_metrics(true_comp, pred_comp)
        metrics.update(
            {
                "method": method,
                "feature_count": len(cols),
                "evaluation_samples": len(evaluation_df),
                "model_setup_time_ms": setup_elapsed * 1000.0,
                "classification_time_ms": infer_elapsed * 1000.0,
                "ms_per_image": infer_elapsed * 1000.0 / max(1, len(evaluation_df)),
            }
        )
        all_results.append(metrics)

        pred_frame = _prediction_frame(evaluation_df, method, preds, dists)
        prediction_frames.append(pred_frame)
        for fruit, g in pred_frame.groupby("fruit"):
            per_fruit_rows.append(
                {
                    "method": method,
                    "fruit": fruit,
                    "accuracy": float(g["correct"].mean()),
                    "samples": len(g),
                }
            )

    # ------------------------------------------------------------------
    # Phase 2: PTD+ pairwise texture disambiguation with optional Laws enrichment.
    # ------------------------------------------------------------------
    colour_cols = colour_feature_columns(feature_df)
    texture_candidates = proposed_texture_candidates(feature_df)
    baseline_sets = baseline_feature_sets(feature_df)
    fusion_cols = baseline_sets.get(BASELINE_FUSION_METHOD, [])
    fusion_models = phase1_models.get(BASELINE_FUSION_METHOD, {})
    if not colour_cols:
        raise ValueError("No Member 2 colour features were found for proposed fusion.")
    if not texture_candidates:
        raise ValueError("No proposed multi-scale texture candidates were found.")
    if not fusion_cols or not fusion_models:
        raise ValueError("Strong Colour+Texture baseline was not fitted; PTD+ cannot preserve it.")

    setup_started = time.perf_counter()
    proposed_models, proposed_config, selection_table = fit_pairwise_residual_models(
        reference_df,
        fusion_models,
        fusion_cols,
        texture_candidates,
    )
    setup_elapsed = time.perf_counter() - setup_started

    infer_started = time.perf_counter()
    proposed_preds, proposed_dists, proposed_meta = predict_pairwise_residual_models(
        proposed_models,
        evaluation_df,
    )
    infer_elapsed = time.perf_counter() - infer_started

    pred_comp = _composite_labels(evaluation_df, proposed_preds)
    metrics = _classification_metrics(true_comp, pred_comp)
    if not proposed_config.empty and "total_effective_feature_count" in proposed_config.columns:
        enabled_cfg = proposed_config[proposed_config.get("enabled", False).astype(bool)] if "enabled" in proposed_config.columns else proposed_config
        mean_count = float(enabled_cfg["total_effective_feature_count"].mean()) if not enabled_cfg.empty else float(len(fusion_cols))
    else:
        mean_count = float(len(fusion_cols))
    metrics.update(
        {
            "method": PROPOSED_METHOD,
            "feature_count": int(round(mean_count)),
            "evaluation_samples": len(evaluation_df),
            "model_setup_time_ms": setup_elapsed * 1000.0,
            "classification_time_ms": infer_elapsed * 1000.0,
            "ms_per_image": infer_elapsed * 1000.0 / max(1, len(evaluation_df)),
        }
    )
    all_results.append(metrics)

    proposed_frame = _prediction_frame(
        evaluation_df, PROPOSED_METHOD, proposed_preds, proposed_dists
    )
    if not proposed_meta.empty:
        for col in [
            "baseline_confidence",
            "texture_confidence",
            "pairwise_rule_available",
            "texture_correction_applied",
        ]:
            if col in proposed_meta.columns:
                proposed_frame[col] = proposed_meta[col].to_numpy()
    prediction_frames.append(proposed_frame)
    for fruit, g in proposed_frame.groupby("fruit"):
        per_fruit_rows.append(
            {
                "method": PROPOSED_METHOD,
                "fruit": fruit,
                "accuracy": float(g["correct"].mean()),
                "samples": len(g),
            }
        )

    results_df = pd.DataFrame(all_results)[
        [
            "method",
            "feature_count",
            "evaluation_samples",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "model_setup_time_ms",
            "classification_time_ms",
            "ms_per_image",
        ]
    ]
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    per_fruit_df = pd.DataFrame(per_fruit_rows)
    baseline_classifier_config = (
        pd.concat(baseline_config_frames, ignore_index=True)
        if baseline_config_frames
        else pd.DataFrame()
    )
    return (
        results_df,
        predictions_df,
        per_fruit_df,
        proposed_config,
        selection_table,
        baseline_classifier_config,
    )


def save_baseline_charts(
    baseline_results: pd.DataFrame,
    baseline_per_fruit: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename, title in [
        ("accuracy", "accuracy_comparison.png", "Optimized Baseline Classification Accuracy"),
        ("macro_f1", "f1_comparison.png", "Optimized Baseline Macro F1 Score"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        vals = baseline_results[metric].to_numpy() * 100.0
        bars = ax.bar(baseline_results["method"], vals)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    pivot = baseline_per_fruit.pivot(index="fruit", columns="method", values="accuracy") * 100.0
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Fruit Optimized Baseline Accuracy")
    ax.set_ylim(0, 100)
    ax.legend(title="Method", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_fruit_accuracy.png", dpi=180)
    plt.close(fig)


def save_enhancement_charts(
    enhancement_results: pd.DataFrame,
    enhancement_per_fruit: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename, title in [
        ("accuracy", "enhancement_accuracy_comparison.png", "Best Baseline vs PTD+: Accuracy"),
        ("macro_f1", "enhancement_f1_comparison.png", "Best Baseline vs PTD+: Macro F1"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = enhancement_results[metric].to_numpy() * 100.0
        bars = ax.bar(enhancement_results["method"], vals)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    pivot = enhancement_per_fruit.pivot(index="fruit", columns="method", values="accuracy") * 100.0
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Fruit Accuracy: Best Baseline vs Proposed PTD+")
    ax.set_ylim(0, 100)
    ax.legend(title="Method", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "enhancement_per_fruit_accuracy.png", dpi=180)
    plt.close(fig)


def save_confusion_matrices(predictions_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, df in predictions_df.groupby("method"):
        labels = sorted(set(df["category"].astype(str)) | set(df["prediction"].astype(str)))
        pos = {label: i for i, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(df["category"].astype(str), df["prediction"].astype(str)):
            cm[pos[t], pos[p]] += 1

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("True category")
        ax.set_title(f"Confusion Matrix - {method}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe = method.lower().replace(" ", "_")
        fig.savefig(out_dir / f"{safe}_confusion_matrix.png", dpi=180)
        plt.close(fig)
