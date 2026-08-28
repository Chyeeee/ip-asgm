from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    INTERNAL_TRAIN_RATIO,
    MAHALANOBIS_ALPHA_CANDIDATES,
    MAHALANOBIS_RIDGE,
    RANDOM_SEED,
)


@dataclass
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where((np.isfinite(std)) & (std > 1e-12), std, 1.0)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, self.mean_)
        return (x - self.mean_) / self.std_


def stratified_reference_split(
    df: pd.DataFrame,
    reference_ratio: float = 0.80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic split stratified by fruit AND category."""
    rng = np.random.default_rng(seed)
    ref_idx: list[int] = []
    eval_idx: list[int] = []

    for _, group in df.groupby(["fruit", "category"], sort=True):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            ref_idx.extend(idx.tolist())
            continue
        n_ref = int(round(n * reference_ratio))
        n_ref = max(1, min(n - 1, n_ref))
        ref_idx.extend(idx[:n_ref].tolist())
        eval_idx.extend(idx[n_ref:].tolist())

    return np.array(ref_idx, dtype=int), np.array(eval_idx, dtype=int)


def stratified_internal_split(
    df: pd.DataFrame,
    ratio: float = INTERNAL_TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one fruit's reference samples by category for validation-only tuning."""
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for _, group in df.groupby("category", sort=True):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        if len(idx) <= 1:
            train.extend(idx.tolist())
            continue
        n_train = int(round(len(idx) * ratio))
        n_train = max(1, min(len(idx) - 1, n_train))
        train.extend(idx[:n_train].tolist())
        val.extend(idx[n_train:].tolist())
    return np.asarray(train, dtype=int), np.asarray(val, dtype=int)


def macro_f1(true: Iterable[str], pred: Iterable[str]) -> float:
    true = list(map(str, true))
    pred = list(map(str, pred))
    labels = sorted(set(true) | set(pred))
    f1s: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true, pred))
        fp = sum(t != label and p == label for t, p in zip(true, pred))
        fn = sum(t == label and p != label for t, p in zip(true, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _fruit_seed(fruit: str, offset: int = 0) -> int:
    return RANDOM_SEED + offset + sum((i + 1) * ord(ch) for i, ch in enumerate(str(fruit)))


@dataclass
class MahalanobisFruitModel:
    fruit: str
    feature_cols: list[str]
    scaler: Standardizer
    classes: list[str]
    centroids: np.ndarray
    precision: np.ndarray
    alpha: float
    feature_weights: np.ndarray

    def distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distance from each sample to every class centroid."""
        z = self.scaler.transform(x)
        diff = z[:, None, :] - self.centroids[None, :, :]
        # Feature weighting changes only the distance contribution; covariance
        # estimation remains based on the standardized reference distribution.
        diff = diff * self.feature_weights[None, None, :]
        d2 = np.einsum("nki,ij,nkj->nk", diff, self.precision, diff, optimize=True)
        return np.sqrt(np.maximum(d2, 0.0))

    def predict_matrix(self, x: np.ndarray) -> tuple[list[str], list[float]]:
        distances = self.distance_matrix(x)
        best = np.argmin(distances, axis=1)
        preds = [self.classes[int(i)] for i in best]
        dists = distances[np.arange(len(distances)), best].astype(float).tolist()
        return preds, dists


def _fit_mahalanobis_one_fruit(
    df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float,
    feature_weight_map: dict[str, float] | None = None,
) -> MahalanobisFruitModel:
    if df.empty:
        raise ValueError("Cannot fit classifier on an empty dataframe.")

    x = df[feature_cols].to_numpy(dtype=float)
    scaler = Standardizer.fit(x)
    z = scaler.transform(x)
    labels = df["category"].astype(str).to_numpy()
    classes = sorted(np.unique(labels).tolist())

    centroids = np.vstack([z[labels == cls].mean(axis=0) for cls in classes])
    class_pos = {c: i for i, c in enumerate(classes)}
    residuals = np.vstack([
        row - centroids[class_pos[label]] for row, label in zip(z, labels)
    ])

    p = z.shape[1]
    denom = max(1, len(z) - len(classes))
    cov = (residuals.T @ residuals) / denom
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - float(alpha)) * cov + float(alpha) * diag
    shrunk = shrunk + np.eye(p, dtype=float) * MAHALANOBIS_RIDGE
    precision = np.linalg.pinv(shrunk, hermitian=True)

    weights = np.ones(p, dtype=float)
    if feature_weight_map:
        for i, col in enumerate(feature_cols):
            weights[i] = float(feature_weight_map.get(col, 1.0))

    return MahalanobisFruitModel(
        fruit=str(df["fruit"].iloc[0]),
        feature_cols=list(feature_cols),
        scaler=scaler,
        classes=classes,
        centroids=centroids,
        precision=precision,
        alpha=float(alpha),
        feature_weights=weights,
    )


def _tune_alpha_for_fruit(
    reference_fruit: pd.DataFrame,
    feature_cols: list[str],
    feature_weight_map: dict[str, float] | None = None,
    candidate_alphas: tuple[float, ...] = MAHALANOBIS_ALPHA_CANDIDATES,
    seed_offset: int = 0,
) -> tuple[float, float, float]:
    fruit = str(reference_fruit["fruit"].iloc[0])
    train_idx, val_idx = stratified_internal_split(
        reference_fruit,
        INTERNAL_TRAIN_RATIO,
        _fruit_seed(fruit, offset=seed_offset),
    )
    train_df = reference_fruit.loc[train_idx]
    val_df = reference_fruit.loc[val_idx]
    if val_df.empty:
        return 1.0, 0.0, 0.0

    true = val_df["category"].astype(str).tolist()
    best: tuple[tuple[float, float, float], float, float, float] | None = None
    for alpha in candidate_alphas:
        model = _fit_mahalanobis_one_fruit(
            train_df,
            feature_cols,
            alpha=float(alpha),
            feature_weight_map=feature_weight_map,
        )
        preds, _ = model.predict_matrix(val_df[feature_cols].to_numpy(dtype=float))
        acc = float(np.mean(np.asarray(preds, dtype=str) == np.asarray(true, dtype=str)))
        f1 = macro_f1(true, preds)
        # Prefer higher F1/accuracy and, on a tie, a more regularized model.
        key = (f1, acc, float(alpha))
        if best is None or key > best[0]:
            best = (key, float(alpha), acc, f1)

    assert best is not None
    return best[1], best[2], best[3]


def fit_tuned_mahalanobis_models(
    reference_df: pd.DataFrame,
    feature_cols: list[str],
    method_name: str,
    per_fruit_weight_maps: dict[str, dict[str, float]] | None = None,
    fixed_alphas: dict[str, float] | None = None,
    seed_offset: int = 0,
) -> tuple[dict[str, MahalanobisFruitModel], pd.DataFrame]:
    """Fit one regularized Mahalanobis model per fruit.

    The final evaluation set is never used for alpha selection.
    """
    models: dict[str, MahalanobisFruitModel] = {}
    rows: list[dict] = []

    for fruit, group in reference_df.groupby("fruit", sort=True):
        weight_map = (per_fruit_weight_maps or {}).get(str(fruit), None)
        if fixed_alphas and str(fruit) in fixed_alphas:
            alpha = float(fixed_alphas[str(fruit)])
            val_acc = np.nan
            val_f1 = np.nan
        else:
            alpha, val_acc, val_f1 = _tune_alpha_for_fruit(
                group,
                feature_cols,
                feature_weight_map=weight_map,
                seed_offset=seed_offset,
            )
        model = _fit_mahalanobis_one_fruit(
            group,
            feature_cols,
            alpha=alpha,
            feature_weight_map=weight_map,
        )
        models[str(fruit)] = model
        rows.append(
            {
                "method": method_name,
                "fruit": fruit,
                "feature_count": len(feature_cols),
                "mahalanobis_alpha": alpha,
                "validation_accuracy": val_acc,
                "validation_macro_f1": val_f1,
            }
        )

    return models, pd.DataFrame(rows)


def predict_with_fruit_models(
    models: dict[str, MahalanobisFruitModel],
    evaluation_df: pd.DataFrame,
) -> tuple[list[str], list[float]]:
    pred_pairs: list[tuple[int, str]] = []
    dist_pairs: list[tuple[int, float]] = []

    for fruit, group in evaluation_df.groupby("fruit", sort=False):
        model = models.get(str(fruit))
        if model is None:
            raise ValueError(f"No fitted classifier for fruit: {fruit}")
        preds, dists = model.predict_matrix(group[model.feature_cols].to_numpy(dtype=float))
        pred_pairs.extend(zip(group.index.tolist(), preds))
        dist_pairs.extend(zip(group.index.tolist(), dists))

    pred_map = dict(pred_pairs)
    dist_map = dict(dist_pairs)
    return (
        [pred_map[i] for i in evaluation_df.index],
        [dist_map[i] for i in evaluation_df.index],
    )
