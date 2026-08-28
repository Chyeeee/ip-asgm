from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TEXTURE_CORRELATION_THRESHOLD


def fisher_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Classical Fisher discriminative score for each texture feature."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=str)
    global_mean = np.nanmean(x, axis=0)
    global_mean = np.where(np.isfinite(global_mean), global_mean, 0.0)
    numerator = np.zeros(x.shape[1], dtype=float)
    denominator = np.zeros(x.shape[1], dtype=float)

    for cls in np.unique(y):
        xc = x[y == cls]
        if xc.size == 0:
            continue
        mean_c = np.nanmean(xc, axis=0)
        var_c = np.nanvar(xc, axis=0)
        mean_c = np.where(np.isfinite(mean_c), mean_c, global_mean)
        var_c = np.where(np.isfinite(var_c), var_c, 0.0)
        numerator += len(xc) * (mean_c - global_mean) ** 2
        denominator += len(xc) * var_c

    scores = numerator / (denominator + 1e-12)
    return np.where(np.isfinite(scores), scores, 0.0)


def _safe_correlation_matrix(x: np.ndarray) -> np.ndarray:
    """Return a finite correlation matrix without divide-by-zero warnings.

    The caller removes zero-variance columns first.  Computing correlation
    explicitly here avoids NumPy's ``corrcoef`` RuntimeWarning when a small
    class-pair subset contains a constant texture descriptor.
    """
    x = np.asarray(x, dtype=float)
    n_features = x.shape[1]
    if n_features == 0:
        return np.empty((0, 0), dtype=float)
    if x.shape[0] <= 1:
        return np.eye(n_features, dtype=float)

    centered = x - np.mean(x, axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    denom = np.outer(norms, norms)
    numer = centered.T @ centered

    corr = np.zeros((n_features, n_features), dtype=float)
    np.divide(numer, denom, out=corr, where=denom > 1e-12)
    corr = np.clip(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def rank_nonredundant_texture_features(
    df: pd.DataFrame,
    texture_cols: list[str],
    corr_threshold: float = TEXTURE_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Rank texture descriptors by Fisher score and suppress near-duplicates.

    Constant or numerically invalid descriptors are removed before correlation
    analysis because they contain no discriminative information and otherwise
    cause divide-by-zero RuntimeWarnings in correlation calculations.
    """
    if not texture_cols:
        return pd.DataFrame()

    x = df[texture_cols].to_numpy(dtype=float)
    y = df["category"].astype(str).to_numpy()

    # Median-impute non-finite values first. Compute each column median only
    # from finite observations so an all-invalid column does not emit NumPy's
    # "All-NaN slice encountered" warning. Such a column receives zero and is
    # removed by the variance check below.
    med = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        finite_values = x[np.isfinite(x[:, j]), j]
        if finite_values.size:
            med[j] = float(np.median(finite_values))
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        x[bad] = np.take(med, np.where(bad)[1])

    # Correlation is undefined for a constant feature.  Such a descriptor also
    # has no class-separation information, so safely exclude it from ranking.
    feature_std = np.std(x, axis=0)
    valid = np.isfinite(feature_std) & (feature_std > 1e-12)
    if not np.any(valid):
        return pd.DataFrame()

    x = x[:, valid]
    valid_cols = [name for name, keep in zip(texture_cols, valid) if keep]

    scores = fisher_scores(x, y)
    order = np.argsort(-scores)
    corr = _safe_correlation_matrix(x)

    kept: list[int] = []
    rows: list[dict] = []
    for rank, idx in enumerate(order, start=1):
        score = float(scores[idx])
        if score <= 0 and kept:
            continue
        redundant_with = ""
        for prev in kept:
            if abs(float(corr[idx, prev])) >= corr_threshold:
                redundant_with = valid_cols[prev]
                break
        in_pool = not redundant_with
        if in_pool:
            kept.append(int(idx))
        rows.append({
            "feature": valid_cols[idx],
            "fisher_score": score,
            "initial_rank": rank,
            "nonredundant_pool": in_pool,
            "redundant_with": redundant_with,
        })

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    pool_features = [valid_cols[i] for i in kept]
    pool_rank = {f: i + 1 for i, f in enumerate(pool_features)}
    ranked["pool_rank"] = ranked["feature"].map(pool_rank)
    return ranked
