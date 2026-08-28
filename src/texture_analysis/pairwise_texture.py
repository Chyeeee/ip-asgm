from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .classifier import MahalanobisFruitModel, _fit_mahalanobis_one_fruit
from .config import (
    LAWS_MIN_EXTRA_NET_GAIN,
    LAWS_REQUIRE_NO_MORE_HARM,
    LAWS_REQUIRE_POSITIVE_FOLDS_NOT_LOWER,
    PAIRWISE_TEXTURE_ALPHA_CANDIDATES,
    PAIRWISE_BASE_CONFIDENCE_THRESHOLDS,
    PAIRWISE_CV_FOLDS,
    PAIRWISE_MIN_CHANGED,
    PAIRWISE_MIN_NET_GAIN,
    PAIRWISE_MIN_CORRECTION_PRECISION,
    PAIRWISE_REQUIRED_POSITIVE_FOLDS,
    PAIRWISE_TEXTURE_CONFIDENCE_THRESHOLDS,
    PAIRWISE_TOP_K_CANDIDATES,
    RANDOM_SEED,
)
from .discriminative_texture import rank_nonredundant_texture_features


def residual_texture_candidates(texture_cols: list[str]) -> list[str]:
    """All texture-only specialist evidence available to PTD+."""
    return [
        c for c in texture_cols
        if c.startswith("enh_glcm_")
        or c.startswith("enh_lbp_")
        or c.startswith("lg_")
        or c.startswith("laws_")
    ]


def texture_feature_groups(texture_cols: list[str]) -> dict[str, list[str]]:
    """Original v8 groups plus optional Laws-enriched versions.

    The original four groups are deliberately unchanged. PTD+ compares them
    against enriched variants only within reference-data cross-validation.
    """
    global_cols = [c for c in texture_cols if c.startswith("enh_glcm_") or c.startswith("enh_lbp_")]
    local_cols = [c for c in texture_cols if c.startswith("lg_local_")]
    gabor_cols = [c for c in texture_cols if c.startswith("lg_gabor_")]
    laws_cols = [c for c in texture_cols if c.startswith("laws_")]

    original = {
        "global_multiscale": global_cols,
        "local_heterogeneity": local_cols,
        "gabor": gabor_cols,
        "all_texture": global_cols + local_cols + gabor_cols,
    }
    groups = dict(original)
    if laws_cols:
        for name, cols in original.items():
            if cols:
                groups[f"{name}_plus_laws"] = cols + laws_cols
    return {k: list(dict.fromkeys(v)) for k, v in groups.items() if v}


def _uses_laws_group(feature_group: str) -> bool:
    return str(feature_group).endswith("_plus_laws")


def _passes_pairwise_gate(cfg: dict | None, fold_count: int) -> bool:
    if cfg is None:
        return False
    required_positive = min(PAIRWISE_REQUIRED_POSITIVE_FOLDS, fold_count)
    return bool(
        cfg["cv_net_gain"] >= PAIRWISE_MIN_NET_GAIN
        and cfg["cv_changed"] >= PAIRWISE_MIN_CHANGED
        and cfg["positive_folds"] >= required_positive
        and cfg["cv_harmful"] < cfg["cv_helpful"]
        and cfg["correction_precision"] >= PAIRWISE_MIN_CORRECTION_PRECISION
    )


def _fruit_seed(fruit: str, offset: int = 0) -> int:
    return RANDOM_SEED + 17000 + offset + sum((i + 1) * ord(ch) for i, ch in enumerate(str(fruit)))


def _supports_from_distances(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(distances, dtype=float)
    shifted = d - np.min(d, axis=1, keepdims=True)
    positive = np.where(shifted > 1e-12, shifted, np.nan)
    scale = np.nanmedian(positive, axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    logits = -shifted / scale[:, None]
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -60.0, 60.0))
    support = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    order = np.argsort(-support, axis=1)
    best = order[:, 0]
    second = order[:, 1] if support.shape[1] > 1 else order[:, 0]
    confidence = support[np.arange(len(support)), best] - support[np.arange(len(support)), second]
    return support, confidence, best, second


def _unordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _stratified_kfold_indices(df: pd.DataFrame, n_splits: int, seed: int) -> list[np.ndarray]:
    """Small deterministic stratified folds without adding an ML dependency."""
    n_splits = max(2, int(n_splits))
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    rng = np.random.default_rng(seed)
    for _, group in df.groupby("category", sort=True):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        for pos, item in enumerate(idx):
            folds[pos % n_splits].append(int(item))
    return [np.asarray(sorted(x), dtype=int) for x in folds]


def _pair_candidate_mask(
    classes: list[str], best: np.ndarray, second: np.ndarray, pair: tuple[str, str]
) -> np.ndarray:
    names_best = np.asarray([classes[int(i)] for i in best], dtype=object)
    names_second = np.asarray([classes[int(i)] for i in second], dtype=object)
    return np.asarray(
        [_unordered_pair(a, b) == pair for a, b in zip(names_best, names_second)],
        dtype=bool,
    )


def _evaluate_rule(
    true: np.ndarray,
    baseline_pred: np.ndarray,
    baseline_conf: np.ndarray,
    pair_mask: np.ndarray,
    texture_pred: np.ndarray,
    texture_conf: np.ndarray,
    base_threshold: float,
    texture_threshold: float,
) -> tuple[int, int, int, np.ndarray]:
    """Return helpful, harmful, changed and corrected predictions for one pair rule."""
    pred = baseline_pred.copy()
    gate_local = (
        pair_mask
        & (baseline_conf <= float(base_threshold))
        & (texture_conf >= float(texture_threshold))
        & (texture_pred != baseline_pred)
    )
    pred[gate_local] = texture_pred[gate_local]
    changed = gate_local & (pred != baseline_pred)
    helpful = int(np.sum(changed & (baseline_pred != true) & (pred == true)))
    harmful = int(np.sum(changed & (baseline_pred == true) & (pred != true)))
    return helpful, harmful, int(np.sum(changed)), pred


@dataclass
class PairwiseTextureRule:
    pair: tuple[str, str]
    feature_group: str
    texture_model: MahalanobisFruitModel
    selected_features: list[str]
    base_confidence_threshold: float
    texture_confidence_threshold: float
    cv_helpful: int
    cv_harmful: int
    cv_changed: int
    positive_folds: int


@dataclass
class PairwiseFruitModel:
    fruit: str
    baseline_model: MahalanobisFruitModel
    rules: dict[tuple[str, str], PairwiseTextureRule]

    def predict_dataframe(self, df: pd.DataFrame) -> tuple[list[str], list[float], dict[str, np.ndarray]]:
        if df.empty:
            return [], [], {
                "baseline_confidence": np.array([], dtype=float),
                "pairwise_rule_available": np.array([], dtype=bool),
                "texture_confidence": np.array([], dtype=float),
                "texture_correction_applied": np.array([], dtype=bool),
            }

        x = df[self.baseline_model.feature_cols].to_numpy(dtype=float)
        bdist = self.baseline_model.distance_matrix(x)
        _, bconf, bbest, bsecond = _supports_from_distances(bdist)
        classes = self.baseline_model.classes
        baseline_pred = np.asarray([classes[int(i)] for i in bbest], dtype=object)
        preds = baseline_pred.copy()
        best_dist = bdist[np.arange(len(bdist)), bbest].astype(float)

        meta = {
            "baseline_confidence": bconf.copy(),
            "pairwise_rule_available": np.zeros(len(df), dtype=bool),
            "texture_confidence": np.zeros(len(df), dtype=float),
            "texture_correction_applied": np.zeros(len(df), dtype=bool),
        }

        # Each sample has exactly one baseline top-2 class pair, so rules cannot
        # fight one another. The strong baseline remains the default prediction.
        pair_to_rows: dict[tuple[str, str], list[int]] = {}
        for i, (a, b) in enumerate(zip(bbest, bsecond)):
            pair = _unordered_pair(classes[int(a)], classes[int(b)])
            if pair in self.rules:
                pair_to_rows.setdefault(pair, []).append(i)

        for pair, rows in pair_to_rows.items():
            rule = self.rules[pair]
            idx = np.asarray(rows, dtype=int)
            meta["pairwise_rule_available"][idx] = True
            candidate = idx[bconf[idx] <= rule.base_confidence_threshold]
            if candidate.size == 0:
                continue
            sub = df.iloc[candidate]
            tdist = rule.texture_model.distance_matrix(
                sub[rule.selected_features].to_numpy(dtype=float)
            )
            _, tconf, tbest, _ = _supports_from_distances(tdist)
            tclasses = rule.texture_model.classes
            tpred = np.asarray([tclasses[int(i)] for i in tbest], dtype=object)
            meta["texture_confidence"][candidate] = tconf
            gate = (tconf >= rule.texture_confidence_threshold) & (tpred != preds[candidate])
            if not np.any(gate):
                continue
            changed_idx = candidate[gate]
            preds[changed_idx] = tpred[gate]
            meta["texture_correction_applied"][changed_idx] = True

        return preds.astype(str).tolist(), best_dist.tolist(), meta


def _cross_validate_pair_rule(
    reference_fruit: pd.DataFrame,
    pair: tuple[str, str],
    baseline_alpha: float,
    baseline_cols: list[str],
    residual_cols: list[str],
) -> tuple[dict | None, list[dict]]:
    """Select a v8-compatible rule, enriching it with Laws only when safer/better.

    This function intentionally reproduces the original v8 search for the four
    non-Laws feature groups. Laws-enriched variants are searched in parallel.
    When both pass the usual validation gates, the enriched rule is accepted
    only if it has a strictly larger CV net correction, no additional harmful
    corrections, and no reduction in the number of positive folds.
    """
    folds = _stratified_kfold_indices(
        reference_fruit,
        PAIRWISE_CV_FOLDS,
        _fruit_seed(str(reference_fruit["fruit"].iloc[0])),
    )
    all_index = reference_fruit.index.to_numpy(dtype=int)
    diagnostics: list[dict] = []
    candidate_groups = texture_feature_groups(residual_cols)
    if not candidate_groups:
        return None, diagnostics

    fold_cache: list[dict] = []
    for fold_no, val_idx in enumerate(folds):
        if len(val_idx) == 0:
            continue
        train_idx = np.setdiff1d(all_index, val_idx, assume_unique=False)
        train_df = reference_fruit.loc[train_idx]
        val_df = reference_fruit.loc[val_idx]
        train_pair = train_df[train_df["category"].astype(str).isin(pair)].copy()
        if train_pair["category"].nunique() < 2:
            continue

        baseline = _fit_mahalanobis_one_fruit(train_df, baseline_cols, alpha=baseline_alpha)
        bdist = baseline.distance_matrix(val_df[baseline_cols].to_numpy(dtype=float))
        _, bconf, bbest, bsecond = _supports_from_distances(bdist)
        classes = baseline.classes
        bpred = np.asarray([classes[int(i)] for i in bbest], dtype=object)
        true = val_df["category"].astype(str).to_numpy(dtype=object)
        pair_mask = _pair_candidate_mask(classes, bbest, bsecond, pair)

        texture_variants: dict[tuple[str, int, float], tuple[list[str], np.ndarray, np.ndarray]] = {}
        for group_name, group_cols in candidate_groups.items():
            ranked = rank_nonredundant_texture_features(train_pair, group_cols)
            if ranked.empty:
                continue
            pool = ranked.loc[ranked["nonredundant_pool"], "feature"].tolist()
            if not pool:
                pool = ranked["feature"].tolist()
            if not pool:
                continue
            top_values = sorted(
                set(min(k, len(pool)) for k in PAIRWISE_TOP_K_CANDIDATES if min(k, len(pool)) > 0)
            )
            for top_k in top_values:
                selected = pool[:top_k]
                for alpha in PAIRWISE_TEXTURE_ALPHA_CANDIDATES:
                    tex_model = _fit_mahalanobis_one_fruit(
                        train_pair, selected, alpha=float(alpha)
                    )
                    tdist = tex_model.distance_matrix(val_df[selected].to_numpy(dtype=float))
                    _, tconf, tbest, _ = _supports_from_distances(tdist)
                    tclasses = tex_model.classes
                    tpred = np.asarray([tclasses[int(i)] for i in tbest], dtype=object)
                    texture_variants[(group_name, int(top_k), float(alpha))] = (
                        list(selected), tconf, tpred
                    )

        fold_cache.append({
            "fold_no": fold_no,
            "true": true,
            "baseline_pred": bpred,
            "baseline_conf": bconf,
            "pair_mask": pair_mask,
            "variants": texture_variants,
        })

    if len(fold_cache) < 2:
        return None, diagnostics

    best_original: tuple[tuple, dict] | None = None
    best_laws: tuple[tuple, dict] | None = None
    variant_keys = sorted(set(k for fc in fold_cache for k in fc["variants"].keys()))
    for feature_group, top_k, alpha in variant_keys:
        uses_laws = _uses_laws_group(feature_group)
        for bthr in PAIRWISE_BASE_CONFIDENCE_THRESHOLDS:
            for tthr in PAIRWISE_TEXTURE_CONFIDENCE_THRESHOLDS:
                total_helpful = total_harmful = total_changed = 0
                fold_net: list[int] = []
                valid_folds = 0
                for fc in fold_cache:
                    variant = fc["variants"].get((feature_group, top_k, alpha))
                    if variant is None:
                        continue
                    _, tconf, tpred = variant
                    helpful, harmful, changed, _ = _evaluate_rule(
                        fc["true"], fc["baseline_pred"], fc["baseline_conf"],
                        fc["pair_mask"], tpred, tconf, bthr, tthr,
                    )
                    total_helpful += helpful
                    total_harmful += harmful
                    total_changed += changed
                    fold_net.append(helpful - harmful)
                    valid_folds += 1
                if valid_folds < 2:
                    continue

                net = total_helpful - total_harmful
                positive_folds = sum(x > 0 for x in fold_net)
                nonnegative_folds = sum(x >= 0 for x in fold_net)
                correction_precision = total_helpful / max(1, total_helpful + total_harmful)
                diagnostics.append({
                    "pair": f"{pair[0]}|{pair[1]}",
                    "feature_group": feature_group,
                    "laws_enriched": uses_laws,
                    "top_k": top_k,
                    "texture_alpha": alpha,
                    "base_confidence_threshold": bthr,
                    "texture_confidence_threshold": tthr,
                    "cv_helpful": total_helpful,
                    "cv_harmful": total_harmful,
                    "cv_net_gain": net,
                    "cv_changed": total_changed,
                    "correction_precision": correction_precision,
                    "positive_folds": positive_folds,
                    "nonnegative_folds": nonnegative_folds,
                })

                # This is exactly the v8 ordering. Original and Laws-enriched
                # candidates are ranked separately so Laws cannot silently
                # replace a proven v8 rule on a tie.
                key = (
                    net,
                    positive_folds,
                    correction_precision,
                    -total_harmful,
                    total_helpful,
                    -total_changed,
                    -top_k,
                    tthr,
                    -bthr,
                )
                cfg = {
                    "feature_group": feature_group,
                    "laws_enriched": bool(uses_laws),
                    "top_k": int(top_k),
                    "texture_alpha": float(alpha),
                    "base_confidence_threshold": float(bthr),
                    "texture_confidence_threshold": float(tthr),
                    "cv_helpful": int(total_helpful),
                    "cv_harmful": int(total_harmful),
                    "cv_net_gain": int(net),
                    "cv_changed": int(total_changed),
                    "correction_precision": float(correction_precision),
                    "positive_folds": int(positive_folds),
                    "nonnegative_folds": int(nonnegative_folds),
                }
                target = best_laws if uses_laws else best_original
                if target is None or key > target[0]:
                    if uses_laws:
                        best_laws = (key, cfg)
                    else:
                        best_original = (key, cfg)

    original_cfg = best_original[1] if best_original is not None else None
    laws_cfg = best_laws[1] if best_laws is not None else None
    original_ok = _passes_pairwise_gate(original_cfg, len(fold_cache))
    laws_ok = _passes_pairwise_gate(laws_cfg, len(fold_cache))

    # If v8 already has a valid rule, preserve it unless Laws demonstrates a
    # strictly stronger and at least equally safe correction pattern.
    if original_ok:
        if laws_ok:
            stronger_net = laws_cfg["cv_net_gain"] >= original_cfg["cv_net_gain"] + LAWS_MIN_EXTRA_NET_GAIN
            no_more_harm = (
                laws_cfg["cv_harmful"] <= original_cfg["cv_harmful"]
                if LAWS_REQUIRE_NO_MORE_HARM else True
            )
            folds_not_lower = (
                laws_cfg["positive_folds"] >= original_cfg["positive_folds"]
                if LAWS_REQUIRE_POSITIVE_FOLDS_NOT_LOWER else True
            )
            if stronger_net and no_more_harm and folds_not_lower:
                laws_cfg = dict(laws_cfg)
                laws_cfg["selection_reason"] = "laws_strictly_improved_v8_rule"
                laws_cfg["v8_cv_net_gain"] = int(original_cfg["cv_net_gain"])
                laws_cfg["v8_cv_harmful"] = int(original_cfg["cv_harmful"])
                return laws_cfg, diagnostics

        original_cfg = dict(original_cfg)
        original_cfg["selection_reason"] = "preserved_v8_rule"
        original_cfg["v8_cv_net_gain"] = int(original_cfg["cv_net_gain"])
        original_cfg["v8_cv_harmful"] = int(original_cfg["cv_harmful"])
        return original_cfg, diagnostics

    # If v8 had no validated rule for this pair, PTD+ may enable a new one only
    # when the Laws-enriched specialist independently passes all normal gates.
    if laws_ok:
        laws_cfg = dict(laws_cfg)
        laws_cfg["selection_reason"] = "laws_enabled_new_validated_pair"
        laws_cfg["v8_cv_net_gain"] = int(original_cfg["cv_net_gain"]) if original_cfg else 0
        laws_cfg["v8_cv_harmful"] = int(original_cfg["cv_harmful"]) if original_cfg else 0
        return laws_cfg, diagnostics

    return None, diagnostics

def fit_pairwise_residual_models(
    reference_df: pd.DataFrame,
    baseline_models: dict[str, MahalanobisFruitModel],
    baseline_cols: list[str],
    texture_candidates: list[str],
) -> tuple[dict[str, PairwiseFruitModel], pd.DataFrame, pd.DataFrame]:
    residual_cols = residual_texture_candidates(texture_candidates)
    if not residual_cols:
        raise ValueError("No residual multi-scale texture columns were found.")

    models: dict[str, PairwiseFruitModel] = {}
    config_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for fruit, group in reference_df.groupby("fruit", sort=True):
        fruit = str(fruit)
        base_final = baseline_models[fruit]
        classes = sorted(group["category"].astype(str).unique().tolist())
        rules: dict[tuple[str, str], PairwiseTextureRule] = {}

        for pair in combinations(classes, 2):
            pair = _unordered_pair(*pair)
            cfg, diagnostics = _cross_validate_pair_rule(
                group, pair, base_final.alpha, baseline_cols, residual_cols
            )
            for row in diagnostics:
                row["fruit"] = fruit
                diagnostic_rows.append(row)

            if cfg is None:
                config_rows.append({
                    "fruit": fruit,
                    "class_pair": f"{pair[0]}|{pair[1]}",
                    "enabled": False,
                    "reason": "pairwise_texture_not_cross_validated_better",
                    "feature_group": "",
                    "laws_enriched": False,
                    "selected_texture_count": 0,
                    "selected_laws_count": 0,
                    "total_effective_feature_count": len(baseline_cols),
                })
                continue

            pair_reference = group[group["category"].astype(str).isin(pair)].copy()
            group_cols = texture_feature_groups(residual_cols).get(cfg["feature_group"], residual_cols)
            ranked = rank_nonredundant_texture_features(pair_reference, group_cols)
            pool = ranked.loc[ranked["nonredundant_pool"], "feature"].tolist()
            if not pool:
                pool = ranked["feature"].tolist()
            selected = pool[: min(cfg["top_k"], len(pool))]
            if not selected:
                continue

            texture_model = _fit_mahalanobis_one_fruit(
                pair_reference, selected, alpha=cfg["texture_alpha"]
            )
            rule = PairwiseTextureRule(
                pair=pair,
                feature_group=cfg["feature_group"],
                texture_model=texture_model,
                selected_features=list(selected),
                base_confidence_threshold=cfg["base_confidence_threshold"],
                texture_confidence_threshold=cfg["texture_confidence_threshold"],
                cv_helpful=cfg["cv_helpful"],
                cv_harmful=cfg["cv_harmful"],
                cv_changed=cfg["cv_changed"],
                positive_folds=cfg["positive_folds"],
            )
            rules[pair] = rule
            config_rows.append({
                "fruit": fruit,
                "class_pair": f"{pair[0]}|{pair[1]}",
                "enabled": True,
                "reason": cfg.get("selection_reason", "cross_validated_pairwise_texture_gain"),
                "feature_group": rule.feature_group,
                "laws_enriched": bool(cfg.get("laws_enriched", False)),
                "selected_texture_count": len(selected),
                "selected_laws_count": sum(str(f).startswith("laws_") for f in selected),
                "selected_texture_features": ";".join(selected),
                "base_confidence_threshold": rule.base_confidence_threshold,
                "texture_confidence_threshold": rule.texture_confidence_threshold,
                "texture_alpha": rule.texture_model.alpha,
                "cv_helpful": rule.cv_helpful,
                "cv_harmful": rule.cv_harmful,
                "cv_net_gain": rule.cv_helpful - rule.cv_harmful,
                "cv_changed": rule.cv_changed,
                "correction_precision": cfg.get("correction_precision", 0.0),
                "positive_folds": rule.positive_folds,
                "v8_cv_net_gain": cfg.get("v8_cv_net_gain", cfg.get("cv_net_gain", 0)),
                "v8_cv_harmful": cfg.get("v8_cv_harmful", cfg.get("cv_harmful", 0)),
                "total_effective_feature_count": len(baseline_cols) + len(selected),
            })

        models[fruit] = PairwiseFruitModel(
            fruit=fruit,
            baseline_model=base_final,
            rules=rules,
        )

    return models, pd.DataFrame(config_rows), pd.DataFrame(diagnostic_rows)


def predict_pairwise_residual_models(
    models: dict[str, PairwiseFruitModel],
    df: pd.DataFrame,
) -> tuple[list[str], list[float], pd.DataFrame]:
    preds = np.empty(len(df), dtype=object)
    dists = np.zeros(len(df), dtype=float)
    baseline_conf = np.zeros(len(df), dtype=float)
    rule_available = np.zeros(len(df), dtype=bool)
    texture_conf = np.zeros(len(df), dtype=float)
    corrected = np.zeros(len(df), dtype=bool)

    for fruit, group in df.groupby("fruit", sort=False):
        model = models[str(fruit)]
        local_pred, local_dist, meta = model.predict_dataframe(group)
        pos = df.index.get_indexer(group.index)
        preds[pos] = np.asarray(local_pred, dtype=object)
        dists[pos] = np.asarray(local_dist, dtype=float)
        baseline_conf[pos] = meta["baseline_confidence"]
        rule_available[pos] = meta["pairwise_rule_available"]
        texture_conf[pos] = meta["texture_confidence"]
        corrected[pos] = meta["texture_correction_applied"]

    meta_df = pd.DataFrame({
        "baseline_confidence": baseline_conf,
        "pairwise_rule_available": rule_available,
        "texture_confidence": texture_conf,
        "texture_correction_applied": corrected,
    })
    return preds.astype(str).tolist(), dists.tolist(), meta_df
