from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import cv2

# Allow: python src/texture_analysis/run_texture_analysis.py
if __package__ in (None, ""):
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    from texture_analysis.config import (
        GLCM_ANGLES_DEG,
        GLCM_BASELINE_DISTANCES,
        LBP_BASELINE,
        project_root_from_script,
    )
    from texture_analysis.enhanced_fusion import extract_enhanced_texture_features
    from texture_analysis.evaluation import (
        PROPOSED_METHOD,
        BASELINE_FUSION_METHOD,
        evaluate_methods,
        save_baseline_charts,
        save_confusion_matrices,
        save_enhancement_charts,
    )
    from texture_analysis.glcm_features import extract_glcm_features
    from texture_analysis.io_utils import (
        ensure_dirs,
        find_colour_csv,
        load_binary_mask,
        load_processed_image,
        pair_processed_and_masks,
        read_colour_features,
    )
    from texture_analysis.lbp_features import extract_lbp_features
    from texture_analysis.local_global_texture import extract_local_global_texture_features
    from texture_analysis.laws_texture import extract_laws_texture_features
    from texture_analysis.roi_quality import finalize_roi_quality, measure_roi
    from texture_analysis.visualization import save_before_after_figures
else:
    from .config import GLCM_ANGLES_DEG, GLCM_BASELINE_DISTANCES, LBP_BASELINE, project_root_from_script
    from .enhanced_fusion import extract_enhanced_texture_features
    from .evaluation import (
        PROPOSED_METHOD,
        BASELINE_FUSION_METHOD,
        evaluate_methods,
        save_baseline_charts,
        save_confusion_matrices,
        save_enhancement_charts,
    )
    from .glcm_features import extract_glcm_features
    from .io_utils import (
        ensure_dirs,
        find_colour_csv,
        load_binary_mask,
        load_processed_image,
        pair_processed_and_masks,
        read_colour_features,
    )
    from .lbp_features import extract_lbp_features
    from .local_global_texture import extract_local_global_texture_features
    from .laws_texture import extract_laws_texture_features
    from .roi_quality import finalize_roi_quality, measure_roi
    from .visualization import save_before_after_figures


BASELINE_METHODS = ["GLCM", "LBP", "Colour_Texture_Fusion"]


def _remove_stale_invention_outputs(comparison_dir: Path) -> None:
    """Remove obsolete Phase-2 artifacts while preserving reusable feature CSVs."""
    stale_names = [
        "proposed_bpartf_config.csv",
        "proposed_residual_texture_selection.csv",
        "proposed_texture_config.csv",
        "proposed_texture_selection.csv",
        "proposed_dw_mstf_fusion_confusion_matrix.png",
        "proposed_adw_mstf_fusion_confusion_matrix.png",
        "proposed_bp_artf_fusion_confusion_matrix.png",
        "proposed_bp_ptd_fusion_confusion_matrix.png",
        "proposed_lg_bpptd_config.csv",
        "proposed_local_global_texture_search.csv",
        "proposed_ptd_plus_config.csv",
        "proposed_ptd_plus_texture_search.csv",
    ]
    for name in stale_names:
        path = comparison_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass



def parse_args() -> argparse.Namespace:
    root = project_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Member 3: compare optimized GLCM, LBP and Colour+Texture Fusion, then evaluate "
            "the texture-only PTD+ enhancement"
        )
    )
    parser.add_argument("--project-root", default=str(root), help="Project root directory")
    parser.add_argument(
        "--preprocessing-dir",
        default="results/preprocessing/MedianFinal",
        help="Folder containing processed images and existing ROI masks",
    )
    parser.add_argument("--colour-csv", default=None, help="Path to Member 2 colour_features.csv")
    parser.add_argument(
        "--output-dir",
        default="results/texture_analysis",
        help="Output folder relative to project root",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional debugging limit per fruit/category")
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help=(
            "Reuse existing v5-compatible colour_texture_features.csv and "
            "multiscale_texture_candidates.csv; reuses them and computes/reuses only the local/Gabor and PTD+ Laws supplements"
        ),
    )
    parser.add_argument(
        "--rebuild-local-global",
        action="store_true",
        help="Force regeneration of only the v8 local/Gabor texture supplement.",
    )
    parser.add_argument(
        "--rebuild-laws",
        action="store_true",
        help="Force regeneration of only the PTD+ Laws texture-energy supplement.",
    )
    return parser.parse_args()


def resolve_under(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _format_results(df: pd.DataFrame) -> str:
    return df.to_string(
        index=False,
        formatters={
            "accuracy": lambda x: f"{x*100:.2f}%",
            "macro_precision": lambda x: f"{x*100:.2f}%",
            "macro_recall": lambda x: f"{x*100:.2f}%",
            "macro_f1": lambda x: f"{x*100:.2f}%",
            "model_setup_time_ms": lambda x: f"{x:.2f}",
            "classification_time_ms": lambda x: f"{x:.2f}",
            "ms_per_image": lambda x: f"{x:.4f}",
        },
    )



def extract_shape_features(mask: np.ndarray) -> dict[str, float]:
    """Extract seven fruit-shape descriptors from the binary ROI mask."""
    binary = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defaults = {
        "shape_area": 0.0,
        "shape_perimeter": 0.0,
        "shape_aspect_ratio": 0.0,
        "shape_circularity": 0.0,
        "shape_extent": 0.0,
        "shape_solidity": 0.0,
        "shape_equivalent_diameter": 0.0,
    }
    if not contours:
        return defaults

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    _, _, width, height = cv2.boundingRect(contour)

    aspect_ratio = float(width) / float(height) if height > 0 else 0.0
    circularity = (
        (4.0 * np.pi * area) / (perimeter ** 2)
        if perimeter > 0 else 0.0
    )

    bounding_area = float(width * height)
    extent = area / bounding_area if bounding_area > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    equivalent_diameter = (
        float(np.sqrt((4.0 * area) / np.pi))
        if area > 0 else 0.0
    )

    return {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_aspect_ratio": aspect_ratio,
        "shape_circularity": float(circularity),
        "shape_extent": float(extent),
        "shape_solidity": float(solidity),
        "shape_equivalent_diameter": equivalent_diameter,
    }

def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    preprocessing_dir = resolve_under(project_root, args.preprocessing_dir).resolve()
    output_dir = resolve_under(project_root, args.output_dir).resolve()
    features_dir = output_dir / "features"
    comparison_dir = output_dir / "comparison"
    before_after_dir = output_dir / "before_after"
    ensure_dirs(output_dir, features_dir, comparison_dir, before_after_dir)
    _remove_stale_invention_outputs(comparison_dir)

    overall_start = time.perf_counter()

    print("=" * 78)
    print(" OPTIMIZED TEXTURE COMPARISON & PTD+ ENHANCEMENT ")
    print("=" * 78)
    print(f"Project root       : {project_root}")
    print(f"Preprocessing input: {preprocessing_dir}")

    colour_csv = find_colour_csv(project_root, args.colour_csv)
    print(f"Colour features    : {colour_csv}")
    print(f"Output             : {output_dir}")
    print(
        "Proposed invention : Pairwise Texture Disambiguation Plus "
    )

    colour_df = read_colour_features(colour_csv)
    if args.limit:
        colour_df = (
            colour_df.groupby(["fruit", "category"], group_keys=False)
            .head(args.limit)
            .reset_index(drop=True)
        )

    id_cols = ["fruit", "category", "image", "relative_path", "processed_path", "mask_path"]
    reuse_main = output_dir / "colour_texture_features.csv"
    reuse_multi = features_dir / "multiscale_texture_candidates.csv"
    reuse_local_global = features_dir / "local_global_texture_candidates.csv"
    reuse_laws = features_dir / "laws_texture_candidates.csv"

    if args.reuse_features:
        if not reuse_main.exists() or not reuse_multi.exists():
            raise FileNotFoundError(
                "--reuse-features requires existing v5/v6 outputs:\n"
                f"  {reuse_main}\n"
                f"  {reuse_multi}"
            )
        print("Feature extraction : REUSED existing v5-compatible feature CSVs")
        base_df = pd.read_csv(reuse_main)
        multi_df = pd.read_csv(reuse_multi)
        merge_keys = [c for c in id_cols if c in base_df.columns and c in multi_df.columns]
        if not merge_keys:
            raise ValueError("Could not find common identifiers to merge cached feature CSVs.")
        enh_cols_cached = [c for c in multi_df.columns if c.startswith("enh_")]
        feature_df = base_df.merge(
            multi_df[merge_keys + enh_cols_cached].drop_duplicates(merge_keys),
            on=merge_keys,
            how="inner",
            validate="one_to_one",
        )
        if args.limit:
            feature_df = (
                feature_df.groupby(["fruit", "category"], group_keys=False)
                .head(args.limit)
                .reset_index(drop=True)
            )

        # v8 can reuse all expensive v5/v7 global descriptors. Only the new
        # local/Gabor texture supplement is computed once when absent/stale.
        supplement_start = time.perf_counter()
        need_supplement = args.rebuild_local_global or (not reuse_local_global.exists())
        local_df = None
        if not need_supplement:
            try:
                local_df = pd.read_csv(reuse_local_global)
                local_keys = [c for c in id_cols if c in feature_df.columns and c in local_df.columns]
                if not local_keys:
                    need_supplement = True
                else:
                    probe = feature_df[local_keys].merge(
                        local_df[local_keys].drop_duplicates(local_keys),
                        on=local_keys,
                        how="left",
                        indicator=True,
                    )
                    if not bool((probe["_merge"] == "both").all()):
                        need_supplement = True
            except Exception:
                need_supplement = True

        if need_supplement:
            print("Local/Gabor supplement: computing v8 texture evidence only...")
            supp_rows: list[dict] = []
            total_supp = len(feature_df)
            for n, (_, r) in enumerate(feature_df.iterrows(), start=1):
                image = load_processed_image(r["processed_path"])
                mask = load_binary_mask(r["mask_path"], image.shape[:2])
                if int(mask.sum()) < 25:
                    continue
                item = {key: r[key] for key in id_cols if key in r.index}
                item.update(extract_local_global_texture_features(image, mask))
                supp_rows.append(item)
                if n == 1 or n % 100 == 0 or n == total_supp:
                    print(f"Extracting local/Gabor supplement: {n}/{total_supp}")
            local_df = pd.DataFrame(supp_rows)
            if local_df.empty:
                raise RuntimeError("No local/Gabor texture features could be extracted.")
            local_df.to_csv(reuse_local_global, index=False)
        else:
            print("Local/Gabor supplement: REUSED existing v8 candidate CSV")

        local_keys = [c for c in id_cols if c in feature_df.columns and c in local_df.columns]
        lg_cols_cached = [c for c in local_df.columns if c.startswith("lg_")]
        before_merge = len(feature_df)
        feature_df = feature_df.merge(
            local_df[local_keys + lg_cols_cached].drop_duplicates(local_keys),
            on=local_keys,
            how="inner",
            validate="one_to_one",
        )
        if len(feature_df) != before_merge:
            raise RuntimeError(
                "Local/Gabor supplement did not cover every reusable feature row. "
                "Rerun with --rebuild-local-global."
            )

        # PTD+ Laws texture energy is a separate compact cache, so existing v8
        # local/Gabor features never need to be rebuilt just to test this invention.
        need_laws = args.rebuild_laws or (not reuse_laws.exists())
        laws_df = None
        if not need_laws:
            try:
                laws_df = pd.read_csv(reuse_laws)
                laws_keys = [c for c in id_cols if c in feature_df.columns and c in laws_df.columns]
                if not laws_keys:
                    need_laws = True
                else:
                    probe = feature_df[laws_keys].merge(
                        laws_df[laws_keys].drop_duplicates(laws_keys),
                        on=laws_keys,
                        how="left",
                        indicator=True,
                    )
                    if not bool((probe["_merge"] == "both").all()):
                        need_laws = True
            except Exception:
                need_laws = True

        if need_laws:
            print("Laws supplement       : computing PTD+ texture-energy evidence only...")
            laws_rows: list[dict] = []
            total_laws = len(feature_df)
            for n, (_, r) in enumerate(feature_df.iterrows(), start=1):
                image = load_processed_image(r["processed_path"])
                mask = load_binary_mask(r["mask_path"], image.shape[:2])
                if int(mask.sum()) < 25:
                    continue
                item = {key: r[key] for key in id_cols if key in r.index}
                item.update(extract_laws_texture_features(image, mask))
                laws_rows.append(item)
                if n == 1 or n % 100 == 0 or n == total_laws:
                    print(f"Extracting Laws supplement: {n}/{total_laws}")
            laws_df = pd.DataFrame(laws_rows)
            if laws_df.empty:
                raise RuntimeError("No Laws texture features could be extracted.")
            laws_df.to_csv(reuse_laws, index=False)
        else:
            print("Laws supplement       : REUSED existing PTD+ candidate CSV")

        laws_keys = [c for c in id_cols if c in feature_df.columns and c in laws_df.columns]
        laws_cols_cached = [c for c in laws_df.columns if c.startswith("laws_")]
        before_laws_merge = len(feature_df)
        feature_df = feature_df.merge(
            laws_df[laws_keys + laws_cols_cached].drop_duplicates(laws_keys),
            on=laws_keys,
            how="inner",
            validate="one_to_one",
        )
        if len(feature_df) != before_laws_merge:
            raise RuntimeError(
                "Laws supplement did not cover every reusable feature row. "
                "Rerun with --rebuild-laws."
            )
        extraction_elapsed = time.perf_counter() - supplement_start

        roi_path = output_dir / "roi_quality_report.csv"
        if roi_path.exists():
            roi_quality_df = pd.read_csv(roi_path)
        else:
            roi_quality_df = feature_df[id_cols].copy()
            roi_quality_df["roi_valid_for_demo"] = True
            roi_quality_df["roi_area_ratio"] = 0.0
            roi_quality_df["roi_largest_component_fraction"] = 0.0
            roi_quality_df["roi_quality_reason"] = "not_recomputed_in_reuse_mode"
        invalid_demo = (
            int((~roi_quality_df["roi_valid_for_demo"].astype(bool)).sum())
            if "roi_valid_for_demo" in roi_quality_df.columns
            else 0
        )
        # Cached CSVs may contain absolute paths from another computer.
        # Rebuild BOTH processed-image and ROI-mask paths from relative_path
        # using the current project directory before shape extraction or
        # downstream visualization/evaluation.
        corrected_processed_paths = []
        corrected_mask_paths = []
        shape_rows = []
        total_shape = len(feature_df)

        for n, (_, r) in enumerate(feature_df.iterrows(), start=1):
            relative_path = Path(str(r["relative_path"]))

            processed_path = (
                preprocessing_dir
                / "ProcessedImages"
                / relative_path
            )

            # Example:
            # Apple/Overripe/Apple_Overripe_001.jpg
            # -> Apple/Overripe/Apple_Overripe_001_mask.png
            mask_relative_path = relative_path.with_name(
                relative_path.stem + "_mask.png"
            )
            mask_path = (
                preprocessing_dir
                / "ROIMasks"
                / mask_relative_path
            )

            image = load_processed_image(processed_path)
            mask = load_binary_mask(mask_path, image.shape[:2])

            corrected_processed_paths.append(str(processed_path))
            corrected_mask_paths.append(str(mask_path))
            shape_rows.append(extract_shape_features(mask))

            if n == 1 or n % 500 == 0 or n == total_shape:
                print(f"Extracting shape features: {n}/{total_shape}")

        # Update the paths globally so later evaluation/visualization no
        # longer receives stale C:\\Users\\HP\\OneDrive... paths.
        feature_df = feature_df.copy()
        feature_df["processed_path"] = corrected_processed_paths
        feature_df["mask_path"] = corrected_mask_paths

        # Add all seven shape descriptors at once to avoid DataFrame
        # fragmentation warnings.
        shape_df = pd.DataFrame(shape_rows, index=feature_df.index)
        feature_df = pd.concat(
            [
                feature_df.drop(
                    columns=[c for c in shape_df.columns if c in feature_df.columns],
                    errors="ignore",
                ),
                shape_df,
            ],
            axis=1,
        )

        # Keep ROI-quality metadata paths consistent too, because it is passed
        # into the demo-selection/visualization stage.
        if "relative_path" in roi_quality_df.columns:
            roi_quality_df = roi_quality_df.copy()

            def _current_processed_path(value):
                rel = Path(str(value))
                return str(preprocessing_dir / "ProcessedImages" / rel)

            def _current_mask_path(value):
                rel = Path(str(value))
                mask_rel = rel.with_name(rel.stem + "_mask.png")
                return str(preprocessing_dir / "ROIMasks" / mask_rel)

            roi_quality_df["processed_path"] = roi_quality_df["relative_path"].map(
                _current_processed_path
            )
            roi_quality_df["mask_path"] = roi_quality_df["relative_path"].map(
                _current_mask_path
            )

        print("Stored image/mask paths rebuilt for current project.")
        print(f"Reused feature rows : {len(feature_df)}")
        print(f"Suspicious ROI masks: {invalid_demo}/{len(roi_quality_df)} (diagnostic only; masks unchanged)")
    else:
        paired_df, pairing_report = pair_processed_and_masks(colour_df, preprocessing_dir)
        pairing_report.to_csv(output_dir / "pairing_report.csv", index=False)
        ok = int((pairing_report["status"] == "ok").sum())
        print(f"Matched image+mask : {ok}/{len(pairing_report)}")
        if ok == 0:
            raise RuntimeError(
                "No processed image/ROI mask pairs were found. Open pairing_report.csv to inspect naming."
            )

        rows: list[dict] = []
        roi_rows: list[dict] = []
        extraction_start = time.perf_counter()
        total = len(paired_df)

        for n, (_, row) in enumerate(paired_df.iterrows(), start=1):
            image = load_processed_image(row["processed_path"])
            mask = load_binary_mask(row["mask_path"], image.shape[:2])

            quality = measure_roi(mask)
            quality_row = {
                key: row[key]
                for key in ["fruit", "category", "image", "relative_path", "processed_path", "mask_path"]
            }
            quality_row.update(quality)
            roi_rows.append(quality_row)

            if mask.sum() < 25:
                print(f"[WARN] Tiny/empty ROI skipped from feature extraction: {row['image']}")
                continue

            glcm = extract_glcm_features(
                image,
                mask,
                distances=GLCM_BASELINE_DISTANCES,
                angles=GLCM_ANGLES_DEG,
                prefix="glcm",
                keep_per_distance=False,
                include_direction_std=True,
            )
            lbp = extract_lbp_features(image, mask, settings=LBP_BASELINE, prefix="lbp")
            enhanced_texture = extract_enhanced_texture_features(image, mask)
            local_global_texture = extract_local_global_texture_features(image, mask)
            laws_texture = extract_laws_texture_features(image, mask)
            shape_features = extract_shape_features(mask)

            out = row.to_dict()
            out.update(glcm)
            out.update(lbp)
            out.update(enhanced_texture)
            out.update(local_global_texture)
            out.update(laws_texture)
            out.update(shape_features)
            rows.append(out)

            if n == 1 or n % 100 == 0 or n == total:
                print(f"Extracting features: {n}/{total}")

        extraction_elapsed = time.perf_counter() - extraction_start
        feature_df = pd.DataFrame(rows)
        if feature_df.empty:
            raise RuntimeError("No valid feature rows were extracted.")

        roi_quality_df = finalize_roi_quality(pd.DataFrame(roi_rows))
        roi_quality_df.to_csv(output_dir / "roi_quality_report.csv", index=False)
        invalid_demo = int((~roi_quality_df["roi_valid_for_demo"].astype(bool)).sum())
        print(f"Suspicious ROI masks: {invalid_demo}/{len(roi_quality_df)} (diagnostic only; masks unchanged)")

    colour_meta = [c for c in ["colour_space", "feature_version"] if c in feature_df.columns]
    glcm_cols = [c for c in feature_df.columns if c.startswith("glcm_")]
    lbp_cols = [c for c in feature_df.columns if c.startswith("lbp_")]
    enh_glcm_cols = [c for c in feature_df.columns if c.startswith("enh_glcm_")]
    enh_lbp_cols = [c for c in feature_df.columns if c.startswith("enh_lbp_")]
    local_global_cols = [c for c in feature_df.columns if c.startswith("lg_")]
    laws_cols = [c for c in feature_df.columns if c.startswith("laws_")]
    shape_cols = [c for c in feature_df.columns if c.startswith("shape_")]
    original_colour_cols = [
        c
        for c in colour_df.columns
        if c not in {"fruit", "category", "image", "relative_path", "colour_space", "feature_version"}
    ]

    # Official accumulated output required by the assignment.
    main_colour_texture_cols = (
        id_cols + colour_meta + original_colour_cols + glcm_cols + lbp_cols + shape_cols
    )
    feature_df[main_colour_texture_cols].to_csv(
        output_dir / "colour_texture_features.csv", index=False
    )

    # Supporting raw/candidate feature files.
    feature_df[id_cols + shape_cols].to_csv(
        features_dir / "shape_features.csv", index=False
    )
    feature_df[id_cols + glcm_cols].to_csv(features_dir / "glcm_features.csv", index=False)
    feature_df[id_cols + lbp_cols].to_csv(features_dir / "lbp_features.csv", index=False)
    feature_df[id_cols + enh_glcm_cols + enh_lbp_cols].to_csv(
        features_dir / "multiscale_texture_candidates.csv", index=False
    )
    feature_df[id_cols + local_global_cols].to_csv(
        features_dir / "local_global_texture_candidates.csv", index=False
    )
    feature_df[id_cols + laws_cols].to_csv(
        features_dir / "laws_texture_candidates.csv", index=False
    )

    print("\nEvaluating with the same final test split for all methods...")
    (
        all_results,
        predictions_df,
        all_per_fruit,
        proposed_config,
        texture_selection,
        baseline_classifier_config,
    ) = evaluate_methods(feature_df.reset_index(drop=True))

    # ------------------------------------------------------------------
    # PHASE 1: ONLY the three assignment-required algorithms.
    # ------------------------------------------------------------------
    baseline_results = all_results[all_results["method"].isin(BASELINE_METHODS)].copy()
    baseline_per_fruit = all_per_fruit[all_per_fruit["method"].isin(BASELINE_METHODS)].copy()
    baseline_results.to_csv(comparison_dir / "algorithm_comparison.csv", index=False)
    baseline_per_fruit.to_csv(comparison_dir / "per_fruit_accuracy.csv", index=False)
    save_baseline_charts(baseline_results, baseline_per_fruit, comparison_dir)

    best_baseline = baseline_results.sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).iloc[0]
    best_baseline_method = str(best_baseline["method"])

    # ------------------------------------------------------------------
    # PHASE 2: strong Colour+Texture baseline vs proposed PTD+.
    # PTD+ preserves the v8 pairwise correction and admits Laws enrichment only when CV proves a stricter gain. In the
    # full assignment dataset this is expected to be the Phase-1 winner; if a
    # different method wins, Phase 1 still reports that result honestly.
    # ------------------------------------------------------------------
    enhancement_baseline_method = (
        BASELINE_FUSION_METHOD
        if BASELINE_FUSION_METHOD in set(baseline_results["method"])
        else best_baseline_method
    )
    enhancement_baseline = baseline_results[
        baseline_results["method"] == enhancement_baseline_method
    ].iloc[0]
    enhancement_results = all_results[
        all_results["method"].isin([enhancement_baseline_method, PROPOSED_METHOD])
    ].copy()
    enhancement_per_fruit = all_per_fruit[
        all_per_fruit["method"].isin([enhancement_baseline_method, PROPOSED_METHOD])
    ].copy()
    enhancement_results.to_csv(comparison_dir / "enhancement_comparison.csv", index=False)
    enhancement_per_fruit.to_csv(
        comparison_dir / "enhancement_per_fruit_accuracy.csv", index=False
    )
    proposed_config.to_csv(comparison_dir / "proposed_ptd_plus_config.csv", index=False)
    enabled_rules = (
        int(proposed_config["enabled"].astype(bool).sum())
        if (not proposed_config.empty and "enabled" in proposed_config.columns)
        else 0
    )
    laws_rules = (
        int((proposed_config["enabled"].astype(bool) & proposed_config["laws_enriched"].astype(bool)).sum())
        if (not proposed_config.empty and {"enabled", "laws_enriched"}.issubset(proposed_config.columns))
        else 0
    )
    baseline_classifier_config.to_csv(comparison_dir / "baseline_classifier_config.csv", index=False)
    texture_selection.to_csv(comparison_dir / "proposed_ptd_plus_texture_search.csv", index=False)
    save_enhancement_charts(enhancement_results, enhancement_per_fruit, comparison_dir)

    predictions_df.to_csv(comparison_dir / "evaluation_predictions.csv", index=False)
    save_confusion_matrices(predictions_df, comparison_dir)

    # Demo image selection now rejects suspicious/tiny ROI examples when a
    # valid alternative exists. This automatically replaces the bad Grape demo.
    demos = save_before_after_figures(
        predictions_df,
        before_after_dir,
        best_baseline_method=enhancement_baseline_method,
        roi_quality_df=roi_quality_df,
    )
    demos.to_csv(before_after_dir / "selected_demo_samples.csv", index=False)

    proposed = all_results[all_results["method"] == PROPOSED_METHOD].iloc[0]
    acc_gain = (proposed["accuracy"] - enhancement_baseline["accuracy"]) * 100.0
    f1_gain = (proposed["macro_f1"] - enhancement_baseline["macro_f1"]) * 100.0

    summary_df = pd.DataFrame(
        [
            {
                "best_baseline": enhancement_baseline_method,
                "baseline_accuracy": enhancement_baseline["accuracy"],
                "baseline_macro_f1": enhancement_baseline["macro_f1"],
                "proposed_method": PROPOSED_METHOD,
                "proposed_accuracy": proposed["accuracy"],
                "proposed_macro_f1": proposed["macro_f1"],
                "accuracy_change_percentage_points": acc_gain,
                "macro_f1_change_percentage_points": f1_gain,
                "improved": bool(proposed["macro_f1"] > enhancement_baseline["macro_f1"]),
            }
        ]
    )
    summary_df.to_csv(comparison_dir / "enhancement_summary.csv", index=False)

    total_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 78)
    print("PHASE 1 - THREE REQUIRED ALGORITHMS")
    print("=" * 78)
    print(_format_results(baseline_results))
    print(f"\nBest baseline      : {best_baseline_method}")
    print(f"Baseline Macro F1  : {best_baseline['macro_f1']*100:.2f}%")

    print("\n" + "=" * 78)
    print("PHASE 2 - COLOUR+TEXTURE BASELINE VS PROPOSED PTD+")
    print("=" * 78)
    print(_format_results(enhancement_results))
    print(f"\nProposed Macro F1  : {proposed['macro_f1']*100:.2f}%")
    print(f"Accuracy change    : {acc_gain:+.2f} percentage points")
    print(f"Macro F1 change    : {f1_gain:+.2f} percentage points")
    print(f"Enabled pair rules : {enabled_rules}")
    print(f"Laws-enriched rules: {laws_rules}")
    if proposed["macro_f1"] > enhancement_baseline["macro_f1"]:
        print("Result             : Proposed PTD+ IMPROVED Macro F1.")
    else:
        print(
            "Result             : Proposed method did not improve final Macro F1; "
            "report the result honestly and inspect proposed_ptd_plus_config.csv."
        )

    print("\nTiming note        : model setup/tuning is reported separately from final inference.")
    print(f"Feature extraction : {extraction_elapsed:.2f} s")
    print(f"Total run time      : {total_elapsed:.2f} s")
    print(f"Main CSV           : {output_dir / 'colour_texture_features.csv'}")
    print(f"ROI quality report : {output_dir / 'roi_quality_report.csv'}")
    print(f"Results saved to   : {output_dir}")


if __name__ == "__main__":
    main()
