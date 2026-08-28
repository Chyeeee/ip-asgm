# Member 3 — Texture-Based Surface Quality Analysis (PTD+ v11)

## Phase 1 — required algorithms

The three required baselines are unchanged and use the same final split plus the same classical regularized Mahalanobis minimum-distance classifier:

1. **GLCM** — 16 descriptors (8 properties × directional mean/std)
2. **LBP** — 14 descriptors (uniform histogram + histogram statistics)
3. **Colour + Texture Feature Fusion** — Member 2 colour features + baseline GLCM + baseline LBP

Member 2 colour descriptors are read unchanged. Member 3 does not alter preprocessing, segmentation, morphology, ROI masks, or colour extraction.

## Phase 2 — proposed invention

### PTD+ — Pairwise Texture Disambiguation Plus

PTD+ is built directly from the successful v8.1 pairwise method. The strong Colour+Texture prediction stays as the default. A texture specialist may change a prediction only for a validated difficult top-two class pair.

The original v8.1 specialist families remain exactly available:

- `global_multiscale` — enhanced GLCM + enhanced LBP
- `local_heterogeneity` — fixed 3×3 ROI-grid texture variation
- `gabor` — multi-scale, multi-orientation Gabor responses
- `all_texture` — global + local + Gabor

PTD+ adds **Laws Texture Energy** only as optional enrichment:

- `global_multiscale_plus_laws`
- `local_heterogeneity_plus_laws`
- `gabor_plus_laws`
- `all_texture_plus_laws`

For each fruit/class pair, the original v8.1 rule and the Laws-enriched candidate are evaluated only on reference-data cross-validation. An existing v8.1 rule is replaced by a Laws-enriched rule only when the enriched rule:

1. gains at least one additional net correction,
2. introduces no additional harmful corrections, and
3. has at least as many positive CV folds.

If those conditions are not met, PTD+ keeps the original v8.1 rule. The final evaluation split is never used for rule or feature selection.

## Laws descriptor

PTD+ uses the classic five Laws vectors (`L5`, `E5`, `S5`, `W5`, `R5`). The ROI grayscale image is locally mean-removed and normalized before 5×5 Laws filtering. Symmetric filter-pair RMS energies produce a compact texture-only descriptor. `L5L5` is excluded because it mainly represents local level rather than texture.

## Warning-safe feature ranking

Constant/zero-variance descriptors are removed before the correlation-based redundancy check, and a safe finite correlation matrix is used. This avoids NumPy divide-by-zero correlation warnings on small class-pair subsets.

## Fastest way to run PTD+

Keep your existing `results/texture_analysis/` folder and run:

```powershell
python src/texture_analysis/run_texture_analysis.py `
  --colour-csv results/colour_analysis/colour_features.csv `
  --reuse-features
```

The first PTD+ run reuses your existing baseline/global and local/Gabor CSVs and computes only:

```text
results/texture_analysis/features/laws_texture_candidates.csv
```

Later runs reuse that Laws CSV too.

To intentionally regenerate only Laws features:

```powershell
python src/texture_analysis/run_texture_analysis.py `
  --colour-csv results/colour_analysis/colour_features.csv `
  --reuse-features `
  --rebuild-laws
```

## Important outputs

```text
results/texture_analysis/colour_texture_features.csv
results/texture_analysis/features/laws_texture_candidates.csv
results/texture_analysis/comparison/algorithm_comparison.csv
results/texture_analysis/comparison/enhancement_comparison.csv
results/texture_analysis/comparison/proposed_ptd_plus_config.csv
results/texture_analysis/comparison/proposed_ptd_plus_texture_search.csv
results/texture_analysis/comparison/evaluation_predictions.csv
```

`proposed_ptd_plus_config.csv` includes `laws_enriched` and `selected_laws_count`, so you can see exactly which class-pair rules actually benefited from Laws texture energy.
