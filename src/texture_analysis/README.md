# Texture-Based Surface Quality Analysis

## Assignment Requirements

1. Download Dataset 2 - Banana Quality Dataset.
2. Use Class_A, Class_B and Defect banana images.
3. Extract GLCM and LBP features.
4. Compare their classification performance.
5. Determine the best texture technique.

## Dataset Structure

```text
project_root/
├── Fruits_Data/
│   ├── Banana/
│   │   ├── Class_A/
│   │   ├── Class_B/
│   │   └── Defect/
│   └── Guava/
│
└── src/
    └── texture_analysis/
```

Only the Banana folder is used.

## Improved Texture Pipeline

The white background is removed from texture analysis.

```text
Original
   ↓
Banana foreground detection
   ↓
Banana ROI / mask
   ↓
Grayscale
   ├─────────────┐
   ↓             ↓
Masked GLCM    Masked LBP
```

For GLCM, only neighbour pairs where BOTH pixels are inside the banana
mask are counted.

For LBP, the texture map is calculated normally but the feature
histogram only uses pixels inside the banana mask.

## Preview Before / After

Run from project root:

```bash
python -m src.texture_analysis.preview
```

This creates, for Class_A, Class_B and Defect:

### GLCM visualization

```text
Original
→ Detected Banana ROI
→ Grayscale ROI
→ GLCM Heatmap
→ Contrast / Dissimilarity / Homogeneity /
  Energy / Correlation / ASM
```

### LBP visualization

```text
Original
→ Detected Banana ROI
→ Grayscale ROI
→ LBP Texture Map
→ Normalized LBP Histogram
```

Files are saved to:

```text
src/texture_analysis/outputs/feature_preview/
```

## Full Experiment

Run:

```bash
python -m src.texture_analysis
```

The same 80/20 stratified split and the same SVM classifier are used
for GLCM and LBP.

Reported metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

The best texture technique is selected using accuracy, with Macro
F1-score as a tie-breaker.


## Latest ROI Improvement

The foreground segmentation was refined for this dataset:

- Banana colour is detected using HSV saturation.
- Small background/noise components are removed.
- Multiple bananas in one image are retained.
- External banana contours are filled so dark bruises and defects remain
  inside the ROI even when those pixels have low saturation.
- GLCM only counts neighbouring pixel pairs inside the banana mask.
- LBP builds its histogram only from banana-mask pixels.

This reduces the effect of the white/light dataset background on the
texture comparison.
