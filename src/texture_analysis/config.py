from pathlib import Path

# -----------------------------------------------------------------------------
# Reproducible experimental splits
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
REFERENCE_RATIO = 0.80
INTERNAL_TRAIN_RATIO = 0.75  # tuning is performed only inside reference data

# -----------------------------------------------------------------------------
# Stronger baseline GLCM (still a standard GLCM descriptor)
# -----------------------------------------------------------------------------
GLCM_LEVELS = 32
GLCM_BASELINE_DISTANCES = (1,)
GLCM_ANGLES_DEG = (0, 45, 90, 135)
GLCM_PROPERTIES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "asm",
    "entropy",
    "variance",
)

# Baseline LBP remains single-scale, but includes histogram summary statistics.
LBP_BASELINE = ((8, 1),)

# -----------------------------------------------------------------------------
# Multi-scale texture candidates used by the v7 BP-PTD texture specialist
# -----------------------------------------------------------------------------
GLCM_ENHANCED_DISTANCES = (1, 2, 3)
LBP_ENHANCED = ((8, 1), (16, 2))
TEXTURE_CORRELATION_THRESHOLD = 0.90

# Regularized Mahalanobis distance classifier. alpha=1.0 is a diagonal
# covariance (very conservative); lower values preserve more correlations.
MAHALANOBIS_ALPHA_CANDIDATES = (0.25, 0.50, 0.75, 0.90, 1.00)
MAHALANOBIS_RIDGE = 1e-3

# ROI quality validation is ONLY for diagnostics/demo selection.
# Member 3 never changes the ROI mask.
ROI_ABSOLUTE_MIN_RATIO = 0.002
ROI_RELATIVE_MEDIAN_FACTOR = 0.20
ROI_MIN_BBOX_DIM_RATIO = 0.03
ROI_MIN_LARGEST_COMPONENT_FRAC = 0.25

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_KEYWORDS = ("mask", "roi")


def project_root_from_script() -> Path:
    """Assume this package is placed under <project>/src/texture_analysis/."""
    here = Path(__file__).resolve()
    if here.parent.parent.name.lower() == "src":
        return here.parent.parent.parent
    return Path.cwd()

# -----------------------------------------------------------------------------
# v11 proposed texture-only invention: PTD+
# Pairwise Texture Disambiguation Plus
# -----------------------------------------------------------------------------
# Instead of replacing the strong fusion baseline, a binary residual-texture
# tie-breaker is learned only for repeatedly confused class pairs. A rule must
# demonstrate positive net correction across multiple reference-data CV folds.
PAIRWISE_CV_FOLDS = 3
PAIRWISE_TOP_K_CANDIDATES = (4, 8, 12, 20, 32)
PAIRWISE_BASE_CONFIDENCE_THRESHOLDS = (0.05, 0.10, 0.16, 0.24, 0.34, 0.45)
PAIRWISE_TEXTURE_CONFIDENCE_THRESHOLDS = (0.00, 0.05, 0.10, 0.16, 0.24)
PAIRWISE_MIN_CHANGED = 2
PAIRWISE_MIN_NET_GAIN = 1
PAIRWISE_MIN_CORRECTION_PRECISION = 0.67
PAIRWISE_REQUIRED_POSITIVE_FOLDS = 2


# PTD+ preserves all four v8 candidate families exactly. Laws texture energy is
# tested only as an optional enrichment of those families. If an enriched rule
# does not produce a strictly stronger cross-validated net correction without
# adding harm, PTD+ keeps the original v8 rule for that class pair.
PAIRWISE_FEATURE_GROUP_NAMES = (
    "global_multiscale",
    "local_heterogeneity",
    "gabor",
    "all_texture",
    "global_multiscale_plus_laws",
    "local_heterogeneity_plus_laws",
    "gabor_plus_laws",
    "all_texture_plus_laws",
)
PAIRWISE_TEXTURE_ALPHA_CANDIDATES = (0.50, 0.90, 1.00)

# Conservative acceptance criteria for Laws enrichment. These are evaluated
# only on reference-data cross-validation; the final evaluation split remains
# completely untouched.
LAWS_MIN_EXTRA_NET_GAIN = 1
LAWS_REQUIRE_NO_MORE_HARM = True
LAWS_REQUIRE_POSITIVE_FOLDS_NOT_LOWER = True
