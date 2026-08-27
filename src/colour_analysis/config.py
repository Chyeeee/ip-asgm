from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "MedianFinal"
    / "ProcessedImages"
)

MASK_DIR = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "MedianFinal"
    / "ROIMasks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "colour_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Stage 1:
# Number of images selected from each category
# of each fruit.
SAMPLE_PER_FRUIT_CATEGORY = 20

# 70% training / 30% testing
TEST_SIZE = 0.30

# Reproducible sampling and train/test split
RANDOM_STATE = 42

# Stage 2 enhanced feature histogram bins
HIST_BINS = 16


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)