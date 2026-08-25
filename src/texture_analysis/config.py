from pathlib import Path

# ============================================================
# PATH SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent

BANANA_DATASET_DIR = PROJECT_ROOT / "Fruits_Data" / "Banana"
OUTPUT_DIR = MODULE_DIR / "outputs"

# ============================================================
# DATASET SETTINGS
# ============================================================

CLASS_NAMES = ["Class_A", "Class_B", "Defect"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# ============================================================
# PREPROCESSING / ROI SETTINGS
# ============================================================

IMAGE_SIZE = (256, 256)

# Banana images use a mostly white/light background.
# These thresholds identify coloured or darker foreground pixels.
MIN_SATURATION = 60
# Remove very small isolated objects/noise.
MIN_COMPONENT_AREA_RATIO = 0.02

# Small padding around detected banana region.
ROI_PADDING = 8

# ============================================================
# GLCM SETTINGS
# ============================================================

GLCM_DISTANCE = 1
GLCM_ANGLES_DEGREES = [0, 45, 90, 135]
GLCM_LEVELS = 32

GLCM_PROPERTIES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

# ============================================================
# LBP SETTINGS
# ============================================================

LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# ============================================================
# CLASSIFICATION SETTINGS
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

SVM_KERNEL = "rbf"
SVM_C = 10.0
