import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "results/texture_analysis/colour_texture_features.csv"

PROCESSED_ROOT = (
    "results/preprocessing/MedianFinal/ProcessedImages"
)

MASK_ROOT = (
    "results/preprocessing/MedianFinal/ROIMasks"
)


# ============================================================
# LOAD MEMBER 3 CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\n==========================================")
print("MEMBER 4 - INPUT CHECK")
print("==========================================")

print(f"Total rows: {len(df)}")
print(f"Total fruits: {df['fruit'].nunique()}")

print("\nFruits:")
print(sorted(df["fruit"].unique()))

print("\nCategories:")
print(sorted(df["category"].unique()))


# ============================================================
# CHECK IMAGE + MASK PATHS
# ============================================================

valid = 0
missing_image = 0
missing_mask = 0

for _, row in df.iterrows():

    relative_path = row["relative_path"]

    # Example:
    # Apple/Overripe/Apple_Overripe_001.jpg

    image_path = os.path.join(
        PROCESSED_ROOT,
        relative_path
    )

    relative_folder = os.path.dirname(relative_path)

    image_name = os.path.splitext(
        os.path.basename(relative_path)
    )[0]

    mask_path = os.path.join(
        MASK_ROOT,
        relative_folder,
        image_name + "_mask.png"
    )

    image_exists = os.path.exists(image_path)
    mask_exists = os.path.exists(mask_path)

    if not image_exists:
        missing_image += 1

    if not mask_exists:
        missing_mask += 1

    if image_exists and mask_exists:
        valid += 1


# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print("PATH CHECK")
print("==========================================")

print(f"Valid image + mask pairs : {valid}")
print(f"Missing processed images : {missing_image}")
print(f"Missing ROI masks        : {missing_mask}")

print("\n==========================================")

if valid == len(df):
    print("SUCCESS: All Member 3 rows are ready for Member 4.")
else:
    print("WARNING: Some files could not be located.")

print("==========================================")