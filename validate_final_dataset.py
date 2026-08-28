import pandas as pd
from pathlib import Path

CSV_PATH = Path(
    "results/blemish_detection/final_analysis/final_features_with_damage.csv"
)

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("FINAL DATASET VALIDATION")
print("=" * 60)

# 1. Dataset size
print("\n1. DATASET SIZE")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# 2. Fruit distribution
print("\n2. FRUIT DISTRIBUTION")
print(df["fruit"].value_counts().sort_index())

# 3. Category distribution
print("\n3. CATEGORY DISTRIBUTION")
print(df["category"].value_counts())

# 4. Fruit x category distribution
print("\n4. FRUIT x CATEGORY DISTRIBUTION")
distribution = pd.crosstab(df["fruit"], df["category"])
print(distribution)

# 5. Missing values
print("\n5. MISSING VALUES")
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing.sort_values(ascending=False))

# 6. Duplicate images
print("\n6. DUPLICATE IMAGE CHECK")
duplicates = df[df.duplicated(subset=["fruit", "category", "image"], keep=False)]

if duplicates.empty:
    print("No duplicate images found.")
else:
    print(f"Duplicate rows found: {len(duplicates)}")
    print(duplicates[["fruit", "category", "image"]])

# 7. Damage percentage validation
print("\n7. DAMAGE PERCENTAGE CHECK")

invalid_damage = df[
    (df["damage_percentage"] < 0)
    | (df["damage_percentage"] > 100)
]

if invalid_damage.empty:
    print("All damage percentages are between 0 and 100.")
else:
    print(f"Invalid damage values found: {len(invalid_damage)}")

# 8. Damage statistics
print("\n8. DAMAGE STATISTICS")
print(df["damage_percentage"].describe())

# 9. Check numeric feature columns
excluded_columns = [
    "fruit",
    "category",
    "image",
    "relative_path",
    "processed_path",
    "mask_path",
    "colour_space",
    "feature_version",
    "damage_level",
    "blemish_mask_path"
]

feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

numeric_features = df[feature_columns].select_dtypes(include="number")

print("\n9. NUMERIC FEATURES")
print(f"Number of candidate numeric features: {numeric_features.shape[1]}")

# 10. Infinite values
print("\n10. INFINITE VALUE CHECK")

import numpy as np

inf_count = np.isinf(numeric_features).sum().sum()

if inf_count == 0:
    print("No infinite values found.")
else:
    print(f"Infinite values found: {inf_count}")

# 11. Constant features
print("\n11. CONSTANT FEATURE CHECK")

constant_features = [
    column
    for column in numeric_features.columns
    if numeric_features[column].nunique() <= 1
]

if not constant_features:
    print("No constant numeric features found.")
else:
    print("Constant features:")
    for column in constant_features:
        print(f"  - {column}")

# 12. Guava warning
print("\n12. GUAVA CATEGORY CHECK")

guava = df[df["fruit"].str.lower() == "guava"]

if not guava.empty:
    print("Guava categories:")
    print(guava["category"].value_counts())
    print(
        "\nNOTE: Guava uses different labels from the standard "
        "Unripe/Ripe/Overripe/Rotten classes."
    )

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)