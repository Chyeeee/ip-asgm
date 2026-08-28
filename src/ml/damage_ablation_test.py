import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = (
    "results/blemish_detection/final_analysis/"
    "final_features_with_damage.csv"
)

OUTPUT_DIR = "results/ml"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

# Remove Guava because its labels are different
df = df[
    df["fruit"].str.lower() != "guava"
].copy()


# ============================================================
# TARGET
# ============================================================

y = df["category"]


# ============================================================
# REMOVE METADATA
# ============================================================

exclude_columns = [
    "category",
    "image",
    "relative_path",
    "processed_path",
    "mask_path",
    "colour_space",
    "feature_version",
    "damage_level",
    "blemish_mask_path",

    # Raw area measurements
    "fruit_pixels",
    "blemish_pixels"
]


base_X = df.drop(
    columns=exclude_columns,
    errors="ignore"
)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    column
    for column in base_X.columns
    if base_X[column].nunique() <= 1
]

base_X = base_X.drop(
    columns=constant_columns
)


# ============================================================
# EXPERIMENT A
# Colour + Texture WITHOUT Member 4 Damage
# ============================================================

X_without_damage = base_X.drop(
    columns=["damage_percentage"],
    errors="ignore"
)


# ============================================================
# EXPERIMENT B
# Colour + Texture + Member 4 Damage
# ============================================================

X_with_damage = base_X.copy()


# ============================================================
# IMPORTANT:
# USE SAME TRAIN / TEST ROWS FOR BOTH EXPERIMENTS
# ============================================================

train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def run_experiment(
    experiment_name,
    X
):

    print("\n" + "=" * 65)
    print(experiment_name)
    print("=" * 65)

    X_train = X.loc[
        train_indices
    ]

    X_test = X.loc[
        test_indices
    ]

    y_train = y.loc[
        train_indices
    ]

    y_test = y.loc[
        test_indices
    ]


    categorical_features = [
        "fruit"
    ]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]


    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "fruit",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )


    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    print(
        f"Features  : {len(numeric_features)} numeric "
        f"+ fruit"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    return {
        "Experiment": experiment_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }


# ============================================================
# RUN BOTH EXPERIMENTS
# ============================================================

result_without = run_experiment(
    "Colour + Texture WITHOUT Damage",
    X_without_damage
)

result_with = run_experiment(
    "Colour + Texture + Damage",
    X_with_damage
)


# ============================================================
# COMPARE
# ============================================================

results = pd.DataFrame([
    result_without,
    result_with
])


print("\n")
print("=" * 65)
print("MEMBER 4 DAMAGE FEATURE ABLATION")
print("=" * 65)

print(
    results.to_string(
        index=False
    )
)


difference = (
    result_with["F1_Score"]
    - result_without["F1_Score"]
)


print("\nF1 difference:")

print(
    f"{difference:+.4f}"
)


if difference > 0:

    print(
        "\nRESULT: Damage percentage IMPROVED "
        "ripeness classification."
    )

elif difference < 0:

    print(
        "\nRESULT: Damage percentage did NOT improve "
        "ripeness classification."
    )

else:

    print(
        "\nRESULT: Damage percentage produced "
        "no change."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "damage_ablation_comparison.csv"
)

results.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)

print("=" * 65)