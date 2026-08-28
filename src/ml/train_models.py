import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = (
    "results/blemish_detection/final_analysis/"
    "final_features_with_damage.csv"
)

OUTPUT_DIR = "results/ml"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("ML MODEL TRAINING")
print("=" * 60)

print(f"\nOriginal dataset: {len(df)} images")


# ============================================================
# REMOVE GUAVA
# ============================================================

df = df[
    df["fruit"].str.lower() != "guava"
].copy()

print(f"Ripeness dataset: {len(df)} images")

print("\nTarget distribution:")
print(df["category"].value_counts())


# ============================================================
# TARGET
# ============================================================

y = df["category"]


# ============================================================
# REMOVE NON-ML COLUMNS
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

    # Do not use raw areas
    "fruit_pixels",
    "blemish_pixels"
]


X = df.drop(
    columns=exclude_columns,
    errors="ignore"
)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    col
    for col in X.columns
    if X[col].nunique() <= 1
]

print(
    f"\nConstant features removed: "
    f"{len(constant_columns)}"
)

for column in constant_columns:
    print(f"  - {column}")

X = X.drop(
    columns=constant_columns
)


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = [
    "fruit"
]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


print("\nFeature configuration:")
print(
    f"Numeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print("\nDataset split:")
print(
    f"Training : {len(X_train)}"
)

print(
    f"Testing  : {len(X_test)}"
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),

        (
            "fruit",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )

    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

best_model = None
best_model_name = None
best_f1 = -1


for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

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
        f"\nAccuracy  : {accuracy:.4f}"
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


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


    print("Confusion Matrix:")

    labels = sorted(
        y.unique()
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print(labels)
    print(cm)


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        }
    )


    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_model = pipeline

        best_model_name = model_name


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "best_ripeness_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_info = {
    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "excluded_columns":
        exclude_columns,

    "target":
        "category"
}

joblib.dump(
    feature_info,
    os.path.join(
        OUTPUT_DIR,
        "feature_info.pkl"
    )
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model    : {best_model_name}"
)

print(
    f"F1 Score : {best_f1:.4f}"
)

print(
    f"\nSaved model:\n"
    f"{model_path}"
)

print(
    f"\nComparison:\n"
    f"{comparison_path}"
)

print("=" * 60)