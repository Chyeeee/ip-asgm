import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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

OUTPUT_DIR = "results/ml/guava_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\n" + "=" * 70)
print("GUAVA QUALITY CLASSIFIER")
print("=" * 70)

print(f"\nOriginal dataset : {len(df)} images")


# ============================================================
# SELECT GUAVA ONLY
# ============================================================

df = df[
    df["fruit"].str.lower() == "guava"
].copy()

print(f"Guava dataset    : {len(df)} images")

print("\nClass distribution:")
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
    "blemish_mask_path",

    "colour_space",
    "feature_version",

    "damage_level",

    # Fruit is constant because all rows are Guava
    "fruit",

    # Raw area values
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
    column
    for column in X.columns
    if X[column].nunique() <= 1
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
# CHECK FEATURES
# ============================================================

print("\nFeature configuration:")
print(f"Features used     : {X.shape[1]}")
print(
    f"Damage included   : "
    f"{'damage_percentage' in X.columns}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print("\nDataset split:")
print(f"Training : {len(X_train)}")
print(f"Testing  : {len(X_test)}")


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)


# No categorical features remain.
# Numerical features can pass directly to Random Forest.

pipeline = Pipeline(
    steps=[
        (
            "classifier",
            rf
        )
    ]
)


# ============================================================
# PARAMETER SEARCH
# ============================================================

parameter_distributions = {

    "classifier__n_estimators": [
        100,
        200,
        300,
        400,
        500
    ],

    "classifier__max_depth": [
        None,
        5,
        10,
        15,
        20
    ],

    "classifier__min_samples_split": [
        2,
        4,
        6,
        8
    ],

    "classifier__min_samples_leaf": [
        1,
        2,
        3,
        4
    ],

    "classifier__max_features": [
        "sqrt",
        "log2",
        0.5,
        None
    ],

    "classifier__bootstrap": [
        True,
        False
    ]
}


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(
    estimator=pipeline,

    param_distributions=
        parameter_distributions,

    n_iter=40,

    scoring="f1_weighted",

    cv=cv,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=2,

    return_train_score=True
)


print("\n" + "=" * 70)
print("STARTING GUAVA MODEL TUNING")
print("=" * 70)

print(
    "\n40 configurations x 5 folds "
    "= approximately 200 model fits"
)


# ============================================================
# TRAIN
# ============================================================

search.fit(
    X_train,
    y_train
)


# ============================================================
# BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("BEST CROSS-VALIDATION RESULT")
print("=" * 70)

print(
    f"\nBest CV F1 Score : "
    f"{search.best_score_:.4f}"
)

print("\nBest Parameters:")

for parameter, value in search.best_params_.items():

    clean_name = parameter.replace(
        "classifier__",
        ""
    )

    print(
        f"  {clean_name:<20}: {value}"
    )


# ============================================================
# TEST BEST MODEL
# ============================================================

best_model = search.best_estimator_

predictions = best_model.predict(
    X_test
)


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


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL GUAVA TEST PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "Class_A",
    "Class_B",
    "Defect"
]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


print("Confusion Matrix:")
print(labels)
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)


# ============================================================
# SAVE SEARCH RESULTS
# ============================================================

search_results = pd.DataFrame(
    search.cv_results_
)

search_results = search_results.sort_values(
    "rank_test_score"
)

search_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "random_search_results.csv"
    ),
    index=False
)


# ============================================================
# SAVE PERFORMANCE
# ============================================================

performance = pd.DataFrame(
    [
        {
            "Model":
                "Guava Random Forest",

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1_Score":
                f1,

            "CV_F1_Score":
                search.best_score_
        }
    ]
)

performance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_performance.csv"
    ),
    index=False
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

best_parameters = {

    parameter.replace(
        "classifier__",
        ""
    ): value

    for parameter, value
    in search.best_params_.items()
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "best_parameters.json"
    ),
    "w"
) as file:

    json.dump(
        best_parameters,
        file,
        indent=4
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "final_guava_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_info = {

    "features":
        list(X.columns),

    "target":
        "category",

    "classes":
        labels,

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test),

    "damage_feature":
        "damage_percentage",

    "constant_features_removed":
        constant_columns,

    "test_accuracy":
        accuracy,

    "test_f1":
        f1,

    "cross_validation_f1":
        search.best_score_
}


joblib.dump(
    feature_info,
    os.path.join(
        OUTPUT_DIR,
        "model_info.pkl"
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("GUAVA MODEL TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest CV F1     : "
    f"{search.best_score_:.4f}"
)

print(
    f"Final Test F1  : "
    f"{f1:.4f}"
)

print(
    f"Final Accuracy : "
    f"{accuracy:.4f}"
)

print(
    f"\nFinal model saved to:\n"
    f"{model_path}"
)

print("=" * 70)