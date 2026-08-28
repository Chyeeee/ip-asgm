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

OUTPUT_DIR = "results/ml/random_forest_tuning"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\n" + "=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

print(
    f"\nOriginal dataset : {len(df)} images"
)


# ============================================================
# REMOVE GUAVA
# ============================================================

# Guava uses:
# Class_A / Class_B / Defect
#
# Other fruits use:
# Unripe / Ripe / Overripe / Rotten
#
# Therefore Guava is excluded from this ripeness classifier.

df = df[
    df["fruit"].str.lower() != "guava"
].copy()

print(
    f"Ripeness dataset : {len(df)} images"
)

print("\nTarget distribution:")

print(
    df["category"].value_counts()
)


# ============================================================
# TARGET
# ============================================================

y = df["category"]


# ============================================================
# REMOVE METADATA / UNSUITABLE FEATURES
# ============================================================

exclude_columns = [

    # Target
    "category",

    # Identification / paths
    "image",
    "relative_path",
    "processed_path",
    "mask_path",
    "blemish_mask_path",

    # Metadata
    "colour_space",
    "feature_version",

    # Derived categorical damage level
    "damage_level",

    # Old raw Otsu damage is retained only for comparison.
    # Exclude it from ML so training uses the final AWDP_A damage feature.
    "raw_damage_percentage",

    # Raw areas
    # Normalised damage_percentage is preferred
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
    print(
        f"  - {column}"
    )


X = X.drop(
    columns=constant_columns
)


# ============================================================
# FEATURE INFORMATION
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

print(
    f"Damage included      : "
    f"{'damage_percentage' in numeric_features}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# Test set is separated BEFORE tuning.
#
# RandomizedSearchCV will only work with X_train / y_train.

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

print(
    f"Training : {len(X_train)}"
)

print(
    f"Testing  : {len(X_test)}"
)


# ============================================================
# PREPROCESSING
# ============================================================

# Random Forest does not require StandardScaler.
#
# Fruit type is categorical, so OneHotEncoder is used.
# All numerical features are passed through unchanged.

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


# ============================================================
# BASE RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
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
            rf
        )
    ]
)


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

parameter_distributions = {

    # Number of trees
    "classifier__n_estimators": [
        200,
        300,
        400,
        500,
        700,
        900
    ],

    # Maximum tree depth
    "classifier__max_depth": [
        None,
        10,
        15,
        20,
        25,
        30,
        40
    ],

    # Minimum samples required to split
    "classifier__min_samples_split": [
        2,
        4,
        6,
        8,
        10
    ],

    # Minimum samples in leaf
    "classifier__min_samples_leaf": [
        1,
        2,
        3,
        4
    ],

    # Number of features considered per split
    "classifier__max_features": [
        "sqrt",
        "log2",
        0.5,
        None
    ],

    # Bootstrap sampling
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

# 50 combinations × 5 folds
# = 250 model fits.
#
# This may take several minutes depending on the computer.

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_distributions,

    n_iter=50,

    scoring="f1_weighted",

    cv=cv,

    verbose=2,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    return_train_score=True
)


print("\n" + "=" * 70)
print("STARTING RANDOMIZED SEARCH")
print("=" * 70)

print(
    "\nSearching 50 parameter combinations "
    "using 5-fold cross-validation."
)

print(
    "Total model fits: approximately 250"
)

print(
    "\nPlease wait until tuning is completed..."
)


# ============================================================
# TRAIN / TUNE
# ============================================================

search.fit(
    X_train,
    y_train
)


# ============================================================
# BEST CROSS-VALIDATION RESULT
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
# BEST MODEL
# ============================================================

best_model = search.best_estimator_


# ============================================================
# FINAL TEST SET EVALUATION
# ============================================================

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


print("\n")
print("=" * 70)
print("FINAL TEST SET PERFORMANCE")
print("=" * 70)


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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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
    "Unripe",
    "Ripe",
    "Overripe",
    "Rotten"
]


cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


print("Confusion Matrix:")

print(
    labels
)

print(
    cm
)


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# SAVE ALL SEARCH RESULTS
# ============================================================

cv_results = pd.DataFrame(
    search.cv_results_
)

cv_results = cv_results.sort_values(
    by="rank_test_score"
)


cv_results_path = os.path.join(
    OUTPUT_DIR,
    "random_search_results.csv"
)

cv_results.to_csv(
    cv_results_path,
    index=False
)


# ============================================================
# SAVE FINAL PERFORMANCE
# ============================================================

performance = pd.DataFrame(
    [
        {
            "Model":
                "Tuned Random Forest",

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


performance_path = os.path.join(
    OUTPUT_DIR,
    "final_performance.csv"
)

performance.to_csv(
    performance_path,
    index=False
)


# ============================================================
# SAVE BEST PARAMETERS
# ============================================================

best_parameters = {

    parameter.replace(
        "classifier__",
        ""
    ): value

    for parameter, value
    in search.best_params_.items()
}


parameters_path = os.path.join(
    OUTPUT_DIR,
    "best_parameters.json"
)


with open(
    parameters_path,
    "w"
) as file:

    json.dump(
        best_parameters,
        file,
        indent=4
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "final_ripeness_model.pkl"
)


joblib.dump(
    best_model,
    model_path
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "target":
        "category",

    "classes":
        labels,

    "excluded_fruit":
        "Guava",

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test),

    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "constant_features_removed":
        constant_columns,

    "damage_feature":
        "damage_percentage",

    "test_accuracy":
        accuracy,

    "test_precision":
        precision,

    "test_recall":
        recall,

    "test_f1":
        f1,

    "cross_validation_f1":
        search.best_score_,

    "best_parameters":
        best_parameters
}


info_path = os.path.join(
    OUTPUT_DIR,
    "model_info.pkl"
)


joblib.dump(
    model_info,
    info_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TUNING COMPLETED")
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
    "\nFiles saved:"
)

print(
    f"  Model             : "
    f"{model_path}"
)

print(
    f"  Parameters        : "
    f"{parameters_path}"
)

print(
    f"  Performance       : "
    f"{performance_path}"
)

print(
    f"  Search results    : "
    f"{cv_results_path}"
)

print(
    f"  Confusion matrix  : "
    f"{cm_path}"
)

print(
    f"  Model information : "
    f"{info_path}"
)


print("\n" + "=" * 70)