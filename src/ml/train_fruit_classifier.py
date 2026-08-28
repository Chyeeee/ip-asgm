import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

OUTPUT_DIR = "results/ml/fruit_classifier"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\n" + "=" * 70)
print("FRUIT TYPE CLASSIFICATION")
print("=" * 70)

print(f"\nTotal images: {len(df)}")

print("\nFruit distribution:")
print(df["fruit"].value_counts())


# ============================================================
# TARGET
# ============================================================

y = df["fruit"]


# ============================================================
# REMOVE NON-FEATURE COLUMNS
# ============================================================

# IMPORTANT:
# Fruit classifier should use colour + texture information only.
# Damage and ripeness/quality labels are excluded.

exclude_columns = [
    "fruit",
    "category",

    "damage_percentage",
    "damage_level",
    "fruit_pixels",
    "blemish_pixels",

    "image",
    "relative_path",
    "processed_path",
    "mask_path",
    "blemish_mask_path",

    "colour_space",
    "feature_version"
]


X = df.drop(
    columns=exclude_columns,
    errors="ignore"
)


# ============================================================
# KEEP NUMERIC FEATURES ONLY
# ============================================================

X = X.select_dtypes(
    include=["number"]
).copy()


print(f"\nInitial numeric features: {X.shape[1]}")


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    column
    for column in X.columns
    if X[column].nunique() <= 1
]

X = X.drop(
    columns=constant_columns
)


print(
    f"Constant features removed: "
    f"{len(constant_columns)}"
)

for column in constant_columns:
    print(f"  - {column}")


print(
    f"\nFinal colour + texture features: "
    f"{X.shape[1]}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

for forbidden in [
    "damage_percentage",
    "fruit_pixels",
    "blemish_pixels"
]:

    if forbidden in X.columns:
        raise ValueError(
            f"{forbidden} should NOT be used "
            f"for fruit identification."
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
# MODEL DEFINITIONS
# ============================================================

models = {

    "SVM": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "KNN": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ]
    )
}


# ============================================================
# BASELINE COMPARISON
# ============================================================

results = []

trained_models = {}


print("\n" + "=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
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

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        }
    )

    trained_models[model_name] = model


# ============================================================
# COMPARISON
# ============================================================

comparison = pd.DataFrame(
    results
).sort_values(
    "F1_Score",
    ascending=False
)


print("\n")
print("=" * 70)
print("FRUIT CLASSIFIER MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)


rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)


parameter_distributions = {

    "n_estimators": [
        200,
        300,
        400,
        500
    ],

    "max_depth": [
        None,
        10,
        20,
        30
    ],

    "min_samples_split": [
        2,
        4,
        6,
        8
    ],

    "min_samples_leaf": [
        1,
        2,
        3,
        4
    ],

    "max_features": [
        "sqrt",
        "log2",
        0.5,
        None
    ],

    "bootstrap": [
        True,
        False
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=parameter_distributions,
    n_iter=40,
    scoring="f1_weighted",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)


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

    print(
        f"  {parameter:<20}: {value}"
    )


# ============================================================
# FINAL TEST
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


print("\n")
print("=" * 70)
print("FINAL FRUIT CLASSIFIER PERFORMANCE")
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

labels = sorted(
    df["fruit"].unique()
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


print("\nConfusion Matrix:")
print(labels)
print(cm)


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
# SAVE RANDOM SEARCH
# ============================================================

search_results = pd.DataFrame(
    search.cv_results_
).sort_values(
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
                "Random Forest Fruit Classifier",

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

with open(
    os.path.join(
        OUTPUT_DIR,
        "best_parameters.json"
    ),
    "w"
) as file:

    json.dump(
        search.best_params_,
        file,
        indent=4
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "final_fruit_classifier.pkl"
)

joblib.dump(
    best_model,
    MODEL_PATH
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "features":
        list(X.columns),

    "target":
        "fruit",

    "classes":
        labels,

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test),

    "constant_features_removed":
        constant_columns,

    "uses_damage_percentage":
        False,

    "test_accuracy":
        accuracy,

    "test_f1":
        f1,

    "cross_validation_f1":
        search.best_score_
}


joblib.dump(
    model_info,
    os.path.join(
        OUTPUT_DIR,
        "model_info.pkl"
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("FRUIT CLASSIFIER TRAINING COMPLETED")
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
    "\nFruit classes:"
)

for label in labels:
    print(f"  - {label}")

print(
    f"\nFinal model saved to:\n"
    f"{MODEL_PATH}"
)

print("=" * 70)