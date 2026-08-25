from pathlib import Path
import argparse
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ==================== CONFIGURATION ====================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RESULTS_DIR = ROOT_DIR / "results"

ORIGINAL_CLASSES = ["Green", "Semi-ripe", "Ripe", "Overripe"]

# False = Green, Semi-ripe, Ripe, Overripe
# True = Green + Semi-ripe become Unripe
USE_THREE_CLASSES = False

TEST_SIZE = 0.20
RANDOM_STATE = 42
HIST_BINS = 16
DOMINANT_BINS = 8
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==================== LABEL ====================

def map_label(label):
    if not USE_THREE_CLASSES:
        return label
    if label in ["Green", "Semi-ripe"]:
        return "Unripe"
    return label

def get_class_names():
    if USE_THREE_CLASSES:
        return ["Unripe", "Ripe", "Overripe"]
    return ORIGINAL_CLASSES.copy()

# ==================== IMAGE FILES ====================

def get_image_files(folder):
    return sorted([
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])

# ==================== BANANA SEGMENTATION ====================

def create_banana_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Detect white/light background
    lower_white = np.array([0, 0, 170], dtype=np.uint8)
    upper_white = np.array([179, 70, 255], dtype=np.uint8)
    background_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert: banana = white, background = black
    mask = cv2.bitwise_not(background_mask)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep largest object only
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(
            clean_mask, [largest], -1, 255, thickness=cv2.FILLED
        )
        mask = clean_mask

    return mask

# ==================== COLOUR SPACE ====================

def convert_colour_space(image_bgr, colour_space):
    if colour_space == "RGB":
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        names = ["R", "G", "B"]
        ranges = [(0, 256), (0, 256), (0, 256)]

    elif colour_space == "HSV":
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # OpenCV H = 0-179. Convert to degrees 0-360
        image[:, :, 0] *= 2.0

        # Convert S and V to percentage
        image[:, :, 1] = image[:, :, 1] / 255.0 * 100.0
        image[:, :, 2] = image[:, :, 2] / 255.0 * 100.0

        names = ["H", "S", "V"]
        ranges = [(0, 360), (0, 100), (0, 100)]

    elif colour_space == "CIELAB":
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Convert OpenCV LAB to standard CIELAB scale
        image[:, :, 0] = image[:, :, 0] / 255.0 * 100.0
        image[:, :, 1] -= 128.0
        image[:, :, 2] -= 128.0

        names = ["L", "a", "b"]
        ranges = [(0, 100), (-128, 128), (-128, 128)]

    else:
        raise ValueError(f"Unknown colour space: {colour_space}")

    return image, names, ranges

# ==================== DOMINANT COLOUR ====================

def calculate_dominant_colour(pixels, channel_ranges):
    quantised = []

    for channel_index, (minimum, maximum) in enumerate(channel_ranges):
        values = pixels[:, channel_index]
        scaled = (values - minimum) / (maximum - minimum)

        indices = np.floor(
            scaled * DOMINANT_BINS
        ).astype(np.int32)

        indices = np.clip(
            indices, 0, DOMINANT_BINS - 1
        )

        quantised.append(indices)

    code = (
        quantised[0] * DOMINANT_BINS * DOMINANT_BINS
        + quantised[1] * DOMINANT_BINS
        + quantised[2]
    )

    counts = np.bincount(
        code, minlength=DOMINANT_BINS ** 3
    )

    dominant_code = np.argmax(counts)
    dominant_pixels = pixels[code == dominant_code]

    return np.mean(dominant_pixels, axis=0)

# ==================== FEATURE EXTRACTION ====================

def extract_colour_features(image_bgr, mask, colour_space):
    image, channel_names, channel_ranges = convert_colour_space(
        image_bgr, colour_space
    )

    # Only banana pixels are used
    pixels = image[mask > 0]

    if len(pixels) == 0:
        raise ValueError("No banana pixels detected")

    features = {}

    # Mean
    means = np.mean(pixels, axis=0)

    # Standard deviation
    stds = np.std(pixels, axis=0)

    for i, channel in enumerate(channel_names):
        features[f"{colour_space}_{channel}_mean"] = float(means[i])
        features[f"{colour_space}_{channel}_std"] = float(stds[i])

    # Histogram
    for i, channel in enumerate(channel_names):
        histogram, _ = np.histogram(
            pixels[:, i],
            bins=HIST_BINS,
            range=channel_ranges[i]
        )

        histogram = histogram.astype(np.float32)

        if histogram.sum() > 0:
            histogram /= histogram.sum()

        for bin_index, value in enumerate(histogram):
            features[
                f"{colour_space}_{channel}_hist_{bin_index}"
            ] = float(value)

    # Dominant colour
    dominant = calculate_dominant_colour(
        pixels, channel_ranges
    )

    for i, channel in enumerate(channel_names):
        features[
            f"{colour_space}_{channel}_dominant"
        ] = float(dominant[i])

    return features

# ==================== BUILD FEATURE DATASET ====================

def build_feature_datasets():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    RESULTS_DIR.mkdir(exist_ok=True)

    colour_spaces = ["RGB", "HSV", "CIELAB"]
    records = {space: [] for space in colour_spaces}

    total_images = 0

    print("\n========== DATASET ==========")

    for class_name in ORIGINAL_CLASSES:
        folder = DATASET_DIR / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        images = get_image_files(folder)

        print(f"{class_name}: {len(images)} images")
        total_images += len(images)

    print("-----------------------------")
    print(f"Total images: {total_images}")

    processed = 0
    skipped = 0

    print("\nExtracting RGB, HSV and CIELAB features...")

    for original_class in ORIGINAL_CLASSES:
        folder = DATASET_DIR / original_class
        images = get_image_files(folder)
        label = map_label(original_class)

        for image_path in images:
            image = cv2.imread(str(image_path))

            if image is None:
                print("Cannot read:", image_path.name)
                skipped += 1
                continue

            mask = create_banana_mask(image)
            banana_area = cv2.countNonZero(mask)

            if banana_area < 100:
                print("Invalid banana mask:", image_path.name)
                skipped += 1
                continue

            try:
                all_features = {}

                for colour_space in colour_spaces:
                    all_features[colour_space] = extract_colour_features(
                        image, mask, colour_space
                    )

            except Exception as error:
                print(f"Error {image_path.name}: {error}")
                skipped += 1
                continue

            for colour_space in colour_spaces:
                row = {
                    "filename": image_path.name,
                    "original_class": original_class,
                    "label": label
                }

                row.update(all_features[colour_space])
                records[colour_space].append(row)

            processed += 1

            if processed % 50 == 0:
                print(f"Processed: {processed}/{total_images}")

    print(f"\nSuccessfully processed: {processed}")
    print(f"Skipped: {skipped}")

    dataframes = {}

    for colour_space in colour_spaces:
        df = pd.DataFrame(records[colour_space])

        output_file = (
            RESULTS_DIR /
            f"features_{colour_space.lower()}.csv"
        )

        df.to_csv(output_file, index=False)
        dataframes[colour_space] = df

        feature_count = len(df.columns) - 3

        print(
            f"{colour_space}: {feature_count} features saved"
        )

    return dataframes

# ==================== FEATURE COLUMNS ====================

def get_feature_columns(dataframe):
    ignore = {"filename", "original_class", "label"}

    return [
        column for column in dataframe.columns
        if column not in ignore
    ]

# ==================== CLASSIFIERS ====================

def create_model(model_name):
    if model_name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance"
                )
            )
        ])

    if model_name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale"
                )
            )
        ])

    raise ValueError("Unknown classifier")

# ==================== CONFUSION MATRIX ====================

def save_confusion_matrix(
    y_test,
    predictions,
    classes,
    colour_space,
    classifier
):
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(
        f"{colour_space} + {classifier}"
    )

    plt.tight_layout()

    output = RESULTS_DIR / (
        f"confusion_matrix_"
        f"{colour_space.lower()}_"
        f"{classifier.lower()}.png"
    )

    plt.savefig(output, dpi=200)
    plt.close()

# ==================== TRAIN + EVALUATE ====================

def evaluate_models(dataframes):
    classes = get_class_names()

    # Same train/test split for RGB, HSV and CIELAB
    reference = dataframes["RGB"]
    labels = reference["label"].to_numpy()
    indices = np.arange(len(reference))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"\nTraining images: {len(train_indices)}")
    print(f"Testing images: {len(test_indices)}")

    results = []

    for colour_space, dataframe in dataframes.items():
        feature_columns = get_feature_columns(dataframe)

        X = dataframe[
            feature_columns
        ].to_numpy(dtype=np.float32)

        y = dataframe["label"].to_numpy()

        X_train = X[train_indices]
        X_test = X[test_indices]

        y_train = y[train_indices]
        y_test = y[test_indices]

        for classifier in ["KNN", "SVM"]:
            print("\n================================")
            print(f"{colour_space} + {classifier}")

            model = create_model(classifier)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test, predictions
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

            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1-score : {f1:.4f}")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    predictions,
                    labels=classes,
                    zero_division=0
                )
            )

            report = classification_report(
                y_test,
                predictions,
                labels=classes,
                output_dict=True,
                zero_division=0
            )

            pd.DataFrame(report).transpose().to_csv(
                RESULTS_DIR /
                (
                    f"classification_report_"
                    f"{colour_space.lower()}_"
                    f"{classifier.lower()}.csv"
                )
            )

            save_confusion_matrix(
                y_test,
                predictions,
                classes,
                colour_space,
                classifier
            )

            results.append({
                "colour_space": colour_space,
                "classifier": classifier,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["accuracy", "f1_score"],
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_DIR / "classification_results.csv",
        index=False
    )

    print("\n========== FINAL COMPARISON ==========")
    print(results_df.to_string(index=False))

    return results_df

# ==================== ACCURACY GRAPH ====================

def save_comparison_graph(results_df):
    names = (
        results_df["colour_space"]
        + " + "
        + results_df["classifier"]
    )

    accuracy = results_df["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        names,
        accuracy
    )

    ax.set_title(
        "Banana Maturity Classification Performance"
    )

    ax.set_xlabel(
        "Colour Space + Classifier"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(0, 100)

    ax.tick_params(
        axis="x",
        rotation=30
    )

    for bar, value in zip(bars, accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "accuracy_comparison.png",
        dpi=200
    )

    plt.close()

# ==================== SAVE BEST MODEL ====================

def save_best_model(results_df, dataframes):
    best = results_df.iloc[0]

    colour_space = best["colour_space"]
    classifier = best["classifier"]

    print("\n========== BEST COMBINATION ==========")
    print("Colour Space:", colour_space)
    print("Classifier  :", classifier)
    print(f"Accuracy    : {best['accuracy']:.4f}")
    print(f"Precision   : {best['precision']:.4f}")
    print(f"Recall      : {best['recall']:.4f}")
    print(f"F1-score    : {best['f1_score']:.4f}")

    dataframe = dataframes[colour_space]

    feature_columns = get_feature_columns(
        dataframe
    )

    X = dataframe[
        feature_columns
    ].to_numpy(dtype=np.float32)

    y = dataframe["label"].to_numpy()

    model = create_model(classifier)
    model.fit(X, y)

    bundle = {
        "colour_space": colour_space,
        "classifier": classifier,
        "feature_columns": feature_columns,
        "classes": get_class_names(),
        "model": model
    }

    output = (
        RESULTS_DIR /
        "best_banana_maturity_model.joblib"
    )

    joblib.dump(bundle, output)

    print("Best model saved:", output)

# ==================== PREDICT NEW IMAGE ====================

def predict_new_image(image_path):
    model_file = (
        RESULTS_DIR /
        "best_banana_maturity_model.joblib"
    )

    if not model_file.exists():
        print("Model does not exist.")
        print("Run training first:")
        print("py banana_maturity.py")
        return

    bundle = joblib.load(model_file)

    image = cv2.imread(str(image_path))

    if image is None:
        print("Cannot read image:", image_path)
        return

    mask = create_banana_mask(image)

    features = extract_colour_features(
        image,
        mask,
        bundle["colour_space"]
    )

    feature_values = [
        features[column]
        for column in bundle["feature_columns"]
    ]

    X_new = pd.DataFrame(
        [feature_values],
        columns=bundle["feature_columns"]
    )

    prediction = bundle[
        "model"
    ].predict(X_new)[0]

    print("\n========== PREDICTION ==========")
    print("Image       :", image_path)
    print("Colour Space:", bundle["colour_space"])
    print("Classifier  :", bundle["classifier"])
    print("Prediction  :", prediction)

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    cv2.imshow(
        "Original Image",
        image
    )

    cv2.imshow(
        "Banana Mask",
        mask
    )

    cv2.imshow(
        f"Prediction: {prediction}",
        segmented
    )

    print("\nPress any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==================== MAIN ====================

def train_and_evaluate():
    print("======================================")
    print("BANANA MATURITY CLASSIFICATION")
    print("======================================")
    print("Dataset:", DATASET_DIR)
    print("Classes:", get_class_names())

    print("\nSTEP 1: Extract colour features")
    dataframes = build_feature_datasets()

    print("\nSTEP 2: Train KNN and SVM")
    results_df = evaluate_models(dataframes)

    print("\nSTEP 3: Generate comparison graph")
    save_comparison_graph(results_df)

    print("\nSTEP 4: Select best combination")
    save_best_model(
        results_df,
        dataframes
    )

    print("\n======================================")
    print("FINISHED")
    print("======================================")
    print("Check the results folder.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Predict maturity of a new banana image"
    )

    args = parser.parse_args()

    if args.predict:
        predict_new_image(
            Path(args.predict)
        )
    else:
        train_and_evaluate()

if __name__ == "__main__":
    main()