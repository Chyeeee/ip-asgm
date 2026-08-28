import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = PROJECT_ROOT / "src"

# Member 2's colour_analysis files use local imports such as:
# from config import HIST_BINS
# Therefore add that folder to sys.path.
COLOUR_DIR = SRC_ROOT / "colour_analysis"

if str(COLOUR_DIR) not in sys.path:
    sys.path.insert(0, str(COLOUR_DIR))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# IMPORT EXISTING PROJECT FUNCTIONS
# ============================================================

# Member 1 - preprocessing
from preprocessing.median_preprocessing_all import (
    median_filter,
    create_roi_mask,
    IMAGE_SIZE,
)

# Member 2 - colour features
from colour_features import (
    extract_enhanced_features,
    get_enhanced_feature_names,
)

# Member 3 - texture
from texture_analysis.glcm_features import (
    extract_glcm_features,
)

from texture_analysis.lbp_features import (
    extract_lbp_features,
)

from texture_analysis.config import (
    GLCM_BASELINE_DISTANCES,
    GLCM_ANGLES_DEG,
    LBP_BASELINE,
)

# Member 4 - blemish/damage
#
# Do NOT import final_damage_quantification.py directly because
# that script executes its full-dataset processing at import time.
# The selected Member 4 functions are reproduced below exactly
# for safe single-image inference.


# ============================================================
# MODEL PATHS
# ============================================================

FRUIT_MODEL_PATH = (
    PROJECT_ROOT
    / "results/ml/fruit_classifier/final_fruit_classifier.pkl"
)

FRUIT_INFO_PATH = (
    PROJECT_ROOT
    / "results/ml/fruit_classifier/model_info.pkl"
)

RIPENESS_MODEL_PATH = (
    PROJECT_ROOT
    / "results/ml/random_forest_tuning/final_ripeness_model.pkl"
)

RIPENESS_INFO_PATH = (
    PROJECT_ROOT
    / "results/ml/random_forest_tuning/model_info.pkl"
)

GUAVA_MODEL_PATH = (
    PROJECT_ROOT
    / "results/ml/guava_model/final_guava_model.pkl"
)

GUAVA_INFO_PATH = (
    PROJECT_ROOT
    / "results/ml/guava_model/model_info.pkl"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results/system/single_image"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MEMBER 4 SELECTED METHOD
# OTSU + MORPHOLOGY 7x7
# ============================================================

def otsu_segmentation(image, roi_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    fruit_pixels = gray[
        roi_mask > 0
    ]

    if len(fruit_pixels) == 0:
        return np.zeros_like(gray)

    threshold, _ = cv2.threshold(
        fruit_pixels,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    blemish_mask = np.zeros_like(
        gray
    )

    blemish_mask[
        (gray < threshold)
        & (roi_mask > 0)
    ] = 255

    return blemish_mask


def apply_morphology(mask, roi_mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    refined = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        kernel
    )

    refined[
        roi_mask == 0
    ] = 0

    return refined

def colour_abnormality_segmentation(
    image,
    roi_mask
):
    """
    Detect colour-abnormal regions inside the fruit ROI.

    Uses LAB chromatic channels (a*, b*) rather than brightness alone.
    This helps distinguish genuine colour changes from simple shadows.

    Returns:
        colour_mask  : binary colour-abnormality mask
        colour_score : normalized chromatic abnormality map [0, 1]
    """

    roi = (roi_mask > 0)

    if np.count_nonzero(roi) == 0:
        return (
            np.zeros(
                roi_mask.shape,
                dtype=np.uint8
            ),
            np.zeros(
                roi_mask.shape,
                dtype=np.float32
            )
        )

    # --------------------------------------------------------
    # Convert to LAB
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    # IMPORTANT:
    # Use mainly a* and b* for segmentation.
    # L* is brightness and is strongly affected by shadows.

    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    roi_a = a_channel[roi]
    roi_b = b_channel[roi]

    # --------------------------------------------------------
    # Robust reference colour of the fruit
    # --------------------------------------------------------

    median_a = float(
        np.median(roi_a)
    )

    median_b = float(
        np.median(roi_b)
    )

    # Chromatic distance from typical fruit colour
    chroma_distance = np.sqrt(
        (a_channel - median_a) ** 2
        +
        (b_channel - median_b) ** 2
    )

    roi_distance = chroma_distance[roi]

    # --------------------------------------------------------
    # Robust adaptive threshold
    # --------------------------------------------------------

    median_distance = float(
        np.median(roi_distance)
    )

    mad = float(
        np.median(
            np.abs(
                roi_distance
                - median_distance
            )
        )
    )

    # Robust threshold.
    # Prevent threshold from becoming unrealistically small.
    colour_threshold = max(
        median_distance + 2.5 * mad,
        float(
            np.percentile(
                roi_distance,
                75
            )
        )
    )

    # --------------------------------------------------------
    # Binary colour abnormality mask
    # --------------------------------------------------------

    colour_mask = np.zeros(
        roi_mask.shape,
        dtype=np.uint8
    )

    colour_mask[
        roi
        &
        (chroma_distance > colour_threshold)
    ] = 255

    # --------------------------------------------------------
    # Normalized colour score for fusion
    # --------------------------------------------------------

    scale = float(
        np.percentile(
            roi_distance,
            95
        )
    )

    if scale <= 0:
        scale = 1.0

    colour_score = (
        chroma_distance
        / scale
    )

    colour_score = np.clip(
        colour_score,
        0.0,
        1.0
    ).astype(np.float32)

    colour_score[
        ~roi
    ] = 0.0

    # --------------------------------------------------------
    # Remove tiny colour noise
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    colour_mask = cv2.morphologyEx(
        colour_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    colour_mask[
        ~roi
    ] = 0

    return (
        colour_mask,
        colour_score
    )

def refine_blemish_candidates(
    image,
    roi_mask,
    otsu_candidate_mask,
    colour_mask,
    colour_score
):
    """
    Colour-Guided Otsu blemish segmentation.

    Combines:
        1. Otsu intensity abnormality
        2. LAB chromatic abnormality
        3. Texture abnormality
        4. ROI boundary suppression
        5. Connected-component filtering

    The purpose is to avoid treating every dark/shadow region as damage,
    while still allowing strongly colour-abnormal defects to be detected.
    """

    roi = (
        roi_mask > 0
    ).astype(np.uint8) * 255

    otsu_candidate = (
        otsu_candidate_mask > 0
    )

    colour_candidate = (
        colour_mask > 0
    )

    fruit_area = int(
        np.count_nonzero(
            roi
        )
    )

    if fruit_area == 0:
        return np.zeros_like(
            roi_mask
        )

    # ========================================================
    # 1. REMOVE ROI BOUNDARY ARTEFACTS
    # ========================================================

    distance = cv2.distanceTransform(
        roi,
        cv2.DIST_L2,
        5
    )

    boundary_margin = max(
        3.0,
        0.008 * np.sqrt(
            float(fruit_area)
        )
    )

    interior = (
        distance
        > boundary_margin
    )

    # ========================================================
    # 2. INTENSITY ABNORMALITY
    # ========================================================

    intensity_score = (
        awdp_intensity_abnormality(
            image,
            roi
        )
    )

    # ========================================================
    # 3. TEXTURE ABNORMALITY
    # ========================================================

    texture_score = (
        awdp_texture_abnormality(
            image,
            roi
        )
    )

    # ========================================================
    # 4. COLOUR-GUIDED OTSU FUSION
    # ========================================================

    # Strong colour abnormality can indicate damage even when
    # the region is not extremely dark.
    strong_colour = (
        colour_score >= 0.60
    )

    # Otsu regions need additional evidence.
    #
    # A dark candidate is retained when it also has:
    #   - some colour abnormality, OR
    #   - meaningful texture abnormality.
    #
    # This suppresses smooth shadows.

    validated_otsu = (
        otsu_candidate
        &
        (
            (colour_score >= 0.22)
            |
            (texture_score >= 0.30)
        )
    )

    # Colour segmentation may introduce a new candidate only
    # when colour deviation is strong enough.
    validated_colour = (
        colour_candidate
        &
        strong_colour
        &
        (
            (texture_score >= 0.18)
            |
            (intensity_score >= 0.08)
            |
            (colour_score >= 0.78)
        )
    )

    # Final fusion
    combined = (
        validated_otsu
        |
        validated_colour
    )

    # Keep only fruit interior
    combined = (
        combined
        &
        interior
        &
        (roi > 0)
    )

    refined = (
        combined.astype(
            np.uint8
        ) * 255
    )

    # ========================================================
    # 5. SMALL MORPHOLOGICAL CLEANUP
    # ========================================================

    # Use a smaller kernel here because the main method has
    # already used Morphology 7x7.
    cleanup_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        cleanup_kernel
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cleanup_kernel
    )

    # ========================================================
    # 6. REMOVE TINY COMPONENTS
    # ========================================================

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            refined,
            connectivity=8
        )
    )

    clean = np.zeros_like(
        refined
    )

    minimum_area = max(
        20,
        int(
            fruit_area
            * 0.00025
        )
    )

    for label_index in range(
        1,
        number_labels
    ):

        area = int(
            stats[
                label_index,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= minimum_area:

            clean[
                labels == label_index
            ] = 255

    clean[
        roi == 0
    ] = 0

    return clean

def calculate_damage(
    blemish_mask,
    roi_mask
):

    fruit_pixels = np.count_nonzero(
        roi_mask > 0
    )

    blemish_pixels = np.count_nonzero(
        blemish_mask > 0
    )

    if fruit_pixels == 0:

        return (
            0,
            0,
            0.0
        )

    damage_percentage = (
        blemish_pixels
        / fruit_pixels
    ) * 100.0

    return (
        fruit_pixels,
        blemish_pixels,
        damage_percentage
    )


def get_damage_level(
    damage_percentage
):

    if damage_percentage < 10:
        return "Low"

    elif damage_percentage < 30:
        return "Moderate"

    elif damage_percentage < 50:
        return "High"

    else:
        return "Severe"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    required = [
        FRUIT_MODEL_PATH,
        FRUIT_INFO_PATH,
        RIPENESS_MODEL_PATH,
        RIPENESS_INFO_PATH,
        GUAVA_MODEL_PATH,
        GUAVA_INFO_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        print("\nERROR: Required model files missing:\n")

        for path in missing:
            print(f"  - {path}")

        raise FileNotFoundError(
            "Some trained model files are missing."
        )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    check_required_files()

    fruit_model = joblib.load(
        FRUIT_MODEL_PATH
    )

    fruit_info = joblib.load(
        FRUIT_INFO_PATH
    )

    ripeness_model = joblib.load(
        RIPENESS_MODEL_PATH
    )

    ripeness_info = joblib.load(
        RIPENESS_INFO_PATH
    )

    guava_model = joblib.load(
        GUAVA_MODEL_PATH
    )

    guava_info = joblib.load(
        GUAVA_INFO_PATH
    )

    # The enhanced fruit classifier must contain the seven
    # shape descriptors used during training.
    fruit_expected = fruit_info.get(
        "features",
        []
    )

    required_shape_features = {
        "shape_area",
        "shape_perimeter",
        "shape_aspect_ratio",
        "shape_circularity",
        "shape_extent",
        "shape_solidity",
        "shape_equivalent_diameter",
    }

    missing_shape_model_features = (
        required_shape_features
        - set(fruit_expected)
    )

    if missing_shape_model_features:
        raise RuntimeError(
            "The loaded fruit classifier/model_info does not "
            "contain the new shape features. Retrain the fruit "
            "classifier with train_fruit_classifier_with_shape.py "
            "before running prediction."
        )

    return {
        "fruit_model": fruit_model,
        "fruit_info": fruit_info,

        "ripeness_model": ripeness_model,
        "ripeness_info": ripeness_info,

        "guava_model": guava_model,
        "guava_info": guava_info,
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    if image is None:
        raise ValueError(
            "Input image is empty."
        )

    # Training preprocessing used 600x600
    resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    processed = median_filter(
        resized
    )

    roi_mask = create_roi_mask(
        processed
    )

    fruit_pixels = np.count_nonzero(
        roi_mask > 0
    )

    if fruit_pixels == 0:

        raise RuntimeError(
            "Fruit ROI could not be detected."
        )

    return (
        resized,
        processed,
        roi_mask
    )


# ============================================================
# MEMBER 2 COLOUR FEATURES
# ============================================================

def extract_colour_features(
    processed,
    roi_mask
):

    # Final dataset uses Enhanced Lab features.
    colour_space = "Lab"

    values = extract_enhanced_features(
        processed,
        roi_mask,
        colour_space
    )

    if values is None:

        raise RuntimeError(
            "Colour feature extraction failed."
        )

    names = get_enhanced_feature_names(
        colour_space
    )

    if len(names) != len(values):

        raise RuntimeError(
            "Colour feature name/value mismatch."
        )

    return {
        name: float(value)
        for name, value
        in zip(names, values)
    }


# ============================================================
# MEMBER 3 TEXTURE FEATURES
# ============================================================

def extract_texture_features(
    processed,
    roi_mask
):

    glcm = extract_glcm_features(
        processed,
        roi_mask,
        distances=GLCM_BASELINE_DISTANCES,
        angles=GLCM_ANGLES_DEG,
        prefix="glcm",
        keep_per_distance=False,
        include_direction_std=True,
    )

    lbp = extract_lbp_features(
        processed,
        roi_mask,
        settings=LBP_BASELINE,
        prefix="lbp"
    )

    features = {}

    features.update(
        glcm
    )

    features.update(
        lbp
    )

    return features


# ============================================================
# SHAPE FEATURES FOR FRUIT IDENTIFICATION
# Must match train_fruit_classifier_with_shape.py
# ============================================================

def extract_shape_features(roi_mask):

    binary = (
        roi_mask > 0
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    defaults = {
        "shape_area": 0.0,
        "shape_perimeter": 0.0,
        "shape_aspect_ratio": 0.0,
        "shape_circularity": 0.0,
        "shape_extent": 0.0,
        "shape_solidity": 0.0,
        "shape_equivalent_diameter": 0.0,
    }

    if not contours:
        return defaults

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = float(
        cv2.contourArea(contour)
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True
        )
    )

    _, _, width, height = (
        cv2.boundingRect(contour)
    )

    aspect_ratio = (
        float(width) / float(height)
        if height > 0
        else 0.0
    )

    circularity = (
        (4.0 * np.pi * area)
        / (perimeter ** 2)
        if perimeter > 0
        else 0.0
    )

    bounding_area = float(
        width * height
    )

    extent = (
        area / bounding_area
        if bounding_area > 0
        else 0.0
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = float(
        cv2.contourArea(hull)
    )

    solidity = (
        area / hull_area
        if hull_area > 0
        else 0.0
    )

    equivalent_diameter = (
        float(
            np.sqrt(
                (4.0 * area) / np.pi
            )
        )
        if area > 0
        else 0.0
    )

    return {
        "shape_area":
            area,

        "shape_perimeter":
            perimeter,

        "shape_aspect_ratio":
            aspect_ratio,

        "shape_circularity":
            float(circularity),

        "shape_extent":
            float(extent),

        "shape_solidity":
            float(solidity),

        "shape_equivalent_diameter":
            equivalent_diameter,
    }


# ============================================================
# MEMBER 4 DAMAGE
# ============================================================

def analyse_damage(
    processed,
    roi_mask
):

    # ========================================================
    # STEP 1: OTSU INTENSITY SEGMENTATION
    # ========================================================

    otsu_mask = otsu_segmentation(
        processed,
        roi_mask
    )

    # ========================================================
    # STEP 2: MORPHOLOGY 7x7
    # ========================================================

    morphology_mask = apply_morphology(
        otsu_mask,
        roi_mask
    )

    # ========================================================
    # STEP 3: LAB COLOUR ABNORMALITY SEGMENTATION
    # ========================================================

    (
        colour_mask,
        colour_score
    ) = colour_abnormality_segmentation(
        processed,
        roi_mask
    )

    # ========================================================
    # STEP 4: COLOUR-GUIDED OTSU FUSION
    # ========================================================

    blemish_mask = refine_blemish_candidates(
        processed,
        roi_mask,
        morphology_mask,
        colour_mask,
        colour_score
    )

    # ========================================================
    # STEP 5: RAW SEGMENTED DAMAGE
    # ========================================================

    (
        fruit_pixels,
        blemish_pixels,
        raw_damage_percentage
    ) = calculate_damage(
        blemish_mask,
        roi_mask
    )

    # ========================================================
    # STEP 6: AWDP DAMAGE QUANTIFICATION
    # ========================================================

    damage_percentage, awdp_confidence_map = (
        calculate_awdp_damage(
            processed,
            roi_mask,
            blemish_mask
        )
    )

    damage_level = get_damage_level(
        damage_percentage
    )

    return {

        "otsu_mask":
            otsu_mask,

        "morphology_mask":
            morphology_mask,

        "colour_mask":
            colour_mask,

        "colour_score":
            colour_score,

        "blemish_mask":
            blemish_mask,

        "awdp_confidence_map":
            awdp_confidence_map,

        "fruit_pixels":
            fruit_pixels,

        "blemish_pixels":
            blemish_pixels,

        "raw_damage_percentage":
            float(
                raw_damage_percentage
            ),

        "damage_percentage":
            float(
                damage_percentage
            ),

        "damage_level":
            damage_level,
    }


# ============================================================
# BUILD BASE FEATURE DICTIONARY
# ============================================================

def extract_base_features(
    processed,
    roi_mask
):

    features = {}

    colour = extract_colour_features(
        processed,
        roi_mask
    )

    texture = extract_texture_features(
        processed,
        roi_mask
    )

    shape = extract_shape_features(
        roi_mask
    )

    features.update(
        colour
    )

    features.update(
        texture
    )

    features.update(
        shape
    )

    return features


# ============================================================
# BUILD DATAFRAME USING EXACT TRAINING FEATURE ORDER
# ============================================================

def build_model_input(
    feature_dictionary,
    expected_features
):

    missing = [
        feature
        for feature in expected_features
        if feature not in feature_dictionary
    ]

    if missing:

        print("\nMissing model features:")

        for feature in missing:
            print(
                f"  - {feature}"
            )

        raise ValueError(
            "Feature extraction does not match "
            "the trained model."
        )

    row = {
        feature:
            feature_dictionary[feature]

        for feature
        in expected_features
    }

    return pd.DataFrame(
        [row],
        columns=expected_features
    )


# ============================================================
# CONFIDENCE
# ============================================================

def predict_with_confidence(
    model,
    X
):

    prediction = model.predict(
        X
    )[0]

    confidence = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X
        )[0]

        confidence = float(
            np.max(probabilities)
        ) * 100.0

    return (
        prediction,
        confidence
    )


# ============================================================
# FRUIT IDENTIFICATION
# ============================================================

def identify_fruit(
    features,
    models
):

    expected = models[
        "fruit_info"
    ]["features"]

    X = build_model_input(
        features,
        expected
    )

    return predict_with_confidence(
        models["fruit_model"],
        X
    )


# ============================================================
# QUALITY / RIPENESS CLASSIFICATION
# ============================================================

def classify_condition(
    fruit,
    base_features,
    damage_percentage,
    models
):

    features = dict(
        base_features
    )

    features[
        "damage_percentage"
    ] = float(
        damage_percentage
    )

    # --------------------------------------------------------
    # GUAVA
    # --------------------------------------------------------

    if fruit.lower() == "guava":

        expected = models[
            "guava_info"
        ]["features"]

        X = build_model_input(
            features,
            expected
        )

        prediction, confidence = (
            predict_with_confidence(
                models["guava_model"],
                X
            )
        )

        return (
            prediction,
            confidence,
            "Quality Category"
        )

    # --------------------------------------------------------
    # OTHER 8 FRUITS
    # --------------------------------------------------------

    # Ripeness model was trained using fruit as a
    # categorical input as well.
    features["fruit"] = fruit

    ripeness_info = models["ripeness_info"]

    numeric_features = ripeness_info["numeric_features"]
    categorical_features = ripeness_info["categorical_features"]

    expected = numeric_features + categorical_features

    X = build_model_input(
        features,
        expected
    )


    prediction, confidence = (
        predict_with_confidence(
            models["ripeness_model"],
            X
        )
    )

    return (
        prediction,
        confidence,
        "Ripeness"
    )


# ============================================================
# RESULT VISUALISATION
# ============================================================

def create_result_visualisation(
    processed,
    blemish_mask,
    fruit,
    fruit_confidence,
    condition,
    condition_confidence,
    damage_percentage
):

    overlay = processed.copy()

    overlay[
        blemish_mask > 0
    ] = (
        0,
        0,
        255
    )

    result = cv2.addWeighted(
        processed,
        0.65,
        overlay,
        0.35,
        0
    )

    fruit_text = (
        f"Fruit: {fruit} "
        f"({fruit_confidence:.1f}%)"
    )

    condition_text = (
        f"Class: {condition} "
        f"({condition_confidence:.1f}%)"
    )

    damage_text = (
        f"Damage: "
        f"{damage_percentage:.2f}%"
    )

    cv2.putText(
        result,
        fruit_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        condition_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        damage_text,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return result


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    resized,
    processed,
    roi_mask,
    damage,
    visualisation
):

    cv2.imwrite(
        str(
            OUTPUT_ROOT
            / "resized.jpg"
        ),
        resized
    )

    cv2.imwrite(
        str(
            OUTPUT_ROOT
            / "processed.jpg"
        ),
        processed
    )

    cv2.imwrite(
        str(
            OUTPUT_ROOT
            / "roi_mask.png"
        ),
        roi_mask
    )

    cv2.imwrite(
        str(
            OUTPUT_ROOT
            / "otsu_mask.png"
        ),
        damage["otsu_mask"]
    )

    if "morphology_mask" in damage:

        cv2.imwrite(
            str(
                OUTPUT_ROOT
                / "morphology_mask.png"
            ),
            damage[
                "morphology_mask"
            ]
        )

    if "colour_mask" in damage:

        cv2.imwrite(
            str(
                OUTPUT_ROOT
                / "colour_mask.png"
            ),
            damage[
                "colour_mask"
            ]
        )

    cv2.imwrite(
        str(
            OUTPUT_ROOT
            / "blemish_mask.png"
        ),
        damage["blemish_mask"]
    )

    confidence_map = np.clip(
        damage["awdp_confidence_map"] * 255.0,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        str(
            OUTPUT_ROOT
            / "awdp_confidence_map.png"
        ),
        confidence_map
    )

    cv2.imwrite(
        str(
            OUTPUT_ROOT
            / "result_visualisation.jpg"
        ),
        visualisation
    )

# ============================================================
# MEMBER 4 - AWDP DAMAGE QUANTIFICATION
# ============================================================

def awdp_intensity_abnormality(image, roi_mask):
    """
    Intensity abnormality:
    darker-than-normal fruit pixels receive higher scores.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    roi_values = gray[roi_mask > 0]

    if roi_values.size == 0:
        return np.zeros_like(
            gray,
            dtype=np.float32
        )

    median_intensity = np.median(
        roi_values
    )

    score = (
        median_intensity - gray
    ) / max(
        median_intensity,
        1.0
    )

    score = np.clip(
        score,
        0.0,
        1.0
    )

    score[roi_mask == 0] = 0.0

    return score


def awdp_colour_abnormality(image, roi_mask):
    """
    Colour abnormality using LAB colour distance from
    the median colour of the fruit ROI.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    roi_pixels = lab[
        roi_mask > 0
    ]

    if roi_pixels.size == 0:
        return np.zeros(
            roi_mask.shape,
            dtype=np.float32
        )

    reference_colour = np.median(
        roi_pixels,
        axis=0
    )

    difference = (
        lab - reference_colour
    )

    distance = np.sqrt(
        np.sum(
            difference ** 2,
            axis=2
        )
    )

    roi_distance = distance[
        roi_mask > 0
    ]

    scale = np.percentile(
        roi_distance,
        95
    )

    if scale <= 0:
        scale = 1.0

    score = distance / scale

    score = np.clip(
        score,
        0.0,
        1.0
    )

    score[roi_mask == 0] = 0.0

    return score


def awdp_texture_abnormality(image, roi_mask):
    """
    Texture abnormality based on local standard deviation.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    window = 11

    local_mean = cv2.blur(
        gray,
        (window, window)
    )

    local_mean_square = cv2.blur(
        gray ** 2,
        (window, window)
    )

    variance = (
        local_mean_square
        - local_mean ** 2
    )

    variance = np.maximum(
        variance,
        0
    )

    local_std = np.sqrt(
        variance
    )

    roi_std = local_std[
        roi_mask > 0
    ]

    if roi_std.size == 0:
        return np.zeros_like(
            gray,
            dtype=np.float32
        )

    scale = np.percentile(
        roi_std,
        95
    )

    if scale <= 0:
        scale = 1.0

    score = local_std / scale

    score = np.clip(
        score,
        0.0,
        1.0
    )

    score[roi_mask == 0] = 0.0

    return score


def calculate_awdp_damage(
    image,
    roi_mask,
    blemish_mask
):
    """
    Abnormality-Weighted Damage Percentage (AWDP_A)

    Final validated weights:

        Intensity = 0.50
        Colour    = 0.30
        Texture   = 0.20

    Returns:
        damage_percentage
        confidence_map
    """

    # Make masks binary
    roi = (
        roi_mask > 0
    ).astype(np.uint8)

    blemish = (
        blemish_mask > 0
    ).astype(np.uint8)

    # -----------------------------------------------
    # Abnormality cues
    # -----------------------------------------------

    intensity = awdp_intensity_abnormality(
        image,
        roi
    )

    colour = awdp_colour_abnormality(
        image,
        roi
    )

    texture = awdp_texture_abnormality(
        image,
        roi
    )

    # -----------------------------------------------
    # AWDP_A confidence
    # -----------------------------------------------

    confidence = (
        0.50 * intensity
        + 0.30 * colour
        + 0.20 * texture
    )

    confidence = np.clip(
        confidence,
        0.0,
        1.0
    )

    # Only Otsu blemish candidates contribute
    weighted_blemish = (
        blemish.astype(np.float32)
        * confidence
    )

    weighted_blemish[
        roi == 0
    ] = 0.0

    fruit_area = np.count_nonzero(
        roi
    )

    if fruit_area == 0:
        return 0.0, confidence

    damage_percentage = (
        np.sum(weighted_blemish)
        / fruit_area
    ) * 100.0

    return (
        float(damage_percentage),
        confidence
    )

# ============================================================
# COMPLETE SINGLE-IMAGE PIPELINE
# ============================================================

def predict_image(
    image_path,
    models=None,
    save_results=True
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    if models is None:
        models = load_models()

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"OpenCV could not read: "
            f"{image_path}"
        )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    (
        resized,
        processed,
        roi_mask
    ) = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # COLOUR + TEXTURE + SHAPE
    # --------------------------------------------------------

    base_features = (
        extract_base_features(
            processed,
            roi_mask
        )
    )

    # --------------------------------------------------------
    # AUTOMATIC FRUIT IDENTIFICATION
    # --------------------------------------------------------

    (
        fruit,
        fruit_confidence
    ) = identify_fruit(
        base_features,
        models
    )

    FRUIT_CONFIDENCE_THRESHOLD = 30.0

    fruit_is_uncertain = (
        fruit_confidence
        < FRUIT_CONFIDENCE_THRESHOLD
    )

    # --------------------------------------------------------
    # DAMAGE ANALYSIS
    # --------------------------------------------------------

    damage = analyse_damage(
        processed,
        roi_mask
    )

    # --------------------------------------------------------
    # RIPENESS / GUAVA QUALITY
    # --------------------------------------------------------

    if fruit_is_uncertain:

        displayed_fruit = "Uncertain"

        condition = "Not Assessed"
        condition_confidence = 0.0
        condition_name = "Ripeness"

    else:

        displayed_fruit = fruit

        (
            condition,
            condition_confidence,
            condition_name
        ) = classify_condition(
            fruit,
            base_features,
            damage[
                "damage_percentage"
            ],
            models
        )

    # --------------------------------------------------------
    # VISUALISATION
    # --------------------------------------------------------

    visualisation = (
        create_result_visualisation(
            processed,
            damage[
                "blemish_mask"
            ],
            displayed_fruit,
            fruit_confidence,
            condition,
            condition_confidence,
            damage[
                "damage_percentage"
            ]
        )
    )

    if save_results:

        save_outputs(
            resized,
            processed,
            roi_mask,
            damage,
            visualisation
        )

    # --------------------------------------------------------
    # RESULT DICTIONARY
    # --------------------------------------------------------

    result = {

        "image":
            image_path.name,

        "fruit":
            str(displayed_fruit),

        "fruit_confidence":
            float(fruit_confidence),

        "condition_type":
            condition_name,

        "condition":
            str(condition),

        "condition_confidence":
            float(condition_confidence),

        "fruit_pixels":
            int(
                damage[
                    "fruit_pixels"
                ]
            ),

        "blemish_pixels":
            int(
                damage[
                    "blemish_pixels"
                ]
            ),

        "raw_damage_percentage":
            float(
                damage[
                    "raw_damage_percentage"
                ]
            ),

        "damage_percentage":
            float(
                damage[
                    "damage_percentage"
                ]
            ),

        "damage_level":
            damage[
                "damage_level"
            ],

        "roi_mask":
            roi_mask.copy(),

        "blemish_mask":
            damage[
                "blemish_mask"
            ].copy(),
    }

    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    print("\n")
    print("=" * 65)
    print("FRUIT QUALITY ASSESSMENT")
    print("=" * 65)

    print(
        f"\nImage                : "
        f"{result['image']}"
    )

    print(
        f"Detected Fruit       : "
        f"{result['fruit']}"
    )

    print(
        f"Fruit Confidence     : "
        f"{result['fruit_confidence']:.2f}%"
    )

    print(
        f"\n{result['condition_type']:<20} : "
        f"{result['condition']}"
    )

    print(
        f"Class Confidence     : "
        f"{result['condition_confidence']:.2f}%"
    )

    print(
        f"\nFruit Area           : "
        f"{result['fruit_pixels']:,} pixels"
    )

    print(
        f"Blemish Area         : "
        f"{result['blemish_pixels']:,} pixels"
    )

    print(
        f"Raw Otsu Damage      : "
        f"{result['raw_damage_percentage']:.2f}%"
    )

    print(
        f"AWDP Damage          : "
        f"{result['damage_percentage']:.2f}%"
    )

    print(
        f"Damage Level         : "
        f"{result['damage_level']}"
    )

    print("\n" + "=" * 65)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_ROOT
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python "
            "src/system/prediction_pipeline.py "
            "<image_path>"
        )

        print(
            "\nExample:"
        )

        print(
            "python "
            "src/system/prediction_pipeline.py "
            "\"test_images/banana.jpg\""
        )

        return

    image_path = sys.argv[1]

    try:

        models = load_models()

        result = predict_image(
            image_path,
            models=models,
            save_results=True
        )

        print_result(
            result
        )

    except Exception as error:

        print("\n" + "=" * 65)
        print("PREDICTION ERROR")
        print("=" * 65)

        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )

        print("\n" + "=" * 65)

        raise


if __name__ == "__main__":
    main()