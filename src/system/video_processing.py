"""
============================================================
FRUIT QUALITY ASSESSMENT - VIDEO PROCESSING
============================================================

Supports:
1. Video file processing
2. Webcam processing
3. Automatic fruit identification
4. Ripeness / Guava quality classification
5. AWDP surface-damage assessment
6. Temporal smoothing
7. Frame skipping for faster processing
8. Saving processed output video

This file reuses the existing prediction_pipeline.py functions.
============================================================
"""

import sys
import time
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = PROJECT_ROOT / "src"

SYSTEM_DIR = SRC_ROOT / "system"

if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))


# ============================================================
# IMPORT EXISTING SINGLE-IMAGE PIPELINE
# ============================================================

from prediction_pipeline import (
    load_models,
    preprocess_image,
    extract_base_features,
    identify_fruit,
    analyse_damage,
    classify_condition,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Analyse one frame every N frames.
# Increase this if video processing is too slow.
PROCESS_EVERY_N_FRAMES = 10

# Number of recent predictions used for smoothing.
SMOOTHING_WINDOW = 5

# Display size
DISPLAY_WIDTH = 960

FRUIT_CONFIDENCE_THRESHOLD = 30.0

# Output directory
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "system"
    / "video"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEMPORAL SMOOTHER
# ============================================================

class PredictionSmoother:
    """
    Smooth predictions across several analysed frames.

    Fruit and condition:
        majority voting

    Confidence and damage:
        average values
    """

    def __init__(
        self,
        window_size=5
    ):

        self.window_size = window_size

        self.fruits = deque(
            maxlen=window_size
        )

        self.fruit_confidences = deque(
            maxlen=window_size
        )

        self.conditions = deque(
            maxlen=window_size
        )

        self.condition_types = deque(
            maxlen=window_size
        )

        self.condition_confidences = deque(
            maxlen=window_size
        )

        self.damage_percentages = deque(
            maxlen=window_size
        )

        self.damage_levels = deque(
            maxlen=window_size
        )

    def reset(self):

        self.fruits.clear()
        self.fruit_confidences.clear()

        self.conditions.clear()
        self.condition_types.clear()
        self.condition_confidences.clear()

        self.damage_percentages.clear()
        self.damage_levels.clear()

    def add(
        self,
        result
    ):

        self.fruits.append(
            result["fruit"]
        )

        self.fruit_confidences.append(
            result["fruit_confidence"]
        )

        self.conditions.append(
            result["condition"]
        )

        self.condition_types.append(
            result["condition_type"]
        )

        self.condition_confidences.append(
            result["condition_confidence"]
        )

        self.damage_percentages.append(
            result["damage_percentage"]
        )

        self.damage_levels.append(
            result["damage_level"]
        )

    @staticmethod
    def majority_vote(values):

        if not values:
            return None

        counter = Counter(values)

        return counter.most_common(1)[0][0]

    def get_smoothed_result(self):

        if not self.fruits:
            return None

        fruit = self.majority_vote(
            list(self.fruits)
        )

        condition = self.majority_vote(
            list(self.conditions)
        )

        condition_type = self.majority_vote(
            list(self.condition_types)
        )

        damage_level = self.majority_vote(
            list(self.damage_levels)
        )

        fruit_confidence = float(
            np.mean(
                self.fruit_confidences
            )
        )

        condition_confidence = float(
            np.mean(
                self.condition_confidences
            )
        )

        damage_percentage = float(
            np.mean(
                self.damage_percentages
            )
        )

        return {
            "fruit":
                fruit,

            "fruit_confidence":
                fruit_confidence,

            "condition":
                condition,

            "condition_type":
                condition_type,

            "condition_confidence":
                condition_confidence,

            "damage_percentage":
                damage_percentage,

            "damage_level":
                damage_level,
        }


# ============================================================
# PROCESS ONE VIDEO FRAME
# ============================================================

def analyse_frame(
    frame,
    models
):
    """
    Run the complete fruit-quality pipeline on one video frame.

    Returns:
        result dictionary
        processed image
        roi mask
        blemish mask
    """

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    (
        resized,
        processed,
        roi_mask
    ) = preprocess_image(
        frame
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
    # FRUIT IDENTIFICATION
    # --------------------------------------------------------

    (
        fruit,
        fruit_confidence
    ) = identify_fruit(
        base_features,
        models
    )

    # --------------------------------------------------------
    # SURFACE DAMAGE
    # --------------------------------------------------------

    damage = analyse_damage(
        processed,
        roi_mask
    )

    # --------------------------------------------------------
    # RIPENESS / GUAVA QUALITY
    # --------------------------------------------------------

    (
        condition,
        condition_confidence,
        condition_type
    ) = classify_condition(
        fruit,
        base_features,
        damage["damage_percentage"],
        models
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "fruit":
            str(fruit),

        "fruit_confidence":
            float(fruit_confidence),

        "condition":
            str(condition),

        "condition_type":
            str(condition_type),

        "condition_confidence":
            float(condition_confidence),

        "damage_percentage":
            float(
                damage[
                    "damage_percentage"
                ]
            ),

        "damage_level":
            str(
                damage[
                    "damage_level"
                ]
            ),

        "raw_damage_percentage":
            float(
                damage[
                    "raw_damage_percentage"
                ]
            ),

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
    }

    return (
        result,
        processed,
        roi_mask,
        damage["blemish_mask"]
    )


# ============================================================
# DRAW BLEMISH OVERLAY
# ============================================================

def create_blemish_overlay(
    processed,
    blemish_mask
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
        0.70,
        overlay,
        0.30,
        0
    )

    return result


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_information_panel(
    frame,
    result,
    fps_value=None,
    analysing=False
):

    display = frame.copy()

    height, width = display.shape[:2]

    # --------------------------------------------------------
    # DARK INFORMATION PANEL
    # --------------------------------------------------------

    panel_height = 220

    overlay = display.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (
            min(width - 10, 570),
            panel_height
        ),
        (0, 0, 0),
        -1
    )

    display = cv2.addWeighted(
        overlay,
        0.65,
        display,
        0.35,
        0
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    cv2.putText(
        display,
        "FRUIT QUALITY ASSESSMENT",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # NO RESULT YET
    # --------------------------------------------------------

    if result is None:

        cv2.putText(
            display,
            "Analysing fruit...",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return display


    fruit_confidence = result["fruit_confidence"]

    if fruit_confidence < FRUIT_CONFIDENCE_THRESHOLD:
        fruit_text = (
            f"Fruit: Uncertain "
            f"({fruit_confidence:.1f}%)"
        )
    else:
        fruit_text = (
            f"Fruit: "
            f"{result['fruit']} "
            f"({fruit_confidence:.1f}%)"
        )

    cv2.putText(
        display,
        fruit_text,
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    # --------------------------------------------------------
    # CONDITION
    # --------------------------------------------------------

    if fruit_confidence < FRUIT_CONFIDENCE_THRESHOLD:

        condition_text = "Ripeness: Not assessed"

    else:

        condition_label = result["condition_type"]

        condition_text = (
            f"{condition_label}: "
            f"{result['condition']} "
            f"({result['condition_confidence']:.1f}%)"
        )

    cv2.putText(
        display,
        condition_text,
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # SURFACE DAMAGE
    # --------------------------------------------------------

    damage_text = (
        f"Surface Damage: "
        f"{result['damage_percentage']:.2f}%"
    )

    cv2.putText(
        display,
        damage_text,
        (30, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    damage_level_text = (
        f"Surface Damage Level: "
        f"{result['damage_level']}"
    )

    cv2.putText(
        display,
        damage_level_text,
        (30, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if fps_value is not None:

        fps_text = (
            f"Processing FPS: "
            f"{fps_value:.1f}"
        )

        cv2.putText(
            display,
            fps_text,
            (
                max(
                    width - 260,
                    20
                ),
                height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # ANALYSIS INDICATOR
    # --------------------------------------------------------

    if analysing:

        cv2.putText(
            display,
            "ANALYSING",
            (
                max(
                    width - 160,
                    20
                ),
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return display


# ============================================================
# RESIZE FOR DISPLAY
# ============================================================

def resize_for_display(
    frame,
    target_width=DISPLAY_WIDTH
):

    height, width = frame.shape[:2]

    if width <= target_width:
        return frame

    ratio = (
        target_width
        / float(width)
    )

    target_height = int(
        height * ratio
    )

    return cv2.resize(
        frame,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    video_path,
    process_every=PROCESS_EVERY_N_FRAMES,
    smoothing_window=SMOOTHING_WINDOW
):

    video_path = Path(
        video_path
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: "
            f"{video_path}"
        )

    # --------------------------------------------------------
    # LOAD MODELS ONCE
    # --------------------------------------------------------

    print("\nLoading trained models...")

    models = load_models()

    print("Models loaded successfully.")

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise ValueError(
            f"Could not open video: "
            f"{video_path}"
        )

    source_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if (
        source_fps <= 0
        or np.isnan(source_fps)
    ):

        source_fps = 30.0

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    output_path = (
        OUTPUT_ROOT
        / (
            video_path.stem
            + "_assessed.mp4"
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        source_fps,
        (
            frame_width,
            frame_height
        )
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # SMOOTHER
    # --------------------------------------------------------

    smoother = PredictionSmoother(
        smoothing_window
    )

    latest_result = None

    latest_processed = None

    latest_blemish_mask = None

    frame_number = 0

    analysed_frames = 0

    failed_frames = 0

    start_time = time.time()

    print("\n" + "=" * 70)

    print(
        "VIDEO FRUIT QUALITY ASSESSMENT"
    )

    print("=" * 70)

    print(
        f"Input video       : "
        f"{video_path}"
    )

    print(
        f"Resolution        : "
        f"{frame_width}x{frame_height}"
    )

    print(
        f"Source FPS        : "
        f"{source_fps:.2f}"
    )

    print(
        f"Total frames      : "
        f"{total_frames}"
    )

    print(
        f"Analyse every     : "
        f"{process_every} frame(s)"
    )

    print(
        f"Smoothing window  : "
        f"{smoothing_window}"
    )

    print(
        "\nPress Q to stop processing."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        should_analyse = (
            frame_number == 1
            or frame_number
            % process_every == 0
        )

        # ----------------------------------------------------
        # ANALYSE SELECTED FRAME
        # ----------------------------------------------------

        if should_analyse:

            try:

                (
                    frame_result,
                    processed,
                    roi_mask,
                    blemish_mask
                ) = analyse_frame(
                    frame,
                    models
                )

                smoother.add(
                    frame_result
                )

                latest_result = (
                    smoother.get_smoothed_result()
                )

                latest_processed = processed

                latest_blemish_mask = (
                    blemish_mask
                )

                analysed_frames += 1

            except Exception as error:

                failed_frames += 1

                print(
                    f"\n[WARNING] "
                    f"Frame {frame_number} "
                    f"could not be analysed: "
                    f"{error}"
                )

        # ----------------------------------------------------
        # CREATE OUTPUT FRAME
        # ----------------------------------------------------

        output_frame = frame.copy()

        elapsed = max(
            time.time() - start_time,
            0.0001
        )

        processing_fps = (
            frame_number / elapsed
        )

        output_frame = (
            draw_information_panel(
                output_frame,
                latest_result,
                fps_value=processing_fps,
                analysing=should_analyse
            )
        )

        # ----------------------------------------------------
        # WRITE OUTPUT
        # ----------------------------------------------------

        writer.write(
            output_frame
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_frame = resize_for_display(
            output_frame
        )

        cv2.imshow(
            "Fruit Quality Assessment - Video",
            display_frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            print(
                "\nProcessing stopped by user."
            )
            break

    # --------------------------------------------------------
    # CLEAN UP
    # --------------------------------------------------------

    capture.release()

    writer.release()

    cv2.destroyAllWindows()

    elapsed = (
        time.time()
        - start_time
    )

    print("\n" + "=" * 70)

    print(
        "VIDEO PROCESSING COMPLETED"
    )

    print("=" * 70)

    print(
        f"Frames read       : "
        f"{frame_number}"
    )

    print(
        f"Frames analysed   : "
        f"{analysed_frames}"
    )

    print(
        f"Failed analyses   : "
        f"{failed_frames}"
    )

    print(
        f"Processing time   : "
        f"{elapsed:.2f} seconds"
    )

    print(
        "\nOutput saved to:"
    )

    print(
        output_path
    )

    print("=" * 70)

    return output_path


# ============================================================
# WEBCAM PROCESSING
# ============================================================

def find_available_camera(
    max_index=5
):
    """
    Search camera indices 0 to max_index.

    A camera is accepted only when OpenCV can open it and successfully
    read a real frame. On Windows, DirectShow is tried first because it
    is often more reliable for USB/integrated webcams.
    """

    print(
        f"Searching camera indices "
        f"0 to {max_index}..."
    )

    for camera_index in range(
        max_index + 1
    ):

        if sys.platform.startswith(
            "win"
        ):

            backends = [
                cv2.CAP_DSHOW,
                cv2.CAP_ANY
            ]

        else:

            backends = [
                cv2.CAP_ANY
            ]

        for backend in backends:

            capture = cv2.VideoCapture(
                camera_index,
                backend
            )

            if not capture.isOpened():

                capture.release()

                continue

            success, frame = capture.read()

            capture.release()

            if (
                success
                and frame is not None
                and frame.size > 0
            ):

                print(
                    f"Webcam detected at "
                    f"camera index "
                    f"{camera_index}."
                )

                return camera_index

    return None


def open_camera_capture(
    camera_index
):
    """
    Open the selected webcam.

    DirectShow is preferred on Windows. If that fails, OpenCV's default
    backend is used as a fallback.
    """

    if sys.platform.startswith(
        "win"
    ):

        capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if capture.isOpened():

            return capture

        capture.release()

    return cv2.VideoCapture(
        camera_index
    )


def process_camera(
    camera_index=0,
    process_every=PROCESS_EVERY_N_FRAMES,
    smoothing_window=SMOOTHING_WINDOW
):

    print("\nLoading trained models...")

    models = load_models()

    print("Models loaded successfully.")

    # --------------------------------------------------------
    # CAMERA SELECTION
    # --------------------------------------------------------

    if (
        camera_index is None
        or str(
            camera_index
        ).lower() == "auto"
    ):

        camera_index = find_available_camera(
            max_index=5
        )

        if camera_index is None:

            raise RuntimeError(
                "No webcam detected. "
                "Please connect or enable a webcam "
                "and try again."
            )

    camera_index = int(
        camera_index
    )

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    capture = open_camera_capture(
        camera_index
    )

    if not capture.isOpened():

        capture.release()

        raise RuntimeError(
            f"Could not open camera "
            f"{camera_index}. "
            "The camera may be disabled, "
            "already in use by another program, "
            "or blocked by camera privacy settings."
        )

    smoother = PredictionSmoother(
        smoothing_window
    )

    latest_result = None

    frame_number = 0

    analysed_frames = 0

    failed_frames = 0

    start_time = time.time()

    print("\n" + "=" * 70)

    print(
        "LIVE FRUIT QUALITY ASSESSMENT"
    )

    print("=" * 70)

    print(
        f"Camera index      : "
        f"{camera_index}"
    )

    print(
        f"Analyse every     : "
        f"{process_every} frame(s)"
    )

    print(
        f"Smoothing window  : "
        f"{smoothing_window}"
    )

    print(
        "\nPress Q to close webcam."
    )

    print("=" * 70)

    while True:

        success, frame = capture.read()

        if not success:

            print(
                "\n[WARNING] "
                "Could not read webcam frame."
            )

            break

        frame_number += 1

        should_analyse = (
            frame_number == 1
            or frame_number
            % process_every == 0
        )

        # ----------------------------------------------------
        # ANALYSE
        # ----------------------------------------------------

        if should_analyse:

            try:

                (
                    frame_result,
                    processed,
                    roi_mask,
                    blemish_mask
                ) = analyse_frame(
                    frame,
                    models
                )

                smoother.add(
                    frame_result
                )

                latest_result = (
                    smoother.get_smoothed_result()
                )

                analysed_frames += 1

            except Exception as error:

                failed_frames += 1

                print(
                    f"\n[WARNING] "
                    f"Frame {frame_number}: "
                    f"{error}"
                )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = max(
            time.time() - start_time,
            0.0001
        )

        processing_fps = (
            frame_number / elapsed
        )

        # ----------------------------------------------------
        # DRAW RESULT
        # ----------------------------------------------------

        output_frame = (
            draw_information_panel(
                frame,
                latest_result,
                fps_value=processing_fps,
                analysing=should_analyse
            )
        )

        display_frame = resize_for_display(
            output_frame
        )

        cv2.imshow(
            "Live Fruit Quality Assessment",
            display_frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    capture.release()

    cv2.destroyAllWindows()

    print("\n" + "=" * 70)

    print(
        "WEBCAM PROCESSING ENDED"
    )

    print("=" * 70)

    print(
        f"Frames read       : "
        f"{frame_number}"
    )

    print(
        f"Frames analysed   : "
        f"{analysed_frames}"
    )

    print(
        f"Failed analyses   : "
        f"{failed_frames}"
    )

    print("=" * 70)


# ============================================================
# COMMAND LINE
# ============================================================

def print_usage():

    print("\nUsage:")

    print(
        "\n1. Process video file:"
    )

    print(
        "python "
        "src/system/video_processing.py "
        "\"test_videos/fruit_test.mp4\""
    )

    print(
        "\n2. Use webcam:"
    )

    print(
        "python "
        "src/system/video_processing.py "
        "--camera auto"
    )

    print(
        "\n3. Use another camera index:"
    )

    print(
        "python "
        "src/system/video_processing.py "
        "--camera 1"
    )


def main():

    if len(sys.argv) < 2:

        print_usage()

        return

    argument = sys.argv[1]

    try:

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        if argument.lower() == "--camera":

            # No index means automatic camera detection.
            camera_index = "auto"

            if len(sys.argv) >= 3:

                camera_argument = (
                    sys.argv[2]
                    .strip()
                )

                if (
                    camera_argument.lower()
                    != "auto"
                ):

                    camera_index = int(
                        camera_argument
                    )

            process_camera(
                camera_index=camera_index
            )

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        else:

            process_video(
                argument
            )

    except KeyboardInterrupt:

        cv2.destroyAllWindows()

        print(
            "\nProcessing interrupted."
        )

    except Exception as error:

        cv2.destroyAllWindows()

        print("\n" + "=" * 70)

        print(
            "VIDEO PROCESSING ERROR"
        )

        print("=" * 70)

        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )

        print("\n" + "=" * 70)

        raise


if __name__ == "__main__":
    main()