import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "results/texture_analysis/colour_texture_features.csv"

PROCESSED_ROOT = (
    "results/preprocessing/MedianFinal/ProcessedImages"
)

ROI_ROOT = (
    "results/preprocessing/MedianFinal/ROIMasks"
)

GROUND_TRUTH_ROOT = (
    "results/blemish_detection/ground_truth"
)

SELECTION_CSV = (
    "results/blemish_detection/ground_truth_selection.csv"
)

BRUSH_SIZE = 12

os.makedirs(
    GROUND_TRUTH_ROOT,
    exist_ok=True
)

os.makedirs(
    os.path.dirname(SELECTION_CSV),
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)


# ============================================================
# SELECT 18 REPRESENTATIVE IMAGES
# ============================================================

def select_images(dataframe):

    selected_rows = []

    fruits = sorted(
        dataframe["fruit"].unique()
    )

    for fruit in fruits:

        fruit_df = dataframe[
            dataframe["fruit"] == fruit
        ]

        # Guava uses different quality labels
        if fruit == "Guava":

            categories = [
                "Class_A",
                "Class_B",
                "Defect"
            ]

        # Other fruits
        else:

            categories = [
                "Ripe",
                "Overripe",
                "Rotten"
            ]

        # Select one image from each category
        for category in categories:

            category_df = fruit_df[
                fruit_df["category"] == category
            ]

            if len(category_df) > 0:

                selected = category_df.sample(
                    n=1,
                    random_state=42
                )

                selected_rows.append(
                    selected
                )

    selected_df = pd.concat(
        selected_rows,
        ignore_index=True
    )

    return selected_df


# ============================================================
# LOAD OR CREATE SELECTION
# ============================================================

# Important:
# Once images have been selected, keep using the same
# ground_truth_selection.csv so the selection does not change.

if os.path.exists(SELECTION_CSV):

    print(
        "\nExisting ground-truth selection found."
    )

    selected_df = pd.read_csv(
        SELECTION_CSV
    )

else:

    selected_df = select_images(
        df
    )

    selected_df.to_csv(
        SELECTION_CSV,
        index=False
    )

    print(
        "\nNew ground-truth selection created."
    )


print(
    f"Images selected: {len(selected_df)}"
)


# ============================================================
# GLOBAL DRAWING VARIABLES
# ============================================================

drawing = False
erase_mode = False

current_mask = None
display_image = None
original_image = None

brush_size = BRUSH_SIZE


# ============================================================
# MOUSE CALLBACK
# ============================================================

def draw_mask(event, x, y, flags, param):

    global drawing
    global erase_mode
    global current_mask
    global display_image
    global original_image
    global brush_size

    # --------------------------------------------------------
    # LEFT CLICK = PAINT BLEMISH
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        erase_mode = False

    # --------------------------------------------------------
    # RIGHT CLICK = ERASE
    # --------------------------------------------------------

    elif event == cv2.EVENT_RBUTTONDOWN:

        drawing = True
        erase_mode = True

    # --------------------------------------------------------
    # STOP DRAWING
    # --------------------------------------------------------

    elif (
        event == cv2.EVENT_LBUTTONUP
        or event == cv2.EVENT_RBUTTONUP
    ):

        drawing = False

    # --------------------------------------------------------
    # DRAW WHILE MOUSE MOVES
    # --------------------------------------------------------

    if drawing:

        if erase_mode:

            value = 0

        else:

            value = 255

        cv2.circle(
            current_mask,
            (x, y),
            brush_size,
            value,
            -1
        )

        update_display()


# ============================================================
# UPDATE DISPLAY
# ============================================================

def update_display():

    global display_image
    global original_image
    global current_mask

    display_image = original_image.copy()

    # Red overlay shows manually labelled blemish
    overlay = display_image.copy()

    overlay[
        current_mask > 0
    ] = (0, 0, 255)

    display_image = cv2.addWeighted(
        display_image,
        0.70,
        overlay,
        0.30,
        0
    )


# ============================================================
# MAIN ANNOTATION LOOP
# ============================================================

print("\n==========================================")
print("GROUND TRUTH ANNOTATION")
print("==========================================")

print("LEFT MOUSE  : Paint blemish")
print("RIGHT MOUSE : Erase")
print("+            : Increase brush size")
print("-            : Decrease brush size")
print("R            : Reset current mask")
print("S            : Save and next")
print("Q            : Quit")
print("==========================================\n")


for index, row in selected_df.iterrows():

    relative_path = row["relative_path"]

    image_path = os.path.join(
        PROCESSED_ROOT,
        relative_path
    )

    relative_folder = os.path.dirname(
        relative_path
    )

    filename = os.path.basename(
        relative_path
    )

    filename_without_extension = (
        os.path.splitext(filename)[0]
    )


    # ========================================================
    # ROI MASK PATH
    # ========================================================

    roi_path = os.path.join(
        ROI_ROOT,
        relative_folder,
        filename_without_extension
        + "_mask.png"
    )


    # ========================================================
    # OUTPUT GROUND TRUTH PATH
    # ========================================================

    output_folder = os.path.join(
        GROUND_TRUTH_ROOT,
        row["fruit"]
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        output_folder,
        filename_without_extension
        + "_gt.png"
    )


    # ========================================================
    # SKIP ALREADY ANNOTATED IMAGE
    # ========================================================

    if os.path.exists(output_path):

        print(
            f"[{index + 1}/{len(selected_df)}] "
            f"Already completed: {filename}"
        )

        continue


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    original_image = cv2.imread(
        image_path
    )

    roi_mask = cv2.imread(
        roi_path,
        cv2.IMREAD_GRAYSCALE
    )


    if (
        original_image is None
        or roi_mask is None
    ):

        print(
            f"FAILED: {relative_path}"
        )

        continue


    # ========================================================
    # CREATE EMPTY GROUND TRUTH MASK
    # ========================================================

    current_mask = np.zeros(
        original_image.shape[:2],
        dtype=np.uint8
    )

    update_display()


    # ========================================================
    # WINDOW
    # ========================================================

    window_name = (
        f"Ground Truth - "
        f"{row['fruit']} - "
        f"{row['category']}"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        900,
        750
    )

    cv2.setMouseCallback(
        window_name,
        draw_mask
    )


    print()
    print(
        f"[{index + 1}/{len(selected_df)}]"
    )

    print(
        f"Fruit    : {row['fruit']}"
    )

    print(
        f"Category : {row['category']}"
    )

    print(
        f"Image    : {filename}"
    )


    # ========================================================
    # INTERACTION LOOP
    # ========================================================

    while True:

        preview = display_image.copy()

        # ----------------------------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            preview,
            f"Brush: {brush_size}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            preview,
            "Left=Paint | Right=Erase | S=Save | Q=Quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            window_name,
            preview
        )

        key = cv2.waitKey(20) & 0xFF


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if key == ord("s"):

            # Ensure ground truth remains inside fruit ROI
            current_mask[
                roi_mask == 0
            ] = 0

            cv2.imwrite(
                output_path,
                current_mask
            )

            print(
                f"Saved: {output_path}"
            )

            break


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        elif key == ord("r"):

            current_mask[:] = 0

            update_display()

            print(
                "Current mask reset."
            )


        # ----------------------------------------------------
        # INCREASE BRUSH
        # ----------------------------------------------------

        elif key in (
            ord("+"),
            ord("=")
        ):

            brush_size = min(
                brush_size + 3,
                60
            )

            print(
                f"Brush size: {brush_size}"
            )


        # ----------------------------------------------------
        # DECREASE BRUSH
        # ----------------------------------------------------

        elif key in (
            ord("-"),
            ord("_")
        ):

            brush_size = max(
                brush_size - 3,
                2
            )

            print(
                f"Brush size: {brush_size}"
            )


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        elif key == ord("q"):

            print(
                "\nAnnotation stopped."
            )

            cv2.destroyAllWindows()

            print(
                "Run the program again later "
                "to continue."
            )

            raise SystemExit


    cv2.destroyWindow(
        window_name
    )


# ============================================================
# FINISHED
# ============================================================

cv2.destroyAllWindows()

print("\n==========================================")
print("GROUND TRUTH ANNOTATION COMPLETED")
print("==========================================")

print(
    f"Ground-truth masks saved to:\n"
    f"{GROUND_TRUTH_ROOT}"
)

print("==========================================")