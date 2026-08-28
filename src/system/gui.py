"""
======================================================================
FRUIT QUALITY ASSESSMENT SYSTEM - GUI
======================================================================

Final graphical user interface for the integrated system.

Functions:
1. Select an image
2. Preview selected image
3. Run complete fruit-quality assessment
4. Display fruit identification
5. Display ripeness / Guava quality
6. Display fruit confidence
7. Display condition confidence
8. Display Otsu + Morphology damage
9. Display AWDP damage
10. Display ROI mask
11. Display blemish mask
12. Launch video analysis
13. Handle uncertain fruit predictions

The GUI reuses prediction_pipeline.py.
======================================================================
"""

import sys
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_DIR = (
    PROJECT_ROOT
    / "src"
    / "system"
)

if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SYSTEM_DIR)
    )


# ======================================================================
# IMPORT EXISTING PIPELINE
# ======================================================================

from prediction_pipeline import (
    load_models,
    predict_image,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

FRUIT_CONFIDENCE_THRESHOLD = 30.0

PREVIEW_WIDTH = 500
PREVIEW_HEIGHT = 300

MASK_WIDTH = 280
MASK_HEIGHT = 180


# ======================================================================
# GUI APPLICATION
# ======================================================================

class FruitQualityGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Fruit Quality Assessment System"
        )

        self.root.geometry("1250x850")
        self.root.minsize(1100, 700)

        # Start maximized on Windows
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        # --------------------------------------------------------------
        # APPLICATION STATE
        # --------------------------------------------------------------

        self.selected_image_path = None

        self.original_photo = None
        self.roi_photo = None
        self.blemish_photo = None

        self.models = None

        # --------------------------------------------------------------
        # TKINTER VARIABLES
        # --------------------------------------------------------------

        self.status_var = tk.StringVar(
            value="Loading trained models..."
        )

        self.file_var = tk.StringVar(
            value="No image selected"
        )

        self.fruit_var = tk.StringVar(
            value="-"
        )

        self.fruit_confidence_var = tk.StringVar(
            value="-"
        )

        self.condition_title_var = tk.StringVar(
            value="Ripeness"
        )

        self.condition_var = tk.StringVar(
            value="-"
        )

        self.condition_confidence_var = tk.StringVar(
            value="-"
        )

        self.fruit_area_var = tk.StringVar(
            value="-"
        )

        self.blemish_area_var = tk.StringVar(
            value="-"
        )

        self.raw_damage_var = tk.StringVar(
            value="-"
        )

        self.awdp_damage_var = tk.StringVar(
            value="-"
        )

        self.damage_level_var = tk.StringVar(
            value="-"
        )

        # --------------------------------------------------------------
        # BUILD INTERFACE
        # --------------------------------------------------------------

        self.configure_styles()

        self.build_header()

        self.build_status_bar()

        self.build_buttons()

        self.build_main_area()

        # --------------------------------------------------------------
        # LOAD MODELS AFTER WINDOW APPEARS
        # --------------------------------------------------------------

        self.root.after(
            100,
            self.load_trained_models
        )


    # ==================================================================
    # STYLE
    # ==================================================================

    def configure_styles(self):

        style = ttk.Style()

        try:
            style.theme_use(
                "clam"
            )
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=(
                "Segoe UI",
                22,
                "bold"
            )
        )

        style.configure(
            "Subtitle.TLabel",
            font=(
                "Segoe UI",
                10
            )
        )

        style.configure(
            "Section.TLabel",
            font=(
                "Segoe UI",
                13,
                "bold"
            )
        )

        style.configure(
            "ResultName.TLabel",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        )

        style.configure(
            "ResultValue.TLabel",
            font=(
                "Segoe UI",
                10
            )
        )

        style.configure(
            "Primary.TButton",
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            padding=8
        )

        style.configure(
            "Normal.TButton",
            font=(
                "Segoe UI",
                10
            ),
            padding=8
        )


    # ==================================================================
    # HEADER
    # ==================================================================

    def build_header(self):

        header = ttk.Frame(
            self.root,
            padding=(
                20,
                18,
                20,
                10
            )
        )

        header.pack(
            fill="x"
        )

        title = ttk.Label(
            header,
            text=(
                "FRUIT QUALITY "
                "ASSESSMENT SYSTEM"
            ),
            style="Title.TLabel"
        )

        title.pack()

        subtitle = ttk.Label(
            header,
            text=(
                "Automatic Fruit Identification, "
                "Ripeness Classification and "
                "Surface Damage Quantification"
            ),
            style="Subtitle.TLabel"
        )

        subtitle.pack(
            pady=(
                5,
                0
            )
        )


    # ==================================================================
    # MAIN AREA
    # ==================================================================

    def build_main_area(self):

        main = ttk.Frame(
            self.root,
            padding=(
                20,
                10
            )
        )

        main.pack(
            fill="both",
            expand=True,
        )

        main.columnconfigure(
            0,
            weight=3
        )

        main.columnconfigure(
            1,
            weight=2
        )

        main.rowconfigure(
            0,
            weight=1
        )

        # --------------------------------------------------------------
        # LEFT SIDE
        # --------------------------------------------------------------

        left = ttk.LabelFrame(
            main,
            text="Image Analysis",
            padding=15
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(
                0,
                10
            )
        )

        # --------------------------------------------------------------
        # FILE NAME
        # --------------------------------------------------------------

        file_label = ttk.Label(
            left,
            textvariable=self.file_var,
            anchor="center"
        )

        file_label.pack(
            fill="x",
            pady=(
                0,
                10
            )
        )

        # --------------------------------------------------------------
        # ORIGINAL IMAGE
        # --------------------------------------------------------------

        ttk.Label(
            left,
            text="Original Image",
            style="Section.TLabel"
        ).pack(
            pady=(
                0,
                8
            )
        )

        self.original_image_label = tk.Label(
            left,
            text="Click here to select an image",
            relief="solid",
            borderwidth=1,
            anchor="center",
            cursor="hand2"
        )


        self.original_image_label.bind(
            "<Button-1>",
            lambda event: self.select_image()
        )

        self.original_image_label.pack(
            pady=(
                0,
                15
            )
        )

        # --------------------------------------------------------------
        # MASK AREA
        # --------------------------------------------------------------

        mask_frame = ttk.Frame(
            left
        )

        mask_frame.pack(
            fill="both",
            expand=True
        )

        mask_frame.columnconfigure(
            0,
            weight=1
        )

        mask_frame.columnconfigure(
            1,
            weight=1
        )

        # ROI -----------------------------------------------------------

        roi_container = ttk.Frame(
            mask_frame
        )

        roi_container.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(
                0,
                5
            )
        )

        ttk.Label(
            roi_container,
            text="Fruit ROI Mask",
            style="Section.TLabel"
        ).pack(
            pady=(
                0,
                5
            )
        )

        self.roi_label = tk.Label(
            roi_container,
            text="ROI mask",
            relief="solid",
            borderwidth=1,
            width=30,
            height=10
        )

        self.roi_label.pack(
            expand=True
        )

        # BLEMISH -------------------------------------------------------

        blemish_container = ttk.Frame(
            mask_frame
        )

        blemish_container.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(
                5,
                0
            )
        )

        ttk.Label(
            blemish_container,
            text="Blemish Mask",
            style="Section.TLabel"
        ).pack(
            pady=(
                0,
                5
            )
        )

        self.blemish_label = tk.Label(
            blemish_container,
            text="Blemish mask",
            relief="solid",
            borderwidth=1,
            width=30,
            height=10
        )

        self.blemish_label.pack(
            expand=True
        )

        # --------------------------------------------------------------
        # RIGHT SIDE
        # --------------------------------------------------------------

        right = ttk.LabelFrame(
            main,
            text="Assessment Results",
            padding=20
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(
                10,
                0
            )
        )

        self.build_results_panel(
            right
        )


    # ==================================================================
    # RESULTS PANEL
    # ==================================================================

    def build_results_panel(
        self,
        parent
    ):

        # --------------------------------------------------------------
        # FRUIT IDENTIFICATION
        # --------------------------------------------------------------

        ttk.Label(
            parent,
            text="Fruit Identification",
            style="Section.TLabel"
        ).pack(
            anchor="w",
            pady=(
                0,
                10
            )
        )

        fruit_frame = ttk.Frame(
            parent
        )

        fruit_frame.pack(
            fill="x",
            pady=(
                0,
                15
            )
        )

        self.add_result_row(
            fruit_frame,
            0,
            "Detected Fruit:",
            self.fruit_var
        )

        self.add_result_row(
            fruit_frame,
            1,
            "Fruit Confidence:",
            self.fruit_confidence_var
        )

        ttk.Separator(
            parent,
            orient="horizontal"
        ).pack(
            fill="x",
            pady=10
        )

        # --------------------------------------------------------------
        # CONDITION
        # --------------------------------------------------------------

        ttk.Label(
            parent,
            text="Condition Assessment",
            style="Section.TLabel"
        ).pack(
            anchor="w",
            pady=(
                5,
                10
            )
        )

        condition_frame = ttk.Frame(
            parent
        )

        condition_frame.pack(
            fill="x",
            pady=(
                0,
                15
            )
        )

        self.condition_name_label = ttk.Label(
            condition_frame,
            textvariable=self.condition_title_var,
            style="ResultName.TLabel"
        )

        self.condition_name_label.grid(
            row=0,
            column=0,
            sticky="w",
            pady=5
        )

        ttk.Label(
            condition_frame,
            text=":",
            style="ResultName.TLabel"
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=5
        )

        ttk.Label(
            condition_frame,
            textvariable=self.condition_var,
            style="ResultValue.TLabel"
        ).grid(
            row=0,
            column=2,
            sticky="w",
            pady=5
        )

        self.add_result_row(
            condition_frame,
            1,
            "Confidence:",
            self.condition_confidence_var
        )

        ttk.Separator(
            parent,
            orient="horizontal"
        ).pack(
            fill="x",
            pady=10
        )

        # --------------------------------------------------------------
        # DAMAGE
        # --------------------------------------------------------------

        ttk.Label(
            parent,
            text="Surface Damage Analysis",
            style="Section.TLabel"
        ).pack(
            anchor="w",
            pady=(
                5,
                10
            )
        )

        damage_frame = ttk.Frame(
            parent
        )

        damage_frame.pack(
            fill="x"
        )

        self.add_result_row(
            damage_frame,
            0,
            "Fruit Area:",
            self.fruit_area_var
        )

        self.add_result_row(
            damage_frame,
            1,
            "Blemish Area:",
            self.blemish_area_var
        )

        self.add_result_row(
            damage_frame,
            2,
            "Raw Otsu Damage:",
            self.raw_damage_var
        )

        self.add_result_row(
            damage_frame,
            3,
            "AWDP Damage:",
            self.awdp_damage_var
        )

        self.add_result_row(
            damage_frame,
            4,
            "Damage Level:",
            self.damage_level_var
        )

        # --------------------------------------------------------------
        # METHOD INFORMATION
        # --------------------------------------------------------------

        ttk.Separator(
            parent,
            orient="horizontal"
        ).pack(
            fill="x",
            pady=15
        )

        ttk.Label(
            parent,
            text="Methods",
            style="Section.TLabel"
        ).pack(
            anchor="w",
            pady=(
                0,
                8
            )
        )

        method_text = (
            "Preprocessing: Median Filtering\n"
            "Fruit Features: Colour + Texture + Shape\n"
            "Fruit Classifier: Random Forest\n"
            "Condition Classifier: Random Forest\n"
            "Blemish Segmentation: Otsu + Morphology 7x7\n"
            "Damage Quantification: AWDP"
        )

        ttk.Label(
            parent,
            text=method_text,
            justify="left"
        ).pack(
            anchor="w"
        )


    # ==================================================================
    # RESULT ROW
    # ==================================================================

    def add_result_row(
        self,
        parent,
        row,
        label,
        variable
    ):

        ttk.Label(
            parent,
            text=label,
            style="ResultName.TLabel"
        ).grid(
            row=row,
            column=0,
            sticky="w",
            pady=5
        )

        ttk.Label(
            parent,
            textvariable=variable,
            style="ResultValue.TLabel"
        ).grid(
            row=row,
            column=1,
            sticky="w",
            padx=(
                15,
                0
            ),
            pady=5
        )


    # ==================================================================
    # BUTTON AREA
    # ==================================================================

    def build_buttons(self):

        button_frame = ttk.Frame(
            self.root,
            padding=(20, 10)
        )

        button_frame.pack(
            side="bottom",
            fill="x",
        )

        self.select_button = ttk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image,
            style="Normal.TButton"
        )

        self.select_button.pack(
            side="left",
            padx=5
        )

        self.analyse_button = ttk.Button(
            button_frame,
            text="Analyze Image",
            command=self.analyse_selected_image,
            style="Primary.TButton",
            state="disabled"
        )

        self.analyse_button.pack(
            side="left",
            padx=5
        )

        self.video_button = ttk.Button(
            button_frame,
            text="Analyze Video",
            command=self.select_video,
            style="Normal.TButton"
        )

        self.video_button.pack(
            side="left",
            padx=5
        )

        self.clear_button = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_results,
            style="Normal.TButton"
        )

        self.clear_button.pack(
            side="right",
            padx=5
        )


    # ==================================================================
    # STATUS BAR
    # ==================================================================

    def build_status_bar(self):

        status_frame = ttk.Frame(
            self.root
        )

        status_frame.pack(
            fill="x",
            side="bottom"
        )

        ttk.Separator(
            status_frame,
            orient="horizontal"
        ).pack(
            fill="x"
        )

        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
            padding=(
                10,
                5
            )
        )

        status_label.pack(
            fill="x"
        )


    # ==================================================================
    # LOAD MODELS
    # ==================================================================

    def load_trained_models(self):

        try:

            self.root.update_idletasks()

            self.models = load_models()

            self.status_var.set(
                "Models loaded successfully. "
                "Select an image to begin."
            )

        except Exception as error:

            self.status_var.set(
                "Failed to load trained models."
            )

            messagebox.showerror(
                "Model Loading Error",
                (
                    "Could not load the trained models.\n\n"
                    f"{type(error).__name__}: {error}"
                )
            )


    # ==================================================================
    # SELECT IMAGE
    # ==================================================================

    def select_image(self):

        file_path = filedialog.askopenfilename(
            title="Select Fruit Image",
            initialdir=str(
                PROJECT_ROOT
            ),
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                ),
                (
                    "JPEG Files",
                    "*.jpg *.jpeg"
                ),
                (
                    "PNG Files",
                    "*.png"
                ),
                (
                    "All Files",
                    "*.*"
                ),
            ]
        )

        if not file_path:
            return

        self.selected_image_path = Path(
            file_path
        )

        self.file_var.set(
            self.selected_image_path.name
        )

        self.clear_result_values()

        try:

            image = cv2.imread(
                str(
                    self.selected_image_path
                )
            )

            if image is None:

                raise ValueError(
                    "OpenCV could not read "
                    "the selected image."
                )

            self.display_cv_image(
                image,
                self.original_image_label,
                PREVIEW_WIDTH,
                PREVIEW_HEIGHT,
                "original"
            )

            self.analyse_button.config(
                state="normal"
            )

            self.status_var.set(
                "Image selected. "
                "Click Analyze Image."
            )

        except Exception as error:

            messagebox.showerror(
                "Image Error",
                str(error)
            )


    # ==================================================================
    # ANALYSE IMAGE
    # ==================================================================

    def analyse_selected_image(self):

        if self.selected_image_path is None:

            messagebox.showwarning(
                "No Image",
                "Please select an image first."
            )

            return

        if self.models is None:

            messagebox.showwarning(
                "Models Not Ready",
                "The trained models are not loaded yet."
            )

            return

        try:

            # ----------------------------------------------------------
            # DISABLE BUTTON DURING PROCESSING
            # ----------------------------------------------------------

            self.analyse_button.config(
                state="disabled"
            )

            self.select_button.config(
                state="disabled"
            )

            self.status_var.set(
                "Analyzing image..."
            )

            self.root.update()

            # ----------------------------------------------------------
            # COMPLETE PIPELINE
            # ----------------------------------------------------------

            result = predict_image(
                self.selected_image_path,
                models=self.models,
                save_results=False
            )

            # ----------------------------------------------------------
            # DISPLAY RESULTS
            # ----------------------------------------------------------

            self.display_prediction(
                result
            )

            # ----------------------------------------------------------
            # DISPLAY MASKS
            # ----------------------------------------------------------

            self.display_masks(
                result
            )

            self.status_var.set(
                "Analysis completed successfully."
            )

        except Exception as error:

            self.status_var.set(
                "Analysis failed."
            )

            messagebox.showerror(
                "Prediction Error",
                (
                    "The image could not be analyzed.\n\n"
                    f"{type(error).__name__}: {error}"
                )
            )

        finally:

            self.select_button.config(
                state="normal"
            )

            self.analyse_button.config(
                state="normal"
            )


    # ==================================================================
    # DISPLAY PREDICTION
    # ==================================================================

    def display_prediction(
        self,
        result
    ):

        fruit = str(
            result.get(
                "fruit",
                "-"
            )
        )

        fruit_confidence = float(
            result.get(
                "fruit_confidence",
                0.0
            )
        )

        condition = str(
            result.get(
                "condition",
                "-"
            )
        )

        condition_confidence = float(
            result.get(
                "condition_confidence",
                0.0
            )
        )

        condition_type = str(
            result.get(
                "condition_type",
                "Ripeness"
            )
        )

        # --------------------------------------------------------------
        # UNCERTAINTY HANDLING
        # --------------------------------------------------------------

        if (
            fruit_confidence
            < FRUIT_CONFIDENCE_THRESHOLD
        ):

            self.fruit_var.set(
                "Uncertain"
            )

            self.fruit_confidence_var.set(
                f"{fruit_confidence:.2f}%"
            )

            self.condition_title_var.set(
                "Ripeness"
            )

            self.condition_var.set(
                "Not Assessed"
            )

            self.condition_confidence_var.set(
                "-"
            )

        else:

            self.fruit_var.set(
                fruit
            )

            self.fruit_confidence_var.set(
                f"{fruit_confidence:.2f}%"
            )

            # Guava may use quality instead of ripeness.
            self.condition_title_var.set(
                condition_type
            )

            self.condition_var.set(
                condition
            )

            self.condition_confidence_var.set(
                f"{condition_confidence:.2f}%"
            )

        # --------------------------------------------------------------
        # DAMAGE
        # --------------------------------------------------------------

        fruit_pixels = int(
            result.get(
                "fruit_pixels",
                0
            )
        )

        blemish_pixels = int(
            result.get(
                "blemish_pixels",
                0
            )
        )

        raw_damage = float(
            result.get(
                "raw_damage_percentage",
                0.0
            )
        )

        awdp_damage = float(
            result.get(
                "damage_percentage",
                0.0
            )
        )

        damage_level = str(
            result.get(
                "damage_level",
                "-"
            )
        )

        self.fruit_area_var.set(
            f"{fruit_pixels:,} pixels"
        )

        self.blemish_area_var.set(
            f"{blemish_pixels:,} pixels"
        )

        self.raw_damage_var.set(
            f"{raw_damage:.2f}%"
        )

        self.awdp_damage_var.set(
            f"{awdp_damage:.2f}%"
        )

        self.damage_level_var.set(
            damage_level
        )


    # ==================================================================
    # DISPLAY MASKS
    # ==================================================================

    def display_masks(
        self,
        result
    ):

        # Different versions of prediction_pipeline.py may use
        # slightly different dictionary keys. Try the common ones.

        roi_mask = result.get(
            "roi_mask"
        )

        blemish_mask = result.get(
            "blemish_mask"
        )

        # --------------------------------------------------------------
        # ROI
        # --------------------------------------------------------------

        if roi_mask is not None:

            self.display_cv_image(
                roi_mask,
                self.roi_label,
                MASK_WIDTH,
                MASK_HEIGHT,
                "roi"
            )

        else:

            self.roi_label.config(
                image="",
                text=(
                    "ROI generated\n"
                    "during processing"
                )
            )

            self.roi_photo = None

        # --------------------------------------------------------------
        # BLEMISH
        # --------------------------------------------------------------

        if blemish_mask is not None:

            self.display_cv_image(
                blemish_mask,
                self.blemish_label,
                MASK_WIDTH,
                MASK_HEIGHT,
                "blemish"
            )

        else:

            self.blemish_label.config(
                image="",
                text=(
                    "Blemish mask generated\n"
                    "during processing"
                )
            )

            self.blemish_photo = None


    # ==================================================================
    # DISPLAY OPENCV IMAGE
    # ==================================================================

    def display_cv_image(
        self,
        cv_image,
        label,
        max_width,
        max_height,
        image_type
    ):

        if cv_image is None:
            return

        image = cv_image.copy()

        # ==============================================================
        # CONVERT OPENCV IMAGE TO RGB
        # ==============================================================

        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        elif image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGBA
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        height, width = image.shape[:2]

        # ==============================================================
        # RESIZE IMAGE WHILE PRESERVING ASPECT RATIO
        # ==============================================================

        scale = min(
            max_width / width,
            max_height / height
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_CUBIC
        )

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation
        )

        # ==============================================================
        # CREATE FIXED-SIZE PREVIEW CANVAS
        # ==============================================================

        # White background
        canvas = Image.new(
            "RGB",
            (max_width, max_height),
            "white"
        )

        pil_image = Image.fromarray(
            resized
        )

        # Centre the image
        x = (
            max_width - new_width
        ) // 2

        y = (
            max_height - new_height
        ) // 2

        canvas.paste(
            pil_image,
            (x, y)
        )

        # ==============================================================
        # DISPLAY
        # ==============================================================

        photo = ImageTk.PhotoImage(
            canvas
        )

        label.config(
            image=photo,
            text="",
            width=max_width,
            height=max_height
        )

        # Keep reference so Tkinter does not delete image
        if image_type == "original":

            self.original_photo = photo

        elif image_type == "roi":

            self.roi_photo = photo

        elif image_type == "blemish":

            self.blemish_photo = photo


    # ==================================================================
    # VIDEO
    # ==================================================================

    def select_video(self):
        """
        Open a small input-source dialog.

        The existing Analyze Video button still calls this method, but the
        user can now choose either a recorded video file or a live webcam.
        """

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Video Input")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        dialog_width = 400
        dialog_height = 250

        self.root.update_idletasks()

        x = (
            self.root.winfo_rootx()
            + max(
                0,
                (
                    self.root.winfo_width()
                    - dialog_width
                ) // 2
            )
        )

        y = (
            self.root.winfo_rooty()
            + max(
                0,
                (
                    self.root.winfo_height()
                    - dialog_height
                ) // 2
            )
        )

        dialog.geometry(
            f"{dialog_width}x{dialog_height}+{x}+{y}"
        )

        ttk.Label(
            dialog,
            text="Select Input Source",
            style="Section.TLabel"
        ).pack(
            pady=(25, 8)
        )

        ttk.Label(
            dialog,
            text=(
                "Choose a recorded video "
                "or use a live webcam."
            )
        ).pack(
            pady=(0, 18)
        )

        ttk.Button(
            dialog,
            text="Open Video File",
            command=lambda: self.open_video_file(
                dialog
            ),
            style="Primary.TButton"
        ).pack(
            fill="x",
            padx=65,
            pady=5
        )

        ttk.Button(
            dialog,
            text="Open Webcam",
            command=lambda: self.open_webcam(
                dialog
            ),
            style="Normal.TButton"
        ).pack(
            fill="x",
            padx=65,
            pady=5
        )

        ttk.Button(
            dialog,
            text="Cancel",
            command=dialog.destroy,
            style="Normal.TButton"
        ).pack(
            fill="x",
            padx=65,
            pady=(5, 15)
        )


    def open_video_file(
        self,
        dialog=None
    ):
        """Select a recorded video and launch video_processing.py."""

        if dialog is not None:
            dialog.destroy()

        video_path = filedialog.askopenfilename(
            title="Select Fruit Video",
            initialdir=str(
                PROJECT_ROOT
                / "test_videos"
            ),
            filetypes=[
                (
                    "Video Files",
                    "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"
                ),
                (
                    "MP4 Files",
                    "*.mp4"
                ),
                (
                    "All Files",
                    "*.*"
                ),
            ]
        )

        if not video_path:
            self.status_var.set(
                "Video selection cancelled."
            )
            return

        self.launch_video_process(
            [
                str(video_path)
            ],
            (
                "Video analysis started. "
                "Press Q in the video window to stop."
            )
        )


    def open_webcam(
        self,
        dialog=None
    ):
        """Launch webcam mode with automatic camera detection."""

        if dialog is not None:
            dialog.destroy()

        continue_camera = messagebox.askyesno(
            "Open Webcam",
            (
                "The system will search for an "
                "available webcam.\n\n"
                "Press Q in the webcam window "
                "to stop.\n\n"
                "Continue?"
            )
        )

        if not continue_camera:
            self.status_var.set(
                "Webcam analysis cancelled."
            )
            return

        self.launch_video_process(
            [
                "--camera",
                "auto"
            ],
            (
                "Webcam analysis started. "
                "Press Q in the webcam window to stop."
            )
        )


    def launch_video_process(
        self,
        arguments,
        success_status
    ):
        """
        Start video_processing.py in a separate worker thread.

        video_processing.py creates its own OpenCV display window.
        """

        video_script = (
            SYSTEM_DIR
            / "video_processing.py"
        )

        if not video_script.exists():

            messagebox.showerror(
                "Video Processing Error",
                (
                    "video_processing.py "
                    "was not found:\n\n"
                    f"{video_script}"
                )
            )

            return

        self.status_var.set(
            "Starting video analysis..."
        )

        def worker():

            try:

                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(video_script),
                        *arguments
                    ],
                    cwd=str(
                        PROJECT_ROOT
                    )
                )

                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        success_status
                    )
                )

                return_code = process.wait()

                if return_code == 0:

                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            "Video/webcam analysis finished."
                        )
                    )

                else:

                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            "Video/webcam analysis ended "
                            "with an error."
                        )
                    )

            except Exception as error:

                error_message = (
                    "Could not start video analysis.\n\n"
                    f"{type(error).__name__}: {error}"
                )

                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Video Error",
                        error_message
                    )
                )

                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        "Video analysis failed."
                    )
                )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()


    # ==================================================================
    # CLEAR RESULT VALUES
    # ==================================================================

    def clear_result_values(self):

        self.fruit_var.set(
            "-"
        )

        self.fruit_confidence_var.set(
            "-"
        )

        self.condition_title_var.set(
            "Ripeness"
        )

        self.condition_var.set(
            "-"
        )

        self.condition_confidence_var.set(
            "-"
        )

        self.fruit_area_var.set(
            "-"
        )

        self.blemish_area_var.set(
            "-"
        )

        self.raw_damage_var.set(
            "-"
        )

        self.awdp_damage_var.set(
            "-"
        )

        self.damage_level_var.set(
            "-"
        )

        self.roi_label.config(
            image="",
            text="ROI mask"
        )

        self.blemish_label.config(
            image="",
            text="Blemish mask"
        )

        self.roi_photo = None
        self.blemish_photo = None


    # ==================================================================
    # CLEAR EVERYTHING
    # ==================================================================

    def clear_results(self):

        self.selected_image_path = None

        self.file_var.set(
            "No image selected"
        )

        self.original_image_label.config(
            image="",
            text=(
                "Select an image "
                "to begin"
            )
        )

        self.original_photo = None

        self.clear_result_values()

        self.analyse_button.config(
            state="disabled"
        )

        self.status_var.set(
            "Ready. Select an image to begin."
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    root = tk.Tk()

    app = FruitQualityGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()