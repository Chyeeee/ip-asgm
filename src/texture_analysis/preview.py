import matplotlib.pyplot as plt
import numpy as np

from .config import (
    BANANA_DATASET_DIR,
    CLASS_NAMES,
    GLCM_PROPERTIES,
    OUTPUT_DIR,
)

from .data_loader import (
    load_dataset,
)

from .glcm_features import (
    create_local_glcm_contrast_map,
    extract_glcm_from_image,
)

from .lbp_features import (
    extract_lbp_from_image,
)

from .preprocessing import (
    create_masked_gray_visual,
    create_masked_roi_visual,
    preprocess_image,
)


# ============================================================
# SELECT SAMPLE IMAGE
# ============================================================

def _choose_one_sample_per_class(
    records,
):
    """
    Automatically choose one image from:

    Class_A
    Class_B
    Defect

    for feature visualization.
    """

    samples = {}

    for class_name in (
        CLASS_NAMES
    ):

        samples[
            class_name
        ] = next(
            record
            for record in records
            if record.label
            == class_name
        )

    return samples


# ============================================================
# OPENCV BGR -> MATPLOTLIB RGB
# ============================================================

def _bgr_to_rgb(
    image,
):
    return image[
        :,
        :,
        ::-1
    ]


# ============================================================
# GLCM VISUALIZATION
# ============================================================

def save_glcm_preview(
    class_name,
    processed,
    glcm_features,
    glcm_matrix,
    local_contrast_map,
    output_dir,
):
    """
    Visualize the complete GLCM feature extraction process:

    Original
        ↓
    Banana ROI
        ↓
    Grayscale ROI
        ↓
    Local GLCM Contrast Map
        ↓
    Global GLCM Heatmap
        ↓
    Six GLCM feature values
    """

    roi_visual = (
        create_masked_roi_visual(
            processed.roi_bgr,
            processed.mask,
        )
    )

    gray_visual = (
        create_masked_gray_visual(
            processed.gray,
            processed.mask,
        )
    )

    # --------------------------------------------------------
    # Create 5 panels
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(
            21,
            5,
        ),
    )

    # ========================================================
    # 1. ORIGINAL
    # ========================================================

    axes[0].imshow(
        _bgr_to_rgb(
            processed.original
        )
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].axis(
        "off"
    )

    # ========================================================
    # 2. DETECTED BANANA ROI
    # ========================================================

    axes[1].imshow(
        _bgr_to_rgb(
            roi_visual
        )
    )

    axes[1].set_title(
        "Detected Banana ROI"
    )

    axes[1].axis(
        "off"
    )

    # ========================================================
    # 3. GRAYSCALE ROI
    # ========================================================

    axes[2].imshow(
        gray_visual,
        cmap="gray",
    )

    axes[2].set_title(
        "Grayscale ROI"
    )

    axes[2].axis(
        "off"
    )

    # ========================================================
    # 4. LOCAL GLCM CONTRAST MAP
    # ========================================================

    contrast_map_display = (
        axes[3].imshow(
            local_contrast_map,
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
    )

    axes[3].set_title(
        "Local GLCM Contrast Map"
    )

    axes[3].axis(
        "off"
    )

    fig.colorbar(
        contrast_map_display,
        ax=axes[3],
        fraction=0.046,
        pad=0.04,
    )

    # ========================================================
    # 5. GLOBAL GLCM HEATMAP
    # ========================================================

    # log1p makes low-frequency relationships easier to see.
    heatmap_display = (
        axes[4].imshow(
            np.log1p(
                glcm_matrix
            ),
            cmap="viridis",
            aspect="auto",
        )
    )

    axes[4].set_title(
        "GLCM Heatmap"
    )

    axes[4].set_xlabel(
        "Neighbour Gray Level"
    )

    axes[4].set_ylabel(
        "Reference Gray Level"
    )

    fig.colorbar(
        heatmap_display,
        ax=axes[4],
        fraction=0.046,
        pad=0.04,
    )

    # ========================================================
    # DISPLAY GLCM FEATURE VALUES
    # ========================================================

    feature_text = "\n".join(
        f"{name}: {value:.4f}"
        for name, value in zip(
            GLCM_PROPERTIES,
            glcm_features,
        )
    )

    fig.text(
        0.5,
        0.01,
        feature_text,
        ha="center",
        va="bottom",
        family="monospace",
        fontsize=10,
    )

    fig.suptitle(
        f"{class_name} - "
        f"GLCM Surface Texture Analysis",
        fontsize=14,
    )

    # Leave room at bottom for numerical features.
    fig.tight_layout(
        rect=[
            0,
            0.18,
            1,
            0.93,
        ]
    )

    output_path = (
        output_dir
        / (
            f"{class_name}"
            "_GLCM_analysis.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# LBP VISUALIZATION
# ============================================================

def save_lbp_preview(
    class_name,
    processed,
    lbp_histogram,
    lbp_image,
    output_dir,
):
    """
    Visualize the LBP feature extraction process:

    Original
        ↓
    Banana ROI
        ↓
    Grayscale ROI
        ↓
    LBP Texture Map
        ↓
    LBP Histogram
    """

    roi_visual = (
        create_masked_roi_visual(
            processed.roi_bgr,
            processed.mask,
        )
    )

    gray_visual = (
        create_masked_gray_visual(
            processed.gray,
            processed.mask,
        )
    )

    # --------------------------------------------------------
    # Hide background in LBP texture map
    # --------------------------------------------------------

    lbp_visual = np.full_like(
        lbp_image,
        np.nan,
        dtype=np.float64,
    )

    banana_pixels = (
        processed.mask > 0
    )

    lbp_visual[
        banana_pixels
    ] = lbp_image[
        banana_pixels
    ]

    # --------------------------------------------------------
    # Create 5 panels
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(
            20,
            4.8,
        ),
    )

    # ========================================================
    # 1. ORIGINAL
    # ========================================================

    axes[0].imshow(
        _bgr_to_rgb(
            processed.original
        )
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].axis(
        "off"
    )

    # ========================================================
    # 2. DETECTED BANANA ROI
    # ========================================================

    axes[1].imshow(
        _bgr_to_rgb(
            roi_visual
        )
    )

    axes[1].set_title(
        "Detected Banana ROI"
    )

    axes[1].axis(
        "off"
    )

    # ========================================================
    # 3. GRAYSCALE ROI
    # ========================================================

    axes[2].imshow(
        gray_visual,
        cmap="gray",
    )

    axes[2].set_title(
        "Grayscale ROI"
    )

    axes[2].axis(
        "off"
    )

    # ========================================================
    # 4. LBP TEXTURE MAP
    # ========================================================

    axes[3].imshow(
        lbp_visual,
        cmap="gray",
    )

    axes[3].set_title(
        "LBP Texture Map"
    )

    axes[3].axis(
        "off"
    )

    # ========================================================
    # 5. LBP HISTOGRAM
    # ========================================================

    axes[4].bar(
        np.arange(
            len(
                lbp_histogram
            )
        ),
        lbp_histogram,
    )

    axes[4].set_title(
        "Normalized LBP Histogram"
    )

    axes[4].set_xlabel(
        "LBP Bin"
    )

    axes[4].set_ylabel(
        "Frequency"
    )

    fig.suptitle(
        f"{class_name} - "
        f"LBP Surface Texture Analysis",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )

    output_path = (
        output_dir
        / (
            f"{class_name}"
            "_LBP_analysis.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# MAIN PREVIEW PROGRAM
# ============================================================

def main():
    """
    Generate visual examples for:

    Class_A
    Class_B
    Defect
    """

    # --------------------------------------------------------
    # Load banana dataset
    # --------------------------------------------------------

    records = load_dataset(
        BANANA_DATASET_DIR
    )

    # --------------------------------------------------------
    # Choose one sample from each class
    # --------------------------------------------------------

    samples = (
        _choose_one_sample_per_class(
            records
        )
    )

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    preview_dir = (
        OUTPUT_DIR
        / "feature_preview"
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 60
    )

    print(
        "BANANA TEXTURE FEATURE VISUALIZATION"
    )

    print(
        "=" * 60
    )

    saved_files = []

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_name in (
        CLASS_NAMES
    ):

        record = samples[
            class_name
        ]

        # ----------------------------------------------------
        # Preprocessing / banana ROI
        # ----------------------------------------------------

        processed = (
            preprocess_image(
                record.path
            )
        )

        # ----------------------------------------------------
        # GLOBAL GLCM FEATURES + HEATMAP
        # ----------------------------------------------------

        (
            glcm_features,
            glcm_matrix,
        ) = (
            extract_glcm_from_image(
                processed.gray,
                processed.mask,
            )
        )

        # ----------------------------------------------------
        # LOCAL GLCM TEXTURE MAP
        # ----------------------------------------------------

        local_contrast_map = (
            create_local_glcm_contrast_map(
                processed.gray,
                processed.mask,
            )
        )

        # ----------------------------------------------------
        # LBP FEATURES + TEXTURE MAP
        # ----------------------------------------------------

        (
            lbp_histogram,
            lbp_image,
        ) = (
            extract_lbp_from_image(
                processed.gray,
                processed.mask,
            )
        )

        # ----------------------------------------------------
        # SAVE GLCM VISUALIZATION
        # ----------------------------------------------------

        glcm_file = (
            save_glcm_preview(
                class_name,
                processed,
                glcm_features,
                glcm_matrix,
                local_contrast_map,
                preview_dir,
            )
        )

        # ----------------------------------------------------
        # SAVE LBP VISUALIZATION
        # ----------------------------------------------------

        lbp_file = (
            save_lbp_preview(
                class_name,
                processed,
                lbp_histogram,
                lbp_image,
                preview_dir,
            )
        )

        saved_files.extend(
            [
                glcm_file,
                lbp_file,
            ]
        )

        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print(
            f"\n{class_name}"
        )

        print(
            f"Sample: "
            f"{record.path.name}"
        )

        print(
            "\nGLCM Features:"
        )

        for name, value in zip(
            GLCM_PROPERTIES,
            glcm_features,
        ):

            print(
                f"  "
                f"{name:<15}: "
                f"{value:.4f}"
            )

        print(
            "\nLBP Features:"
        )

        print(
            f"  Histogram bins : "
            f"{len(lbp_histogram)}"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VISUALIZATIONS SAVED"
    )

    print(
        "=" * 60
    )

    for path in (
        saved_files
    ):

        print(
            f"- {path}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()