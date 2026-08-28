from __future__ import annotations

import pandas as pd

META_COLUMNS = {
    "fruit",
    "category",
    "image",
    "relative_path",
    "colour_space",
    "feature_version",
    "processed_path",
    "mask_path",
}


def colour_feature_columns(df: pd.DataFrame) -> list[str]:
    """Member 2 colour features, read as-is and never recomputed by Member 3."""
    return [
        c
        for c in df.columns
        if c not in META_COLUMNS
        and not c.startswith("glcm_")
        and not c.startswith("lbp_")
        and not c.startswith("enh_")
        and not c.startswith("lg_")
        and not c.startswith("laws_")
        and not c.startswith("roi_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def glcm_feature_columns(df: pd.DataFrame, enhanced: bool = False) -> list[str]:
    prefix = "enh_glcm_" if enhanced else "glcm_"
    return [c for c in df.columns if c.startswith(prefix)]


def lbp_feature_columns(df: pd.DataFrame, enhanced: bool = False) -> list[str]:
    prefix = "enh_lbp_" if enhanced else "lbp_"
    return [c for c in df.columns if c.startswith(prefix)]


def local_global_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("lg_")]


def laws_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("laws_")]


def baseline_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """The three assignment-required baseline algorithms. v8 leaves these unchanged."""
    colour = colour_feature_columns(df)
    glcm = glcm_feature_columns(df, enhanced=False)
    lbp = lbp_feature_columns(df, enhanced=False)
    return {
        "GLCM": glcm,
        "LBP": lbp,
        "Colour_Texture_Fusion": colour + glcm + lbp,
    }


def proposed_texture_candidates(df: pd.DataFrame) -> list[str]:
    """Texture-only evidence available to the PTD+ specialist."""
    return (
        glcm_feature_columns(df, enhanced=True)
        + lbp_feature_columns(df, enhanced=True)
        + local_global_feature_columns(df)
        + laws_feature_columns(df)
    )
