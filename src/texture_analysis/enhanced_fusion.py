from __future__ import annotations

import numpy as np

from .config import GLCM_ANGLES_DEG, GLCM_ENHANCED_DISTANCES, LBP_ENHANCED
from .glcm_features import extract_glcm_features
from .lbp_features import extract_lbp_features


def extract_enhanced_texture_features(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Extract multi-scale texture candidates for the PTD+ specialist.

    MEMBER-3 BOUNDARY:
      * processed image is used unchanged
      * existing ROI mask is used unchanged
      * no CLAHE / histogram equalisation
      * no Gaussian / median filtering
      * no thresholding / morphology / ROI repair
      * no new colour extraction

    Only texture representation is expanded; PTD+ later validates class-pair-specific texture tie-breakers.
    """
    glcm = extract_glcm_features(
        image_bgr,
        mask,
        distances=GLCM_ENHANCED_DISTANCES,
        angles=GLCM_ANGLES_DEG,
        prefix="enh_glcm",
        keep_per_distance=True,
        include_direction_std=True,
    )
    lbp = extract_lbp_features(
        image_bgr,
        mask,
        settings=LBP_ENHANCED,
        prefix="enh_lbp",
        include_stats=True,
    )
    return {**glcm, **lbp}
