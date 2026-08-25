import cv2
import numpy as np


def calculate_metrics(predicted_mask, ground_truth_mask):
    """
    Compare a predicted blemish mask with the manually
    annotated ground-truth mask.

    White (255) = blemish
    Black (0) = non-blemish
    """

    # Make sure both masks are binary
    predicted = predicted_mask > 0
    ground_truth = ground_truth_mask > 0

    # True Positive:
    # Algorithm says blemish AND ground truth says blemish
    tp = np.logical_and(predicted, ground_truth).sum()

    # False Positive:
    # Algorithm says blemish but ground truth says no
    fp = np.logical_and(
        predicted,
        np.logical_not(ground_truth)
    ).sum()

    # False Negative:
    # Ground truth says blemish but algorithm missed it
    fn = np.logical_and(
        np.logical_not(predicted),
        ground_truth
    ).sum()

    # IoU
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # Dice coefficient
    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0
    )

    # Precision
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    # Recall
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall
    }