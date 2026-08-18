"""Segmentation and temporal-consistency metrics."""

import numpy as np


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(prediction, target).sum() / union)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def adjacent_mask_iou(masks: list[np.ndarray]) -> float:
    return 1.0 if len(masks) < 2 else float(np.mean([iou(a, b) for a, b in zip(masks, masks[1:])]))

