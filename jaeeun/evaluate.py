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


# --- Temporal consistency, with motion taken out ------------------------------------------
#
# A plain difference between neighbouring frames scores a subject who turns their head as
# badly as hair that is boiling. The functions below first move the previous frame onto the
# current one with optical flow and measure what is left; what is left is flicker. Frames
# are BGR arrays as cv2.imread returns them, and every measurement is limited to the hair,
# because the untouched face and background would pull the average toward zero.


def optical_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 25, 3, 5, 1.2, 0,
    )


def warp_forward(previous: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Move ``previous`` along ``flow`` so it lines up with the next frame."""
    import cv2

    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    return cv2.remap(
        previous, (grid_x + flow[..., 0]).astype(np.float32), (grid_y + flow[..., 1]).astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def flow_aligned_flicker(frames: list[np.ndarray], hair_masks: list[np.ndarray]) -> float:
    """Mean per-pixel difference (0-255) left inside the hair after motion compensation.

    Lower is steadier. Note that this cannot judge ``video.smooth_frames``, which blends
    along the same optical flow and so lowers this number by construction.
    """
    values = []
    for previous, current, mask in zip(frames, frames[1:], hair_masks[1:]):
        mask = mask.astype(bool)
        if not mask.any():
            continue
        residual = np.abs(current.astype(np.float32) - warp_forward(previous, optical_flow(previous, current)).astype(np.float32))
        values.append(float(residual.mean(axis=2)[mask].mean()))
    return float(np.mean(values)) if values else float("nan")


def colour_stability(frames: list[np.ndarray], hair_masks: list[np.ndarray]) -> dict[str, float]:
    """How far the hair colour wanders over a clip, in CIE-Lab.

    ``lightness_std`` includes slow drift; ``step`` is the average jump between
    neighbouring frames, which is what reads as flicker.
    """
    import cv2

    means = []
    for frame, mask in zip(frames, hair_masks):
        mask = mask.astype(bool)
        if mask.any():
            means.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)[mask].mean(axis=0))
    if len(means) < 2:
        return {"lightness_std": float("nan"), "tint_std": float("nan"), "step": float("nan")}
    means = np.stack(means)
    return {
        "lightness_std": float(means[:, 0].std()),
        "tint_std": float(means[:, 1:].std(axis=0).mean()),
        "step": float(np.linalg.norm(np.diff(means, axis=0), axis=1).mean()),
    }


def frame_status_summary(output_dir) -> dict[str, float]:
    """Share of frames generated, interpolated and held, from the worker's frame_status.txt."""
    from pathlib import Path

    states = [line.split()[-1] for line in (Path(output_dir) / "frame_status.txt").read_text().splitlines() if line.strip()]
    total = len(states) or 1
    return {state: states.count(state) / total for state in ("detected", "interpolated", "held")}

