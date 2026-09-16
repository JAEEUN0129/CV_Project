"""Create a frame-aligned hair-and-bangs mask video for VACE.

Unlike a first-frame mask propagated by an object tracker, this mask is rebuilt
from the face and hair parsed in every frame. Missing detections are filled from
the nearest valid frame and a short temporal median removes one-frame flicker.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .models import HumanParser


def _normalise_labels(labels: dict[int, str]) -> dict[str, int]:
    return {name.lower().replace(" ", "_"): class_id for class_id, name in labels.items()}


def _frame_mask(
    class_map: np.ndarray,
    labels: dict[int, str],
    forehead_ratio: float,
    bangs_style: str = "straight",
    side_short_ratio: float = 0.15,
    side_long_ratio: float = 0.45,
    side_direction: str = "right",
    edit_type: str = "bangs",
) -> np.ndarray | None:
    ids = _normalise_labels(labels)
    if "hair" not in ids or "face" not in ids:
        raise ValueError("Human parser must provide both hair and face classes")

    hair = class_map == ids["hair"]
    face = class_map == ids["face"]
    ys, xs = np.where(face)
    if not hair.any() or not len(ys):
        return None

    if edit_type == "short_hair":
        return hair
    if edit_type == "wave":
        expanded = cv2.dilate(hair.astype(np.uint8), np.ones((31, 61), np.uint8)) > 0
        return hair | (expanded & ~face)

    face_top, face_bottom = int(ys.min()), int(ys.max())
    rows, columns = np.indices(face.shape)
    if bangs_style == "side":
        face_left, face_right = int(xs.min()), int(xs.max())
        normalised_x = np.clip(
            (columns - face_left) / max(1, face_right - face_left), 0.0, 1.0
        )
        if side_direction == "left":
            normalised_x = 1.0 - normalised_x
        depth = side_short_ratio + (side_long_ratio - side_short_ratio) * normalised_x
        forehead_bottom = face_top + (face_bottom - face_top + 1) * depth
        forehead = face & (rows <= forehead_bottom)
    else:
        forehead_bottom = face_top + round((face_bottom - face_top + 1) * forehead_ratio)
        forehead = face & (rows <= forehead_bottom)
    mask = hair | forehead

    # Join small gaps between the parsed hairline and the forehead band.
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def _fill_missing(masks: list[np.ndarray | None]) -> list[np.ndarray]:
    valid = [index for index, mask in enumerate(masks) if mask is not None]
    if not valid:
        raise ValueError("No frame contained both a detectable face and hair")
    return [
        mask if mask is not None else masks[min(valid, key=lambda valid_index: abs(valid_index - index))]
        for index, mask in enumerate(masks)
    ]


def _temporal_median(masks: list[np.ndarray], window: int) -> list[np.ndarray]:
    radius = window // 2
    smoothed = []
    for index in range(len(masks)):
        start = max(0, index - radius)
        stop = min(len(masks), index + radius + 1)
        votes = np.stack(masks[start:stop]).sum(axis=0)
        smoothed.append(votes >= ((stop - start) // 2 + 1))
    return smoothed


def build_mask_video(
    source: Path,
    output: Path,
    masked_video_output: Path | None = None,
    forehead_ratio: float = 0.28,
    temporal_window: int = 3,
    bangs_style: str = "straight",
    side_short_ratio: float = 0.15,
    side_long_ratio: float = 0.45,
    side_direction: str = "right",
    edit_type: str = "bangs",
) -> Path:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    parser = HumanParser()
    masks: list[np.ndarray | None] = []

    with tempfile.TemporaryDirectory(prefix="vace-bangs-") as temp_dir:
        frame_path = Path(temp_dir) / "frame.png"
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(frame_path), frame):
                raise ValueError("Could not write a temporary video frame")
            class_map, labels = parser.predict(frame_path)
            masks.append(
                _frame_mask(
                    class_map,
                    labels,
                    forehead_ratio,
                    bangs_style,
                    side_short_ratio,
                    side_long_ratio,
                    side_direction,
                    edit_type,
                )
            )
    capture.release()

    if not masks:
        raise ValueError(f"Video contains no readable frames: {source}")
    complete_masks = _fill_missing(masks)
    complete_masks = _temporal_median(complete_masks, temporal_window)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create mask video {output}")
    for mask in complete_masks:
        frame = np.where(mask, 255, 0).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"Wrote {len(complete_masks)} mask frames at {fps:.3f} fps to {output}")

    if masked_video_output is not None:
        masked_video_output.parent.mkdir(parents=True, exist_ok=True)
        source_capture = cv2.VideoCapture(str(source))
        masked_writer = cv2.VideoWriter(
            str(masked_video_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
            True,
        )
        if not source_capture.isOpened() or not masked_writer.isOpened():
            source_capture.release()
            masked_writer.release()
            raise ValueError(f"Could not create masked source video {masked_video_output}")
        written = 0
        for mask in complete_masks:
            ok, frame = source_capture.read()
            if not ok:
                break
            frame[mask] = 128
            masked_writer.write(frame)
            written += 1
        source_capture.release()
        masked_writer.release()
        if written != len(complete_masks):
            raise ValueError(
                f"Masked source contains {written} frames, expected {len(complete_masks)}"
            )
        print(f"Wrote {written} masked source frames to {masked_video_output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masked-video-output", type=Path)
    parser.add_argument("--bangs-style", choices=("straight", "choppy", "side"), default="straight")
    parser.add_argument("--forehead-ratio", type=float, default=0.28)
    parser.add_argument("--side-short-ratio", type=float, default=0.15)
    parser.add_argument("--side-long-ratio", type=float, default=0.45)
    parser.add_argument("--side-direction", choices=("left", "right"), default="right")
    parser.add_argument(
        "--edit-type",
        choices=("bangs", "see_through_bangs", "short_hair", "remove_bangs", "wave"),
        default="bangs",
    )
    parser.add_argument("--temporal-window", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.forehead_ratio <= 0.5:
        parser.error("--forehead-ratio must be between 0 and 0.5")
    if not 0 <= args.side_short_ratio <= args.side_long_ratio <= 0.5:
        parser.error("side ratios must satisfy 0 <= short <= long <= 0.5")
    if args.temporal_window < 1 or args.temporal_window % 2 == 0:
        parser.error("--temporal-window must be a positive odd number")
    build_mask_video(
        args.source,
        args.output,
        args.masked_video_output,
        args.forehead_ratio,
        args.temporal_window,
        args.bangs_style,
        args.side_short_ratio,
        args.side_long_ratio,
        args.side_direction,
        args.edit_type,
    )


if __name__ == "__main__":
    main()
