"""Build a VACE edit mask from a source-video hair mask.

The bangs variant adds only the upper forehead from the human-parser face mask.
The wavy variant expands the editable region sideways and downwards while
protecting parsed face and arm pixels. This gives long waves room around their
outer silhouette without uniformly expanding the mask into the face.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import HumanParser


def first_frame(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read the first frame of {video}")
    return frame


def build_mask(
    source: Path,
    base_mask: Path,
    output: Path,
    add_bangs: bool = False,
    add_wavy_space: bool = False,
    forehead_ratio: float = 0.28,
    side_pixels: int = 40,
    down_pixels: int = 30,
) -> Path:
    frame = first_frame(source)
    height, width = frame.shape[:2]
    mask = cv2.imread(str(base_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask {base_mask}")
    if mask.shape != (height, width):
        raise ValueError(
            f"Mask size {mask.shape[1]}x{mask.shape[0]} does not match "
            f"video size {width}x{height}"
        )
    result = mask >= 128

    if add_bangs or add_wavy_space:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame_path = output.parent / "vace-mask-first-frame.png"
        cv2.imwrite(str(frame_path), frame)
        class_map, labels = HumanParser().predict(frame_path)
        ids = {name.replace(" ", "_"): class_id for class_id, name in labels.items()}

    if add_bangs:
        if "face" not in ids:
            raise ValueError("Human parser does not provide a face class")
        face = class_map == ids["face"]
        ys, xs = np.where(face)
        if not len(ys):
            raise ValueError("No face was detected in the first video frame")

        face_top, face_bottom = int(ys.min()), int(ys.max())
        forehead_bottom = face_top + round((face_bottom - face_top + 1) * forehead_ratio)
        forehead = face & (np.indices(face.shape)[0] <= forehead_bottom)
        result |= forehead

    if add_wavy_space:
        base = result.copy()
        horizontal_kernel = np.ones((1, side_pixels * 2 + 1), dtype=np.uint8)
        expanded = cv2.dilate(base.astype(np.uint8), horizontal_kernel) > 0
        if down_pixels:
            downward_kernel = np.ones((down_pixels + 1, 1), dtype=np.uint8)
            expanded = cv2.dilate(
                expanded.astype(np.uint8),
                downward_kernel,
                anchor=(0, down_pixels),
            ) > 0

        protected_names = ("face", "left_arm", "right_arm")
        protected = np.zeros_like(base)
        for name in protected_names:
            if name in ids:
                protected |= class_map == ids[name]
        result = base | (expanded & ~protected)

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((result.astype(np.uint8) * 255)).save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bangs", action="store_true")
    parser.add_argument("--wavy", action="store_true")
    parser.add_argument("--forehead-ratio", type=float, default=0.28)
    parser.add_argument("--side-pixels", type=int, default=40)
    parser.add_argument("--down-pixels", type=int, default=30)
    args = parser.parse_args()
    if not 0 <= args.forehead_ratio <= 0.5:
        parser.error("--forehead-ratio must be between 0 and 0.5")
    if args.side_pixels < 0 or args.down_pixels < 0:
        parser.error("--side-pixels and --down-pixels must be non-negative")
    print(
        build_mask(
            args.source,
            args.base_mask,
            args.output,
            add_bangs=args.bangs,
            add_wavy_space=args.wavy,
            forehead_ratio=args.forehead_ratio,
            side_pixels=args.side_pixels,
            down_pixels=args.down_pixels,
        )
    )


if __name__ == "__main__":
    main()
