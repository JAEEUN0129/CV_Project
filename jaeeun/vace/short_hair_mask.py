"""Build old-hair union target-anchor masks for long-to-short video edits."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..models import HumanParser
from ..prepare_vace_mask_video import _fill_missing, _temporal_median


def _parts(class_map: np.ndarray, labels: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    ids = {name.lower().replace(" ", "_"): class_id for class_id, name in labels.items()}
    if "hair" not in ids or "face" not in ids:
        raise ValueError("Human parser must provide both hair and face classes")
    return class_map == ids["hair"], class_map == ids["face"]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(ys):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _align_target_hair(
    target_hair: np.ndarray,
    target_face_box: tuple[int, int, int, int],
    frame_face_box: tuple[int, int, int, int],
) -> np.ndarray:
    tx1, ty1, tx2, ty2 = target_face_box
    fx1, fy1, fx2, fy2 = frame_face_box
    scale_x = (fx2 - fx1) / max(1, tx2 - tx1)
    scale_y = (fy2 - fy1) / max(1, ty2 - ty1)
    matrix = np.array(
        [[scale_x, 0.0, fx1 - tx1 * scale_x], [0.0, scale_y, fy1 - ty1 * scale_y]],
        dtype=np.float32,
    )
    height, width = target_hair.shape
    return cv2.warpAffine(
        target_hair.astype(np.uint8), matrix, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0


def build_short_hair_mask_video(
    source: Path,
    anchor: Path,
    output: Path,
    masked_video_output: Path,
    temporal_window: int = 3,
    target_output: Path | None = None,
    target_anchor_output: Path | None = None,
) -> Path:
    parser = HumanParser()
    anchor_map, anchor_labels = parser.predict(anchor)
    target_hair, target_face = _parts(anchor_map, anchor_labels)
    target_face_box = _bbox(target_face)
    if not target_hair.any() or target_face_box is None:
        raise ValueError("The short-hair anchor must contain detectable hair and face regions")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if target_hair.shape != (height, width):
        raise ValueError(
            f"Anchor size {target_hair.shape[1]}x{target_hair.shape[0]} does not match "
            f"video size {width}x{height}"
        )

    masks: list[np.ndarray | None] = []
    target_masks: list[np.ndarray | None] = []
    with tempfile.TemporaryDirectory(prefix="vace-short-") as temp_dir:
        frame_path = Path(temp_dir) / "frame.png"
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(frame_path), frame):
                raise ValueError("Could not write a temporary video frame")
            class_map, labels = parser.predict(frame_path)
            old_hair, frame_face = _parts(class_map, labels)
            frame_face_box = _bbox(frame_face)
            if not old_hair.any() or frame_face_box is None:
                masks.append(None)
                target_masks.append(None)
                continue
            new_hair = _align_target_hair(target_hair, target_face_box, frame_face_box)
            masks.append(old_hair | new_hair)
            target_masks.append(new_hair)
    capture.release()
    if not masks:
        raise ValueError(f"Video contains no readable frames: {source}")
    masks = _temporal_median(_fill_missing(masks), temporal_window)
    target_masks = _temporal_median(_fill_missing(target_masks), temporal_window)

    if target_anchor_output is not None:
        target_anchor_output.parent.mkdir(parents=True, exist_ok=True)
        anchor_mask = np.where(target_hair, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(target_anchor_output), anchor_mask):
            raise ValueError(f"Could not create target anchor mask {target_anchor_output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
    )
    source_capture = cv2.VideoCapture(str(source))
    masked_video_output.parent.mkdir(parents=True, exist_ok=True)
    masked_writer = cv2.VideoWriter(
        str(masked_video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
    )
    target_writer = None
    if target_output is not None:
        target_output.parent.mkdir(parents=True, exist_ok=True)
        target_writer = cv2.VideoWriter(
            str(target_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
        )
    if (
        not writer.isOpened()
        or not source_capture.isOpened()
        or not masked_writer.isOpened()
        or (target_writer is not None and not target_writer.isOpened())
    ):
        writer.release()
        source_capture.release()
        masked_writer.release()
        if target_writer is not None:
            target_writer.release()
        raise ValueError("Could not create short-hair mask outputs")
    written = 0
    for mask, target_mask in zip(masks, target_masks):
        ok, frame = source_capture.read()
        if not ok:
            break
        mask_frame = np.where(mask, 255, 0).astype(np.uint8)
        writer.write(cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR))
        if target_writer is not None:
            target_frame = np.where(target_mask, 255, 0).astype(np.uint8)
            target_writer.write(cv2.cvtColor(target_frame, cv2.COLOR_GRAY2BGR))
        frame[mask] = 128
        masked_writer.write(frame)
        written += 1
    writer.release()
    source_capture.release()
    masked_writer.release()
    if target_writer is not None:
        target_writer.release()
    if written != len(masks):
        raise ValueError(f"Wrote {written} frames, expected {len(masks)}")
    print(f"Wrote {written} old-hair union anchor-hair mask frames to {output}")
    if target_output is not None:
        print(f"Wrote {written} target short-hair silhouette frames to {target_output}")
    if target_anchor_output is not None:
        print(f"Wrote target short-hair anchor silhouette to {target_anchor_output}")
    print(f"Wrote {written} masked source frames to {masked_video_output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masked-video-output", type=Path, required=True)
    parser.add_argument("--target-output", type=Path)
    parser.add_argument("--target-anchor-output", type=Path)
    parser.add_argument("--temporal-window", type=int, default=3)
    args = parser.parse_args()
    if args.temporal_window < 1 or args.temporal_window % 2 == 0:
        parser.error("--temporal-window must be a positive odd number")
    build_short_hair_mask_video(
        args.source,
        args.anchor,
        args.output,
        args.masked_video_output,
        args.temporal_window,
        args.target_output,
        args.target_anchor_output,
    )


if __name__ == "__main__":
    main()
