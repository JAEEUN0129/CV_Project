"""Select and export a sharp anchor-frame candidate from a video."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class AnchorFrame:
    frame_index: int
    timestamp_seconds: float
    fps: float
    sharpness: float
    width: int
    height: int
    face_bbox: tuple[int, int, int, int] | None


def select_anchor(source: Path, output: Path, sample_every: int = 1) -> AnchorFrame:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    try:
        str(cascade_path).encode("ascii")
        cascade_path_is_safe = True
    except UnicodeEncodeError:
        # OpenCV's Windows file storage cannot reliably open models through a Unicode path.
        cascade_path_is_safe = False
    detector = (
        cv2.CascadeClassifier(str(cascade_path))
        if cascade_path.is_file() and cascade_path_is_safe
        else None
    )
    if detector is not None and detector.empty():
        detector = None
    best_face: tuple[float, int, object, tuple[int, int, int, int]] | None = None
    best_fallback: tuple[float, int, object] | None = None
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fallback_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if best_fallback is None or fallback_score > best_fallback[0]:
                best_fallback = fallback_score, index, frame.copy()
            faces = (
                detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if detector is not None
                else []
            )
            if len(faces):
                x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
                face_gray = gray[y : y + height, x : x + width]
                face_score = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
                if best_face is None or face_score > best_face[0]:
                    best_face = face_score, index, frame.copy(), (int(x), int(y), int(width), int(height))
        index += 1
    capture.release()
    if best_face is not None:
        sharpness, frame_index, frame, face_bbox = best_face
    elif best_fallback is not None:
        sharpness, frame_index, frame = best_fallback
        face_bbox = None
    else:
        raise ValueError(f"Video contains no readable frames: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise ValueError(f"Could not save anchor frame to {output}")
    height, width = frame.shape[:2]
    return AnchorFrame(frame_index, frame_index / fps, fps, sharpness, width, height, face_bbox)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--sample-every", type=int, default=1)
    args = parser.parse_args()
    if args.sample_every < 1:
        parser.error("--sample-every must be at least 1")
    selected = select_anchor(args.source, args.output, args.sample_every)
    payload = asdict(selected)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
