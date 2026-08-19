"""Assemble a folder of images into an MP4.

The hairstyle worker names its outputs after the input frames and skips any frame where
no face was found, so the surviving files are neither consecutively numbered nor free of
gaps. ffmpeg needs an unbroken sequence, so the frames are renumbered into a scratch
folder first.

Usage:
    python jaeeun/frames_to_video.py --frames <dir> --output <file.mp4> --fps 7.5
"""

import argparse
import shutil
import sys
from pathlib import Path

# Run as `python jaeeun/frames_to_video.py`, the project root is not on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaeeun.video import frames_to_mp4  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# Written next to each result by the hairstyle worker; not frames of the video.
COMPANION_SUFFIXES = ("_aligned", "_mask")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True, help="Directory holding the frames")
    parser.add_argument("--output", type=Path, required=True, help="MP4 to write")
    parser.add_argument(
        "--fps",
        type=float,
        default=7.5,
        help="Rate the frames were sampled at. 30fps footage sampled every 4th frame is 7.5, "
        "not 8 — using the requested rate instead plays the result back too fast.",
    )
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="Also treat the worker's *_aligned.png and *_mask.png companion files as frames. "
        "They sit in the same folder as the results and are not part of the video.",
    )
    options = parser.parse_args()

    frames = sorted(
        path
        for path in options.frames.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
        and not (not options.include_extras and path.stem.endswith(COMPANION_SUFFIXES))
    )
    if not frames:
        raise SystemExit(f"이미지를 찾을 수 없습니다: {options.frames}")

    staging = options.frames.parent / f"{options.frames.name}_sequence"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for index, frame in enumerate(frames):
        shutil.copy(frame, staging / f"frame_{index:06d}{frame.suffix.lower()}")

    result = frames_to_mp4(staging, options.output, options.fps)
    print(f"{len(frames)}장 -> {result} ({options.fps}fps, {len(frames) / options.fps:.1f}초)")


if __name__ == "__main__":
    main()
