"""Video mask propagation, temporal smoothing and MP4 assembly."""

from pathlib import Path
import subprocess

import cv2
import numpy as np


def mask_center(mask: np.ndarray) -> tuple[int, int]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        raise ValueError("The seed mask is empty.")
    x, y, width, height = cv2.boundingRect(points)
    return x + width // 2, y + height // 2


def save_sam2_masks(predictions, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame_index, object_ids, logits in predictions:
        for object_id, mask_logits in zip(object_ids, logits):
            mask = (mask_logits > 0).detach().cpu().numpy().squeeze().astype(np.uint8) * 255
            path = output_dir / f"frame_{int(frame_index):06d}_object_{int(object_id)}.png"
            cv2.imwrite(str(path), mask)
            paths.append(path)
    return paths


def smooth_frames(
    frame_paths: list[Path],
    output_dir: Path,
    alpha: float = 0.15,
    region_paths: list[Path] | None = None,
) -> list[Path]:
    """Apply optical-flow-aligned temporal blending to edited frames.

    When ``region_paths`` is given, blending is confined to the edited region of each
    frame. Optical flow is never exact, so blending the whole frame leaves a ghost of
    the previous frame over areas that were meant to stay untouched — the face and the
    background above all, which the project promises to preserve pixel for pixel.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_source = previous_result = None
    outputs = []
    for index, path in enumerate(frame_paths):
        current = cv2.imread(str(path))
        result = current
        if previous_source is not None:
            prev_gray = cv2.cvtColor(previous_source, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            height, width = curr_gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            map_x = (grid_x - flow[..., 0]).astype(np.float32)
            map_y = (grid_y - flow[..., 1]).astype(np.float32)
            warped = cv2.remap(previous_result, map_x, map_y, cv2.INTER_LINEAR)
            blended = cv2.addWeighted(current, 1 - alpha, warped, alpha, 0)
            if region_paths is None:
                result = blended
            else:
                region = cv2.imread(str(region_paths[index]), cv2.IMREAD_GRAYSCALE)
                weight = (cv2.GaussianBlur(region, (0, 0), 2.0).astype(np.float32) / 255.0)[..., None]
                result = (blended * weight + current * (1 - weight)).astype(np.uint8)
        destination = output_dir / f"frame_{index:06d}.png"
        cv2.imwrite(str(destination), result)
        outputs.append(destination)
        previous_source, previous_result = current, result
    return outputs


def frames_to_mp4(
    frames_dir: Path, output: Path, fps: float = 8, audio_source: Path | None = None
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    video_only = output.with_name(output.stem + "_video_only.mp4") if audio_source else output
    command = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_only),
    ]
    subprocess.run(command, check=True, capture_output=True)
    if audio_source:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_only), "-i", str(audio_source), "-map", "0:v:0",
            "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output),
        ], check=True, capture_output=True)
        video_only.unlink(missing_ok=True)
    return output


def make_test_video(image: Path, frames_dir: Path, output: Path, frame_count: int = 4, fps: int = 4) -> Path:
    """Create a tiny deterministic motion clip used by the integration test."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(image))
    for index in range(frame_count):
        transform = np.float32([[1, 0, index * 2], [0, 1, 0]])
        frame = cv2.warpAffine(source, transform, (source.shape[1], source.shape[0]), borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(str(frames_dir / f"frame_{index:06d}.png"), frame)
    return frames_to_mp4(frames_dir, output, fps)


def convert_frames_for_sam2(source_dir: Path, output_dir: Path) -> list[Path]:
    """Convert arbitrary extracted frames to SAM 2's numeric JPEG convention."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    outputs = []
    for index, source_path in enumerate(source_paths):
        destination = output_dir / f"{index:06d}.jpg"
        cv2.imwrite(str(destination), cv2.imread(str(source_path)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        outputs.append(destination)
    return outputs
