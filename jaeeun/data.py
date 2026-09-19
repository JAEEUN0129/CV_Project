"""Dataset catalog and image/video preprocessing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    purpose: str
    homepage: str
    access_note: str


DATASETS = (
    DatasetInfo("LIP", "single-person human parsing", "https://sysu-hcp.net/lip/", "Verify current research-use terms."),
    DatasetInfo("CIHP", "multi-person human parsing", "https://github.com/Engineering-Course/CIHP_PGN", "Verify image-source terms."),
    DatasetInfo("ATR", "fashion human parsing", "https://github.com/lemondan/HumanParsing-Dataset", "Verify redistribution terms."),
    DatasetInfo("ModaNet", "fashion segmentation", "https://github.com/eBay/modanet", "Review the dataset license."),
    DatasetInfo("DeepFashion2", "clothing masks and landmarks", "https://github.com/switchablenorms/DeepFashion2", "Non-commercial research terms."),
    DatasetInfo("VITON-HD", "virtual try-on", "https://github.com/shadow2496/VITON-HD", "Review source-data restrictions."),
    DatasetInfo("SA-V", "video object segmentation", "https://ai.meta.com/datasets/segment-anything-video/", "CC BY 4.0; no human-part labels."),
)


def square_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return left, top, left + side, top + side


def preprocess_image(source: Path, destination: Path, size: int = 512) -> Path:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source).convert("RGB") as image:
        # Preserve the full head and outfit. Center-cropping portrait images can cut
        # off hair, so resize to fit and pad the remaining area instead.
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        offset = ((size - image.width) // 2, (size - image.height) // 2)
        canvas.paste(image, offset)
        canvas.save(destination)
    return destination


def sampled_fps(video: Path, target_fps: int = 8) -> float:
    """Real-time rate of the frames ``extract_frames`` keeps.

    Only whole frames can be skipped, so the kept frames rarely land on ``target_fps``
    exactly — sampling 30 fps footage every 4th frame yields 7.5, not 8. Reassembling at
    ``target_fps`` instead of this rate would play the clip back fast and cut the audio
    short, so the caller that rebuilds the MP4 must use this value.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    capture.release()
    return source_fps / max(1, round(source_fps / target_fps))


def extract_frames(video: Path, output_dir: Path, target_fps: int = 8) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear first: frames are numbered from zero, so a shorter clip would otherwise leave
    # the tail of a previous one behind and later stages would treat it as this video.
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()
    capture = cv2.VideoCapture(str(video))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    stride, index, paths = max(1, round(source_fps / target_fps)), 0, []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            path = output_dir / f"{len(paths):06d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)
        index += 1
    capture.release()
    return paths
