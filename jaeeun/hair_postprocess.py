"""Post-processing applied to hairstyle-transfer results.

The hairstyle model copies shape and colour together from one reference photo, and it
regenerates the hair independently for every frame. Two consequences follow, and both are
addressed here by locating the hair in the *generated* image and adjusting only that region:

* the colour cannot be chosen — picking a bob also forces the reference's blonde;
* the colour drifts across a clip, because nothing ties one frame's hair to the next.

The mask has to come from the generated image rather than the original. The hair occupies
a different area after restyling, and the generated frame is a re-aligned square crop whose
pixels do not correspond to the source frame at all.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .models import HumanParser


def _read_image(path: Path) -> np.ndarray:
    """Read an image from any path.

    OpenCV's own reader goes through the ANSI filesystem API on Windows and silently
    fails on non-ASCII paths, which this project has — the hairstyle presets are named
    in Korean. Reading the bytes ourselves and decoding them sidesteps that entirely.
    """
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> Path:
    """Write an image to any path, for the same reason as ``_read_image``."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"이미지를 저장할 수 없습니다: {path}")
    encoded.tofile(str(path))
    return path


def _hair_mask(parser: HumanParser, image: Path) -> np.ndarray:
    """Locate hair in a generated image. Returns a 0-255 mask."""
    class_map, labels = parser.predict(image)
    hair_id = next(
        (id for id, name in labels.items() if name.replace(" ", "_") == "hair"), None
    )
    if hair_id is None:
        raise ValueError("부위 분할 모델에 머리카락 항목이 없습니다.")
    return np.where(class_map == hair_id, 255, 0).astype(np.uint8)


def _mean_lab(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    import cv2

    region = mask > 127
    if not region.any():
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[region].mean(axis=0)


def recolor_generated_hair(
    images: Iterable[Path],
    color: str,
    output_dir: Path,
    strength: float = 0.72,
    parser: HumanParser | None = None,
) -> list[Path]:
    """Repaint the hair of already-generated images in a chosen colour.

    This is what separates shape from colour: the model supplies the shape, and the colour
    comes from the caller instead of from the reference photo.
    """
    import cv2

    parser = parser or HumanParser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    results = []
    for image in images:
        source = _read_image(image)
        mask = _hair_mask(parser, image)
        target = np.full_like(source, rgb[::-1], dtype=np.uint8)
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        # Keep the generated luminance: that is where the strand detail and the shading
        # from the scene live. Only the two colour channels are replaced.
        target_lab[..., 0] = source_lab[..., 0]
        recolored = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
        alpha = (cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0 * strength)[..., None]
        destination = output_dir / image.name
        _write_image(destination, (source * (1 - alpha) + recolored * alpha).astype(np.uint8))
        results.append(destination)
    return results


def stabilize_hair_color(
    frames: list[Path],
    output_dir: Path,
    parser: HumanParser | None = None,
) -> tuple[list[Path], dict[str, float]]:
    """Pull every frame's hair colour towards the colour typical of the whole clip.

    Each frame is generated independently, so the hair shade wanders over a sequence even
    though the reference never changes. The median across the clip is used as the anchor
    rather than the first frame, so one badly generated frame cannot drag the rest with it.

    Only the two colour channels are shifted; luminance is untouched, which keeps the
    per-frame shading that makes the hair sit in the scene.
    """
    import cv2

    parser = parser or HumanParser()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = [_read_image(frame) for frame in frames]
    masks = [_hair_mask(parser, frame) for frame in frames]
    means = [_mean_lab(image, mask) for image, mask in zip(loaded, masks)]

    measured = [value for value in means if value is not None]
    if not measured:
        raise ValueError("어느 프레임에서도 머리카락을 찾지 못했습니다.")
    anchor = np.median(np.stack(measured), axis=0)

    results = []
    for frame, image, mask, mean in zip(frames, loaded, masks, means):
        destination = output_dir / frame.name
        if mean is None:
            _write_image(destination, image)
            results.append(destination)
            continue
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        shift = (anchor - mean)[1:]  # colour channels only; luminance is left alone
        alpha = (cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0)[..., None]
        lab[..., 1:] += shift * alpha
        corrected = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        _write_image(destination, corrected)
        results.append(destination)

    before = np.stack(measured)
    report = {
        "before_swing": float(np.abs(before.max(axis=0) - before.min(axis=0))[1:].max()),
        "anchor_a": float(anchor[1]),
        "anchor_b": float(anchor[2]),
    }
    return results, report


def measure_color_swing(frames: list[Path], parser: HumanParser | None = None) -> dict[str, float]:
    """Report how much the hair colour moves across a clip, for before/after comparison."""
    import cv2

    parser = parser or HumanParser()
    means = []
    for frame in frames:
        image = _read_image(frame)
        mean = _mean_lab(image, _hair_mask(parser, frame))
        if mean is not None:
            means.append(mean)
    stacked = np.stack(means)
    steps = np.abs(np.diff(stacked, axis=0))[:, 1:]
    return {
        # Swing: the widest gap between any two frames — what reads as the hair changing
        # shade over the clip. Step: the typical frame-to-frame jump, which reads as flicker.
        "swing": float(np.abs(stacked.max(axis=0) - stacked.min(axis=0))[1:].max()),
        "step": float(steps.max(axis=1).mean()) if len(steps) else 0.0,
        "frames": len(means),
    }
