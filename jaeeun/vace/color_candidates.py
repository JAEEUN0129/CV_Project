"""Build requested and recommended hair-colour anchors for a VACE run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .anchor_editor import FluxAnchorEditor
from .style_options import COLOR_OPTIONS


@dataclass(frozen=True)
class HairColorCandidate:
    color_id: str
    name: str
    prompt_color: str
    anchor: Path


PERSONAL_COLOR_PALETTES = {
    "spring_warm": (
        ("honey_brown", "허니 브라운", "warm honey brown"),
        ("caramel_brown", "캐러멜 브라운", "warm caramel brown"),
        ("peach_brown", "피치 브라운", "warm peach brown"),
    ),
    "summer_cool": (
        ("ash_brown", "애쉬 브라운", "cool ash brown"),
        ("rose_brown", "로즈 브라운", "muted cool rose brown"),
        ("blue_black", "블루 블랙", "cool blue black"),
    ),
    "autumn_warm": (
        ("chocolate_brown", "초콜릿 브라운", "warm chocolate brown"),
        ("copper_brown", "코퍼 브라운", "warm copper brown"),
        ("chestnut_brown", "체스트넛 브라운", "warm chestnut brown"),
    ),
    "winter_cool": (
        ("deep_black", "딥 블랙", "deep cool black"),
        ("burgundy", "버건디", "cool burgundy red"),
        ("violet_black", "바이올렛 블랙", "cool violet black"),
    ),
}

# Display swatches and recolouring use the same target colours.
PALETTE_HEX = {
    "honey_brown": "#B8794E", "caramel_brown": "#A96C43", "peach_brown": "#C88467",
    "ash_brown": "#82746F", "rose_brown": "#9A6F78", "blue_black": "#202A3A",
    "chocolate_brown": "#633F32", "copper_brown": "#A75E3B", "chestnut_brown": "#875035",
    "deep_black": "#17191D", "burgundy": "#713B4E", "violet_black": "#30263B",
}


def requested_hair_mask(image: Path):
    import numpy as np
    from PIL import Image
    from tempfile import TemporaryDirectory
    from ..models import HumanParser

    pixels = np.array(Image.open(image).convert("RGB"))
    # Letterboxed portraits otherwise become extremely narrow at parser resolution.
    nonblack = np.max(pixels, axis=2) > 8
    columns = np.flatnonzero(nonblack.mean(axis=0) > 0.01)
    rows = np.flatnonzero(nonblack.mean(axis=1) > 0.01)
    if not len(columns) or not len(rows):
        raise ValueError("요청한 스타일 이미지가 비어 있습니다.")
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    parser = HumanParser()
    with TemporaryDirectory() as directory:
        crop_path = Path(directory) / "portrait.png"
        Image.fromarray(pixels[y0:y1, x0:x1]).save(crop_path)
        class_map, labels = parser.predict(crop_path)
    hair_ids = [index for index, label in labels.items() if label.lower() == "hair"]
    if not hair_ids:
        raise ValueError("The parser does not provide a hair class.")
    mask = np.zeros(pixels.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = class_map == hair_ids[0]
    if not mask.any():
        raise ValueError("요청한 스타일 이미지에서 머리 영역을 찾지 못했습니다.")
    Image.fromarray(mask.astype(np.uint8) * 255).save(image.with_name("requested-hair-mask.png"))
    overlay = pixels.copy()
    overlay[mask] = (pixels[mask] * 0.5 + np.array([103, 88, 216]) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(image.with_name("requested-hair-overlay.png"))
    return mask


def recolor_requested(image: Path, mask, color: str, output: Path) -> Path:
    """Change colour only; retain image geometry and every pixel outside hair."""
    import cv2
    import numpy as np
    from PIL import Image

    source = np.array(Image.open(image).convert("RGB"))
    if mask.shape != source.shape[:2] or not mask.any():
        raise ValueError("A nonempty, frame-aligned hair mask is required.")
    lab = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255
    target = cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]
    edited = lab.copy()
    # Shift average lightness to the chosen colour while retaining strand shading.
    edited[..., 0] = np.clip(lab[..., 0] + target[0] - np.median(lab[..., 0][mask]), 0, 100)
    edited[..., 1:] = target[1:]
    rgb_edit = cv2.cvtColor(edited, cv2.COLOR_LAB2RGB)
    # Feather inward only, so skin/background pixels never change.
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2) * mask
    blended = source * (1 - alpha[..., None]) + rgb_edit * 255 * alpha[..., None]
    result = source.copy()
    result[mask] = np.clip(np.rint(blended[mask]), 0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output)
    return output


def _prompt(base_prompt: str, color: str) -> str:
    return (
        f"{base_prompt} Change the hair color to {color}. "
        "Follow the hairstyle instructions above, including selected attributes over reference attributes. "
        "Apply the color consistently across the hair with natural strands and shading."
    )


def build_color_candidates(
    source: Path,
    reference: Path | None,
    mask: Path,
    output_dir: Path,
    personal_color: str,
    flux_python: Path,
    flux_worker: Path,
    base_prompt: str,
    cpu_offload: bool = True,
    palette: tuple | None = None,
    include_requested: bool = False,
    requested_color: str | None = None,
) -> list[HairColorCandidate]:
    """Generate the supplied palette, or three personal-colour recommendations."""
    try:
        palette = palette if palette is not None else PERSONAL_COLOR_PALETTES[personal_color]
    except KeyError as error:
        supported = ", ".join(sorted(PERSONAL_COLOR_PALETTES))
        raise ValueError(f"Unknown personal colour {personal_color!r}; use: {supported}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    editor = FluxAnchorEditor(flux_python, flux_worker, cpu_offload=cpu_offload)
    candidates = []
    if include_requested:
        if requested_color:
            color = COLOR_OPTIONS[requested_color][2]
            instruction = f"Set the hair color to {color}."
        elif reference is not None:
            color = "the same as in the supplied anchor image"
            instruction = "Match the hair color of the hairstyle reference image."
        else:
            color = "the same as in the supplied anchor image"
            instruction = "Preserve the original hair color of the source person."
        output = output_dir / "anchor-requested.png"
        editor.create(source, reference, mask, f"{base_prompt} {instruction}", output)
        candidates.append(HairColorCandidate("requested", "요청한 스타일", color, output))
        requested_anchor = output
        hair_mask = requested_hair_mask(requested_anchor)
    for color_id, name, prompt_color in palette:
        output = output_dir / f"anchor-{personal_color}-{color_id}.png"
        if include_requested:
            recolor_requested(requested_anchor, hair_mask, PALETTE_HEX[color_id], output)
        else:
            editor.create(source, reference, mask, _prompt(base_prompt, prompt_color), output)
        candidates.append(HairColorCandidate(color_id, name, prompt_color, output))
    return candidates


def write_manifest(
    output: Path,
    personal_color: str,
    candidates: list[HairColorCandidate],
    classifier_result: dict | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "personal_color": personal_color,
        "classifier_result": classifier_result,
        "candidates": [
            {**asdict(candidate), "anchor": str(candidate.anchor)}
            for candidate in candidates
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def select_candidate(
    candidates: list[HairColorCandidate], color_id: str
) -> HairColorCandidate:
    for candidate in candidates:
        if candidate.color_id == color_id:
            return candidate
    available = ", ".join(candidate.color_id for candidate in candidates)
    raise ValueError(f"Unknown colour candidate {color_id!r}; choose: {available}")
