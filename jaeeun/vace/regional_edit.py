"""Regional hairstyle edits with exact preservation outside the selected region."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from ..models import HumanParser


def regional_masks(image: Path, bangs_style: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.array(Image.open(image).convert("RGB"))
    nonblack = pixels.max(axis=2) > 8
    xs = np.flatnonzero(nonblack.mean(axis=0) > .01)
    ys = np.flatnonzero(nonblack.mean(axis=1) > .01)
    if not len(xs) or not len(ys):
        raise ValueError("부분 편집할 인물 이미지가 비어 있습니다.")
    x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
    with TemporaryDirectory() as directory:
        crop = Path(directory) / "portrait.png"
        Image.fromarray(pixels[y0:y1, x0:x1]).save(crop)
        cropped, labels = HumanParser("jonathandinu/face-parsing").predict(crop)
    classes = np.zeros(pixels.shape[:2], dtype=np.uint8)
    classes[y0:y1, x0:x1] = cropped
    ids = {name.lower(): i for i, name in labels.items()}
    face_ids = [ids[n] for n in ("skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth") if n in ids]
    face = np.isin(classes, face_ids)
    ys, xs = np.where(face)
    if not len(ys) or "hair" not in ids:
        raise ValueError("앞머리와 옆머리를 나눌 얼굴 영역을 찾지 못했습니다.")
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    brow_ids = [ids[n] for n in ("l_brow", "r_brow") if n in ids]
    brow_y = np.where(np.isin(classes, brow_ids))[0]
    # Protect eyebrows/eyes; the forehead and existing fringe above them are editable.
    bottom = int(brow_y.min()) if len(brow_y) else int(ys.min() + height * .28)
    rows, cols = np.indices(classes.shape)
    fringe_zone = ((cols >= xs.min() - width * .22) & (cols <= xs.max() + width * .22)
                   & (rows >= max(0, ys.min() - height * .65)) & (rows < bottom))
    # Curtain bangs descend beside the eyes toward the temples. Give them side
    # room without opening the eyes/nose to editing, and lock it for body edits.
    side_zone = ((cols >= xs.min() - width * .25) & (cols <= xs.max() + width * .25)
                 & ((cols <= xs.min() + width * .12) | (cols >= xs.max() - width * .12))
                 & (rows >= bottom) & (rows < bottom + height * .35))
    hair = classes == ids["hair"]
    protected_ids = [ids[n] for n in ("nose", "l_eye", "r_eye", "l_brow", "r_brow",
                                     "mouth", "u_lip", "l_lip", "neck", "cloth", "ear_r") if n in ids]
    fringe = (fringe_zone | (side_zone if bangs_style == "커튼뱅" else False)) & ~np.isin(classes, protected_ids)
    radius = max(3, round(width * .12))
    expanded = cv2.dilate(hair.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
    body = expanded & ~(fringe_zone | side_zone) & ~face & ~np.isin(classes, protected_ids)
    return fringe, body


def composite_region(source: Path, generated: Path, mask: np.ndarray, output: Path) -> Path:
    original = np.array(Image.open(source).convert("RGB"))
    edited = np.array(Image.open(generated).convert("RGB"))
    if edited.shape != original.shape or mask.shape != original.shape[:2] or not mask.any():
        raise ValueError("부분 편집 이미지와 마스크의 크기 또는 영역이 올바르지 않습니다.")
    result = original.copy()
    result[mask] = edited[mask]
    Image.fromarray(result).save(output)
    return output


def edit_region(editor, source: Path, mask: np.ndarray, prompt: str, output: Path) -> Path:
    if not mask.any():
        raise ValueError("부분 편집 영역이 비어 있습니다.")
    mask_path = output.with_name(output.stem + "-mask.png")
    raw = output.with_name(output.stem + "-raw.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    editor.create(source, None, mask_path, prompt, raw)
    # The worker blurs masks; restore protected regions explicitly after inference.
    return composite_region(source, raw, mask, output)
