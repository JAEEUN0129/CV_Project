"""Generate an image-edit mask for the selected anchor frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..models import HumanParser


def build_anchor_mask(source: Path, output: Path, edit_type: str) -> Path:
    class_map, labels = HumanParser().predict(source)
    ids = {name.lower().replace(" ", "_"): class_id for class_id, name in labels.items()}
    if "hair" not in ids:
        raise ValueError("Human parser does not provide a hair class")
    hair = class_map == ids["hair"]
    result = hair.copy()

    if edit_type == "short_hair":
        # Removing long hair requires enough editable context for the image
        # model to reconstruct the revealed neck, clothes, and background.
        # Expand outside the old silhouette while keeping the parsed face fixed.
        hair_height, hair_width = hair.shape
        face = class_map == ids["face"] if "face" in ids else np.zeros_like(hair)
        face_ys, face_xs = np.where(face)
        scale_width = int(face_xs.max() - face_xs.min() + 1) if len(face_xs) else hair_width // 3
        scale_height = int(face_ys.max() - face_ys.min() + 1) if len(face_ys) else hair_height // 3
        kernel_width = max(15, round(scale_width * 0.18)) | 1
        kernel_height = max(15, round(scale_height * 0.14)) | 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_width, kernel_height)
        )
        expanded = cv2.dilate(hair.astype(np.uint8), kernel) > 0

        # Give the lower ends extra room because those pixels must become neck,
        # clothing, or background rather than another copy of the long hair.
        lower_kernel_height = max(11, round(scale_height * 0.10)) | 1
        lower_kernel = np.ones((lower_kernel_height, 3), dtype=np.uint8)
        expanded |= cv2.dilate(hair.astype(np.uint8), lower_kernel) > 0
        result = expanded & ~face
    elif edit_type in {"see_through_bangs", "curtain_bangs", "remove_bangs"}:
        if "face" not in ids:
            raise ValueError("Human parser does not provide a face class")
        face = class_map == ids["face"]
        ys, _ = np.where(face)
        if not len(ys):
            raise ValueError("No face was detected in the selected anchor frame")
        face_top, face_bottom = int(ys.min()), int(ys.max())
        rows, columns = np.indices(face.shape)
        if edit_type == "curtain_bangs":
            _, xs = np.where(face)
            face_left, face_right = int(xs.min()), int(xs.max())
            normalised_x = np.clip(
                (columns - face_left) / max(1, face_right - face_left), 0.0, 1.0
            )
            distance_from_centre = np.abs(normalised_x - 0.5) * 2.0
            depth = 0.12 + (0.42 - 0.12) * distance_from_centre
            forehead_bottom = face_top + (face_bottom - face_top + 1) * depth
            result |= face & (rows <= forehead_bottom)
        else:
            ratio = 0.32 if edit_type == "see_through_bangs" else 0.25
            forehead_bottom = face_top + round((face_bottom - face_top + 1) * ratio)
            result |= face & (rows <= forehead_bottom)
    elif edit_type == "wave":
        expanded = cv2.dilate(hair.astype(np.uint8), np.ones((31, 61), np.uint8)) > 0
        if "face" in ids:
            expanded &= class_map != ids["face"]
        result |= expanded
    elif edit_type != "short_hair":
        raise ValueError(f"Unknown edit type: {edit_type}")

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(result, 255, 0).astype(np.uint8)).save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--edit-type",
        choices=(
            "see_through_bangs",
            "curtain_bangs",
            "short_hair",
            "remove_bangs",
            "wave",
        ),
        required=True,
    )
    args = parser.parse_args()
    print(build_anchor_mask(args.source, args.output, args.edit_type))


if __name__ == "__main__":
    main()
