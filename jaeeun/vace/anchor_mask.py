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

    if edit_type in {"see_through_bangs", "remove_bangs"}:
        if "face" not in ids:
            raise ValueError("Human parser does not provide a face class")
        face = class_map == ids["face"]
        ys, _ = np.where(face)
        if not len(ys):
            raise ValueError("No face was detected in the selected anchor frame")
        face_top, face_bottom = int(ys.min()), int(ys.max())
        ratio = 0.32 if edit_type == "see_through_bangs" else 0.25
        forehead_bottom = face_top + round((face_bottom - face_top + 1) * ratio)
        result |= face & (np.indices(face.shape)[0] <= forehead_bottom)
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
        choices=("see_through_bangs", "short_hair", "remove_bangs", "wave"),
        required=True,
    )
    args = parser.parse_args()
    print(build_anchor_mask(args.source, args.output, args.edit_type))


if __name__ == "__main__":
    main()
