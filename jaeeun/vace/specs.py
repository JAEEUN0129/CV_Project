"""Stable input contract for hairstyle-edit experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditSpec:
    name: str
    prompt: str
    mask_strategy: str
    restores_removed_hair: bool = False


EDIT_SPECS = {
    "see_through_bangs": EditSpec(
        name="see_through_bangs",
        mask_strategy="bangs",
        prompt=(
            "Apply natural wispy see-through bangs from the anchor image. Preserve the exact "
            "identity, face, existing hair length and color, expression, clothing, lighting, "
            "camera motion, and background. Keep the sparse separated fringe strands and the "
            "same hairstyle temporally consistent in every frame."
        ),
    ),
    "curtain_bangs": EditSpec(
        name="curtain_bangs",
        mask_strategy="symmetric_side_bangs",
        prompt=(
            "Apply natural center-parted curtain bangs from the anchor image. Keep a visible "
            "centre part and make the fringe flow symmetrically down both sides of the forehead "
            "toward the temples. Preserve the exact identity, face, existing hair length and "
            "color, expression, clothing, lighting, camera motion, and background. Keep the "
            "curtain bangs temporally consistent in every frame."
        ),
    ),
    "short_hair": EditSpec(
        name="short_hair",
        mask_strategy="remove_old_hair",
        restores_removed_hair=True,
        prompt=(
            "Apply the short hairstyle from the anchor image. Remove all remaining long hair "
            "inside the editable region and reconstruct the revealed neck, clothing, and "
            "background. Preserve identity, face, expression, lighting, and motion."
        ),
    ),
    "remove_bangs": EditSpec(
        name="remove_bangs",
        mask_strategy="remove_bangs",
        restores_removed_hair=True,
        prompt=(
            "Match the anchor image with no bangs. Remove the original fringe and naturally "
            "restore the forehead and hairline. Preserve all facial features, the remaining "
            "hairstyle, hair color, clothing, lighting, background, and motion."
        ),
    ),
    "wave": EditSpec(
        name="wave",
        mask_strategy="outward_hair",
        prompt=(
            "Apply the hairstyle from the anchor image with clearly defined waves continuing "
            "through the lower lengths and ends. Preserve identity, face, hair color and length, "
            "clothing, lighting, background, and motion. Keep the waves temporally consistent."
        ),
    ),
}
