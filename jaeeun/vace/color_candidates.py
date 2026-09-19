"""Build three personal-colour hairstyle anchors for a VACE run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .anchor_editor import FluxAnchorEditor


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


def _prompt(base_prompt: str, color: str) -> str:
    return (
        f"{base_prompt} Change the hair color to {color}. "
        "Keep the exact hairstyle shape, length, texture, and bangs from the reference. "
        "Apply the color consistently across the hair with natural strands and shading."
    )


def build_color_candidates(
    source: Path,
    reference: Path,
    mask: Path,
    output_dir: Path,
    personal_color: str,
    flux_python: Path,
    flux_worker: Path,
    base_prompt: str,
    cpu_offload: bool = True,
) -> list[HairColorCandidate]:
    """Generate exactly three Flux anchors for one personal-colour result."""
    try:
        palette = PERSONAL_COLOR_PALETTES[personal_color]
    except KeyError as error:
        supported = ", ".join(sorted(PERSONAL_COLOR_PALETTES))
        raise ValueError(f"Unknown personal colour {personal_color!r}; use: {supported}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    editor = FluxAnchorEditor(flux_python, flux_worker, cpu_offload=cpu_offload)
    candidates = []
    for color_id, name, prompt_color in palette:
        output = output_dir / f"anchor-{personal_color}-{color_id}.png"
        editor.create(
            source,
            reference,
            mask,
            _prompt(base_prompt, prompt_color),
            output,
        )
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