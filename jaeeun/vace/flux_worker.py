"""Isolated FLUX.2 Klein reference-guided inpainting worker."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
from PIL import Image, ImageFilter


def _working_size(width: int, height: int, max_area: int = 1024 * 1024) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(max_area / (width * height)))
    resized_width = max(16, round(width * scale / 16) * 16)
    resized_height = max(16, round(height * scale / 16) * 16)
    return resized_width, resized_height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument(
        "--edit-mode", choices=("masked", "reference_only"), default="masked"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--mask-blur", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FLUX anchor generation requires a CUDA GPU")
    source = Image.open(args.source).convert("RGB")
    reference = Image.open(args.reference).convert("RGB") if args.reference else None
    width, height = _working_size(*source.size)
    source_work = source.resize((width, height), Image.Resampling.LANCZOS)
    if args.edit_mode == "masked":
        if args.mask is None:
            parser.error("--mask is required when --edit-mode=masked")
        mask = Image.open(args.mask).convert("L")
        if mask.size != source.size:
            raise ValueError(f"Mask size {mask.size} does not match source size {source.size}")
        composite_mask = (
            mask.filter(ImageFilter.GaussianBlur(args.mask_blur)) if args.mask_blur else mask
        )
        mask_work = mask.resize((width, height), Image.Resampling.NEAREST)
        pipe = Flux2KleinInpaintPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )
    else:
        composite_mask = None
        mask_work = None
        pipe = Flux2KleinPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    common_args = {
        "height": height,
        "width": width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }
    if args.edit_mode == "masked":
        prompt = (
            "Follow the requested hairstyle attributes. Keep the source person's "
            "exact identity, pose, lighting, and background. Change only the white masked hair "
            "region. " + args.prompt
        )
        result = pipe(
            prompt=prompt,
            image=source_work,
            mask_image=mask_work,
            **({"image_reference": reference} if reference is not None else {}),
            **common_args,
        ).images[0]
    else:
        prompt = (
            "Image 1 is the source person. If Image 2 is provided, use it as the hairstyle reference, "
            "with explicit requested attributes taking priority. "
            "Keep the exact face identity, facial features, expression, pose, hands, clothing, "
            "lighting, camera framing, and background from Image 1. " + args.prompt
        )
        result = pipe(
            prompt=prompt,
            image=[source_work, reference] if reference is not None else [source_work],
            **common_args,
        ).images[0]
    result = result.resize(source.size, Image.Resampling.LANCZOS)
    # Keep pixels outside the edit mask exactly identical to the source even if
    # the inpainting decoder introduces small whole-image color differences.
    if composite_mask is not None:
        result = Image.composite(result, source, composite_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
