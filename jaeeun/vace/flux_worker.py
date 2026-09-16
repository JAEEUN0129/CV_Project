"""Isolated FLUX.1 Kontext reference-guided inpainting worker."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import FluxKontextInpaintPipeline
from PIL import Image, ImageFilter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mask-blur", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FLUX anchor generation requires a CUDA GPU")
    source = Image.open(args.source).convert("RGB")
    reference = Image.open(args.reference).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    if mask.size != source.size:
        raise ValueError(f"Mask size {mask.size} does not match source size {source.size}")
    if args.mask_blur:
        mask = mask.filter(ImageFilter.GaussianBlur(args.mask_blur))

    pipe = FluxKontextInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    result = pipe(
        prompt=args.prompt,
        image=source,
        mask_image=mask,
        image_reference=reference,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
