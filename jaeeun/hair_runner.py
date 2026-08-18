"""Hairstyle transfer worker, executed by the isolated HairFastGAN environment.

This file is deliberately standalone: it must import nothing from the ``jaeeun`` package,
because the environment it runs in has HairFastGAN's dependencies rather than the main
project's. ``hairstyle.py`` drives it from the main environment as a subprocess and reads
the progress lines printed here.

Usage:
    python jaeeun/hair_runner.py --faces <dir> --reference <image> --output <dir>
"""

import argparse
import sys
from pathlib import Path

import torch

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def install_cpu_compatibility() -> None:
    """Undo HairFastGAN's assumption that a CUDA device is always present.

    The published checkpoints were saved from GPU tensors and several modules pin
    themselves to the GPU regardless of the requested device, so both loading and module
    placement have to be redirected before the model package is imported.
    """
    original_load = torch.load

    def load_on_cpu(*args, **kwargs):
        # Forced rather than defaulted: some call sites pass an explicit CUDA
        # map_location, which fails outright on a CPU-only build.
        kwargs["map_location"] = "cpu"
        kwargs.setdefault("weights_only", False)
        return original_load(*args[:1], **kwargs)

    def asks_for_cuda(value) -> bool:
        return "cuda" in str(value) if isinstance(value, (str, torch.device)) else False

    def redirect_to_cpu(original):
        def wrapper(self, *args, **kwargs):
            args = tuple("cpu" if asks_for_cuda(arg) else arg for arg in args)
            if asks_for_cuda(kwargs.get("device")):
                kwargs["device"] = "cpu"
            return original(self, *args, **kwargs)

        return wrapper

    torch.load = load_on_cpu
    torch.nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.to = redirect_to_cpu(torch.nn.Module.to)
    torch.Tensor.to = redirect_to_cpu(torch.Tensor.to)


def build_model(repo: Path, weights: Path):
    sys.path.insert(0, str(repo))
    from hair_swap import HairFast, get_parser
    from models.sean_codes.models.pix2pix_model import SEAN_OPT

    # The blending network defaults to GPU 0 and asserts CUDA is present; declaring no
    # GPUs is the switch this code already provides for CPU runs.
    SEAN_OPT.gpu_ids = []

    args = get_parser().parse_args([])
    args.device = "cpu"
    args.ckpt = str(weights / "StyleGAN/ffhq.pt")
    args.rotate_checkpoint = str(weights / "Rotate/rotate_best.pth")
    args.blending_checkpoint = str(weights / "Blending/checkpoint.pth")
    args.pp_checkpoint = str(weights / "PostProcess/pp_model.pth")
    return HairFast(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer one hairstyle onto every face image in a folder.")
    parser.add_argument("--faces", type=Path, required=True, help="Directory of face images to restyle")
    parser.add_argument("--reference", type=Path, required=True, help="Image holding the desired hairstyle")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the restyled images")
    parser.add_argument("--repo", type=Path, default=Path("external/HairFastGAN"))
    parser.add_argument("--weights", type=Path, default=Path("external/HairFastGAN_weights/pretrained_models"))
    parser.add_argument("--seed", type=int, default=3407, help="Fixed so a rerun reproduces the same styling")
    options = parser.parse_args()

    faces = sorted(path for path in options.faces.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not faces:
        print("ERROR no input images found", flush=True)
        sys.exit(1)

    install_cpu_compatibility()
    model = build_model(options.repo.resolve(), options.weights.resolve())
    options.output.mkdir(parents=True, exist_ok=True)
    print(f"TOTAL {len(faces)}", flush=True)

    from torchvision.utils import save_image

    completed = 0
    for index, face in enumerate(faces):
        try:
            with torch.no_grad():
                # align=True crops each face to the 1024px layout the model was trained
                # on, and raises when no face is found — which is the expected outcome
                # once the subject turns away, not an error worth aborting the run for.
                result, aligned, *_ = model.swap(
                    face, options.reference, options.reference, align=True, seed=options.seed
                )
            save_image(result, options.output / f"{face.stem}.png")
            save_image(aligned, options.output / f"{face.stem}_aligned.png")
            completed += 1
            print(f"OK {face.name}", flush=True)
        except Exception as error:
            print(f"SKIP {face.name} {type(error).__name__}: {str(error)[:80]}", flush=True)
        print(f"PROGRESS {index + 1} {len(faces)}", flush=True)

    print(f"COMPLETED {completed}", flush=True)
    if not completed:
        sys.exit(2)


if __name__ == "__main__":
    main()
