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


def choose_device() -> str:
    """Pick the device and, on a CPU-only host, adapt the model package to it.

    The redirection below is applied only when there is no GPU. Applying it unconditionally
    would quietly pin a GPU host to the CPU, which is the difference between seconds and
    minutes per frame.
    """
    if torch.cuda.is_available():
        return "cuda"
    install_cpu_compatibility()
    return "cpu"


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


def build_model(repo: Path, weights: Path, device: str):
    sys.path.insert(0, str(repo))
    from hair_swap import HairFast, get_parser
    from models.sean_codes.models.pix2pix_model import SEAN_OPT

    # The blending network asserts CUDA is present whenever any GPU id is listed, so an
    # empty list is the switch this code already provides for CPU runs.
    SEAN_OPT.gpu_ids = [0] if device == "cuda" else []

    args = get_parser().parse_args([])
    args.device = device
    args.ckpt = str(weights / "StyleGAN/ffhq.pt")
    args.rotate_checkpoint = str(weights / "Rotate/rotate_best.pth")
    args.blending_checkpoint = str(weights / "Blending/checkpoint.pth")
    args.pp_checkpoint = str(weights / "PostProcess/pp_model.pth")
    return HairFast(args)


# The model's own face parser labels hair 13 after its label remapping. Every stage of
# the transfer relies on this, so the same value is used here rather than a second parser.
HAIR_LABEL = 13


def hair_mask_of(image: "torch.Tensor", repo: Path) -> "torch.Tensor":
    """Return the hair mask the model itself sees in a generated image.

    The project's own body parser is trained on full-length photographs and mislabels
    these tight, re-aligned face crops — it found 7% hair where roughly a third of the
    frame is hair. The model's internal parser is trained on exactly this framing, and it
    has already run during generation, so reusing it costs nothing and is what the
    transfer itself trusted.
    """
    import torchvision.transforms as T
    from models.Net import get_segmentation

    # The parser expects ImageNet-normalised input; callers inside the repo do this too.
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    batched = image.unsqueeze(0) if image.dim() == 3 else image
    parsing = get_segmentation(normalize(batched.clip(0, 1)), resize=False)
    return (parsing[0, 0] == HAIR_LABEL).float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer one hairstyle onto every face image in a folder.")
    parser.add_argument("--faces", type=Path, required=True, help="Directory of face images to restyle")
    parser.add_argument("--reference", type=Path, required=True, help="Image holding the desired hairstyle")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the restyled images")
    parser.add_argument("--repo", type=Path, default=Path("external/HairFastGAN"))
    parser.add_argument("--weights", type=Path, default=Path("external/HairFastGAN_weights/pretrained_models"))
    parser.add_argument("--seed", type=int, default=3407, help="Fixed so a rerun reproduces the same styling")
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Skip writing the hair mask. It is free to produce and needed by every "
        "post-processing step, so leave it on unless disk space is tight.",
    )
    options = parser.parse_args()

    faces = sorted(path for path in options.faces.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not faces:
        print("ERROR no input images found", flush=True)
        sys.exit(1)

    device = choose_device()
    print(f"DEVICE {device}", flush=True)
    model = build_model(options.repo.resolve(), options.weights.resolve(), device)
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
            if not options.no_mask:
                mask = hair_mask_of(result, options.repo)
                save_image(mask, options.output / f"{face.stem}_mask.png")
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
