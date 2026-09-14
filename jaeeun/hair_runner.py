"""Hairstyle transfer worker, executed by the isolated HairFastGAN environment.

This file is deliberately standalone: it must import nothing from the ``jaeeun`` package,
because the environment it runs in has HairFastGAN's dependencies rather than the main
project's. ``hairstyle.py`` drives it from the main environment as a subprocess and reads
the progress lines printed here.

Usage:
    python jaeeun/hair_runner.py --faces <dir> --reference <image> --output <dir>
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# A sibling module, imported by its own name: the script's folder is first on sys.path, so
# this does not go through the jaeeun package and pulls in nothing from the main project.
import frame_align

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
    parser.add_argument(
        "--reference",
        type=Path,
        help="Image holding the desired hairstyle. Sets both shape and colour unless one "
        "of them is given separately.",
    )
    parser.add_argument(
        "--shape",
        type=Path,
        help="Image to take the hair shape from. The model accepts shape and colour from "
        "different photos; passing one image for both is what ties a bob to its blonde.",
    )
    parser.add_argument(
        "--color",
        type=Path,
        help="Image to take the hair colour from",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for the restyled images")
    parser.add_argument("--repo", type=Path, default=Path("external/HairFastGAN"))
    parser.add_argument("--weights", type=Path, default=Path("external/HairFastGAN_weights/pretrained_models"))
    parser.add_argument("--seed", type=int, default=3407, help="Fixed so a rerun reproduces the same styling")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report which frames the face detector can handle and stop. Loading the "
        "networks takes half a minute and each frame costs minutes, so answering "
        "'will this clip work at all' before that is worth a separate mode.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Skip writing the hair mask. It is free to produce and needed by every "
        "post-processing step, so leave it on unless disk space is tight.",
    )
    parser.add_argument(
        "--stabilise-sigma",
        type=float,
        default=0.5,
        help="Temporal smoothing of the face landmarks, in frames. 0 aligns every frame on "
        "its own. See frame_align.smooth_landmarks for the sweep behind the default.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=6,
        help="Longest run of frames without a detected face that is bridged by "
        "interpolation. Longer absences are held on the last result instead.",
    )
    options = parser.parse_args()

    shape = options.shape or options.reference
    color = options.color or options.reference
    # The check mode never generates anything, so it needs no reference images.
    if not options.check_only and (shape is None or color is None):
        print("ERROR --reference, or both --shape and --color, must be given", flush=True)
        sys.exit(1)

    faces = sorted(path for path in options.faces.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not faces:
        print("ERROR no input images found", flush=True)
        sys.exit(1)

    detector, predictor = frame_align.load_detector(
        options.weights.resolve() / "ShapeAdaptor" / "shape_predictor_68_face_landmarks.dat"
    )

    if options.check_only:
        # Only the landmark detector is needed, so this stays in the seconds range. A frame
        # counts as usable if it will be generated: detected, or inside a gap short enough
        # to interpolate. Held frames reuse a neighbour's result, so they are the ones worth
        # telling the user how to reshoot.
        print(f"TOTAL {len(faces)}", flush=True)
        landmarks = []
        for index, face in enumerate(faces):
            landmarks.append(frame_align.detect_landmarks(Image.open(face), detector, predictor))
            print(f"PROGRESS {index + 1} {len(faces)}", flush=True)
        _, status = frame_align.fill_gaps(landmarks, options.max_gap)
        usable = 0
        for face, state in zip(faces, status):
            if state == frame_align.HELD:
                print(f"UNUSABLE {face.name}", flush=True)
            else:
                usable += 1
                print(f"USABLE {face.name}", flush=True)
        print(f"COMPLETED {usable}", flush=True)
        sys.exit(0 if usable else 2)

    # The model now receives crops aligned here rather than aligning with align=True, so
    # the references need the same treatment. Failing on them is an input problem the user
    # can fix, not something to discover as every frame failing one by one.
    try:
        shape_image = frame_align.align_one(Image.open(shape), detector, predictor)
        color_image = shape_image if color == shape else frame_align.align_one(Image.open(color), detector, predictor)
    except ValueError:
        print("ERROR no face found in the shape or colour reference image", flush=True)
        sys.exit(1)

    device = choose_device()
    print(f"DEVICE {device}", flush=True)
    print(f"SHAPE {shape.name}", flush=True)
    print(f"COLOR {color.name}", flush=True)
    model = build_model(options.repo.resolve(), options.weights.resolve(), device)
    options.output.mkdir(parents=True, exist_ok=True)
    print(f"TOTAL {len(faces)}", flush=True)

    from torchvision.utils import save_image

    # Align the whole clip before generating anything: smoothing needs every frame's
    # landmarks, and a gap can only be interpolated once the frame after it is known.
    aligned = frame_align.align_clip(
        [Image.open(face) for face in faces], detector, predictor,
        sigma=options.stabilise_sigma, max_gap=options.max_gap,
    )

    def copy_outputs(source_stem: str, target_stem: str) -> None:
        for suffix in ("", "_aligned", "_mask"):
            source = options.output / f"{source_stem}{suffix}.png"
            if source.exists():
                shutil.copy(source, options.output / f"{target_stem}{suffix}.png")

    # Every input frame gets an output. Dropping the ones without a face used to shorten
    # the clip and splice its two sides together, which played as a jump; holding the last
    # result keeps the timeline, so the sound track can stay too.
    quads = np.full((len(faces), 4, 2), np.nan)
    completed = 0
    last_done = None
    waiting = []  # frames before the first result, filled in once there is one
    for index, (face, (crop, quad, state)) in enumerate(zip(faces, aligned)):
        if crop is None:
            print(f"HELD {face.name}", flush=True)
            if last_done:
                copy_outputs(last_done, face.stem)
            else:
                waiting.append(face.stem)
            print(f"PROGRESS {index + 1} {len(faces)}", flush=True)
            continue

        if state == frame_align.INTERPOLATED:
            print(f"INTERPOLATED {face.name}", flush=True)
        try:
            with torch.no_grad():
                result = model.swap(crop, shape_image, color_image, align=False, seed=options.seed)
            save_image(result, options.output / f"{face.stem}.png")
            crop.save(options.output / f"{face.stem}_aligned.png")
            if not options.no_mask:
                mask = hair_mask_of(result, options.repo)
                save_image(mask, options.output / f"{face.stem}_mask.png")
            quads[index] = quad
            completed += 1
            print(f"OK {face.name}", flush=True)
            for stem in waiting:
                copy_outputs(face.stem, stem)
            waiting.clear()
            last_done = face.stem
        except Exception as error:
            # Still reported as SKIP, but held like a faceless frame so the clip keeps
            # its length.
            print(f"SKIP {face.name} {type(error).__name__}: {str(error)[:80]}", flush=True)
            if last_done:
                copy_outputs(last_done, face.stem)
            else:
                waiting.append(face.stem)
        print(f"PROGRESS {index + 1} {len(faces)}", flush=True)

    # Where each crop came from in its source frame, for putting the result back into the
    # original picture later. Held frames stay NaN: they have no crop of their own.
    np.save(options.output / "quads.npy", quads)
    (options.output / "frame_status.txt").write_text(
        "\n".join(f"{face.name} {state}" for face, (_, _, state) in zip(faces, aligned)) + "\n"
    )

    print(f"COMPLETED {completed}", flush=True)
    if not completed:
        sys.exit(2)


if __name__ == "__main__":
    main()
