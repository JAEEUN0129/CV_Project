"""Standalone HairCLIP colour-editing worker.

The upstream project assumes CUDA and imports e4e and HairCLIP through two conflicting
top-level ``models`` packages.  This worker runs the two stages sequentially, clears the
e4e modules, and redirects upstream ``.cuda()`` calls when only a CPU is available.
"""

import argparse
from argparse import Namespace
import importlib.util
from pathlib import Path
import sys
import types

import torch


def install_cpu_compatibility() -> None:
    if torch.cuda.is_available():
        return
    original_load = torch.load

    def load_on_cpu(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_on_cpu
    torch.nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def clear_package(name: str) -> None:
    for loaded in list(sys.modules):
        if loaded == name or loaded.startswith(name + "."):
            del sys.modules[loaded]


def install_e4e_cpu_ops(repo: Path) -> None:
    """Replace e4e's CUDA extensions with HairCLIP's native PyTorch operators."""
    op_dir = repo / "models" / "stylegan2" / "op"

    def load_file(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    fused = load_file("_hairclip_cpu_fused", op_dir / "fused_act.py")
    upsample = load_file("_hairclip_cpu_upfirdn", op_dir / "upfirdn2d.py")
    compatibility = types.ModuleType("models.stylegan2.op")
    compatibility.FusedLeakyReLU = fused.FusedLeakyReLU
    compatibility.fused_leaky_relu = fused.fused_leaky_relu
    compatibility.upfirdn2d = upsample.upfirdn2d
    sys.modules["models.stylegan2.op"] = compatibility


def invert(source: Path, repo: Path, checkpoint: Path, landmark_model: Path, device: str):
    e4e = repo / "encoder4editing"
    sys.path.insert(0, str(e4e))
    if device == "cpu":
        import models
        import models.stylegan2
        install_e4e_cpu_ops(repo)
    from PIL import Image
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    from models.psp import pSp
    from utils.alignment import align_face
    import dlib
    from torchvision import transforms

    ckpt = torch.load(checkpoint, map_location="cpu")
    options = dict(ckpt["opts"])
    options["checkpoint_path"] = str(checkpoint)
    options["device"] = device
    net = pSp(Namespace(**options)).eval().to(device)
    aligned = align_face(str(source), dlib.shape_predictor(str(landmark_model)))
    tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(aligned).unsqueeze(0).to(device)
    with torch.inference_mode():
        _, latent = net(tensor.float(), randomize_noise=False, return_latents=True)
    del net, ckpt, tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    sys.path.remove(str(e4e))
    clear_package("models")
    clear_package("configs")
    clear_package("utils")
    return latent


def edit(
    latent, prompt: str, repo: Path, checkpoint: Path, clip_checkpoint: Path,
    output: Path, device: str,
) -> None:
    sys.path.insert(0, str(repo))
    import clip

    original_clip_load = clip.load
    runtime_device = device

    def load_local_clip(name, device=None, *args, **kwargs):
        return original_clip_load(
            str(clip_checkpoint), device=runtime_device, *args, **kwargs
        )

    clip.load = load_local_clip

    ckpt = torch.load(checkpoint, map_location="cpu")
    options = dict(ckpt["opts"])
    options.update({
        "checkpoint_path": str(checkpoint),
        "editing_type": "color",
        "input_type": "text",
    })
    from mapper.hairclip_mapper import HairCLIPMapper

    net = HairCLIPMapper(Namespace(**options)).eval().to(device)
    latent = latent.to(device).float()
    text = clip.tokenize([prompt]).to(device)
    empty = torch.zeros((1, 1), device=device)
    with torch.inference_mode():
        edited_latent = latent + 0.1 * net.mapper(latent, empty, text, empty, empty)
        result, _ = net.decoder(
            [edited_latent], input_is_latent=True, return_latents=True,
            randomize_noise=False, truncation=1,
        )
    from torchvision.utils import save_image
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output, normalize=True, value_range=(-1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--e4e-checkpoint", type=Path, required=True)
    parser.add_argument("--landmark-model", type=Path, required=True)
    args = parser.parse_args()
    install_cpu_compatibility()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE " + device, flush=True)
    latent = invert(args.source, args.repo, args.e4e_checkpoint, args.landmark_model, device)
    print("INVERTED", flush=True)
    edit(
        latent, args.prompt, args.repo, args.checkpoint, args.clip_checkpoint,
        args.output, device,
    )
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
