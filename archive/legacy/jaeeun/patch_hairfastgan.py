"""Make the cloned HairFastGAN checkout runnable on this project's machines.

HairFastGAN assumes an NVIDIA GPU with a CUDA toolchain. Four assumptions break on a
CPU-only machine, and one of them would also break on a GPU host whose image lacks the
compiler. This script rewrites those spots in the checkout so the same code runs on both.

The checkout lives under ``external/`` and is not tracked by git, so these edits cannot be
committed with the rest of the project — they have to be reapplied after every clone.
Run once after cloning:

    python jaeeun/patch_hairfastgan.py

Every edit is idempotent; running it again reports the files as already done.
"""

import argparse
import re
from pathlib import Path

MARKER = "# patched: CUDA extension is optional"
# `name = load(` ... `)` at module scope, spanning the whole call.
LOAD_CALL = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*) = load\((.*?)^\)\s*$", re.M | re.S)
# The name alone, which still matches once the call has been indented into a try block.
EXTENSION_NAME = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*) = load\(", re.M)
CPU_BRANCH = re.compile(r"if input\.device\.type == [\"']cpu[\"']:")
HARDCODED_DEVICE = re.compile(r"""device=['"]cuda['"]""")
DEVICE_EXPRESSION = "device=('cuda' if torch.cuda.is_available() else 'cpu')"


def patch_custom_ops(root: Path) -> list[str]:
    """Let the StyleGAN2 ops run without their compiled CUDA kernels.

    Two edits per file. First, compiling at import time must not be fatal. Second, the
    pure-PyTorch branch has to be chosen whenever the kernels are missing, not only when
    the tensors are on the CPU — otherwise a GPU host without a compiler dereferences a
    module that failed to load.
    """
    changed = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if " = load(" not in source:
            continue
        original = source

        match = LOAD_CALL.search(source)
        name_match = EXTENSION_NAME.search(source)
        extension = name_match.group(1) if name_match else None
        if match and MARKER not in source:
            call = f"{extension} = load({match.group(2)})"
            indented = "\n".join(("    " + line) if line.strip() else line for line in call.splitlines())
            source = source[: match.start()] + (
                f"{MARKER} — the native branches below cover its absence.\n"
                f"try:\n{indented}\nexcept Exception:\n    {extension} = None\n"
            ) + source[match.end():]

        if extension and f"{extension} is None or" not in source:
            source = CPU_BRANCH.sub(
                f'if {extension} is None or input.device.type == "cpu":', source
            )

        if source != original:
            path.write_text(source, encoding="utf-8")
            changed.append(str(path.relative_to(root)))
    return changed


def patch_hardcoded_devices(root: Path) -> list[str]:
    """Follow whatever device exists instead of demanding CUDA."""
    changed = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not HARDCODED_DEVICE.search(source):
            continue
        if not re.search(r"^\s*import torch\b", source, re.M):
            continue
        path.write_text(HARDCODED_DEVICE.sub(DEVICE_EXPRESSION, source), encoding="utf-8")
        changed.append(str(path.relative_to(root)))
    return changed


def patch_bicubic(root: Path) -> list[str]:
    """The downsampler builds a CUDA tensor type name from a default of True.

    On a CPU-only build that type does not exist at all, so the default has to be decided
    at construction time. Every call site relies on the default.
    """
    path = root / "utils/bicubic.py"
    source = path.read_text(encoding="utf-8")
    if "cuda=None" in source:
        return []
    source = source.replace(
        "    def __init__(self, factor=4, cuda=True, padding='reflect'):",
        "    # patched: default to whatever the machine actually has.\n"
        "    def __init__(self, factor=4, cuda=None, padding='reflect'):\n"
        "        if cuda is None:\n"
        "            cuda = torch.cuda.is_available()",
    )
    path.write_text(source, encoding="utf-8")
    return [str(path.relative_to(root))]


def patch_mask_interpolation(root: Path) -> list[str]:
    """Nearest-neighbour interpolation has no integer CPU kernel.

    Nearest picks existing values rather than averaging them, so casting the label map to
    float leaves the labels themselves untouched.
    """
    path = root / "utils/image_utils.py"
    source = path.read_text(encoding="utf-8")
    if "mask.float(), size=(256, 256)" in source:
        return []
    source = source.replace(
        "mask = F.interpolate(mask, size=(256, 256), mode='nearest')",
        "# patched: no integer nearest-neighbour kernel exists on the CPU.\n"
        "        mask = F.interpolate(mask.float(), size=(256, 256), mode='nearest')",
    )
    path.write_text(source, encoding="utf-8")
    return [str(path.relative_to(root))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/HairFastGAN"))
    root = parser.parse_args().repo
    if not root.is_dir():
        raise SystemExit(f"HairFastGAN 체크아웃을 찾을 수 없습니다: {root}")

    steps = {
        "CUDA 확장 선택적 사용": patch_custom_ops,
        "고정된 cuda 장치 지정": patch_hardcoded_devices,
        "축소 필터 기본 장치": patch_bicubic,
        "마스크 보간 자료형": patch_mask_interpolation,
    }
    for label, step in steps.items():
        changed = step(root)
        print(f"{label}: {'수정 ' + ', '.join(changed) if changed else '이미 적용됨'}")


if __name__ == "__main__":
    main()
