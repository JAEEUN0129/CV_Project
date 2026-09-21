"""Download project checkpoints with resumable transfers.

Run this as a background process from the repository root. Progress and errors are
written to ``artifacts/jaeeun/model_download.log`` by the launcher.
"""

from pathlib import Path

from huggingface_hub import snapshot_download
import requests


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_small.pt"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination} already exists", flush=True)
        return
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    print(f"[download] {url} -> {destination} (resume={existing})", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        downloaded = existing if append else 0
        with partial.open(mode) as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded % (25 * 1024 * 1024) < len(chunk):
                    print(f"[progress] {destination.name}: {downloaded // (1024 * 1024)} MiB", flush=True)
    partial.replace(destination)
    print(f"[done] {destination} ({destination.stat().st_size} bytes)", flush=True)


def main() -> None:
    download_file(SAM2_URL, CHECKPOINTS / "sam2" / "sam2.1_hiera_small.pt")
    destination = CHECKPOINTS / "diffusion" / "stable-diffusion-2-inpainting"
    # The original Stability AI repository currently requires authentication.
    # Use the public sd2-community mirror of the same deprecated SD2 checkpoint.
    repo_id = "sd2-community/stable-diffusion-2-inpainting"
    print(f"[download] {repo_id} -> {destination}", flush=True)
    # The repository ships every weight four times over (full and half precision, each
    # as .bin and .safetensors) plus two standalone single-file checkpoints that this
    # folder-style pipeline never reads — 26 GB in total for 2.6 GB of useful weights.
    # Take only the half-precision safetensors, which InpaintingEditor loads by asking
    # for the "fp16" variant; keep the two in step if either is ever changed.
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        allow_patterns=["*.json", "*.txt", "*.fp16.safetensors"],
    )
    print("[done] all requested model files", flush=True)


if __name__ == "__main__":
    main()
