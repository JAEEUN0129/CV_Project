"""Build and optionally execute reproducible VACE inference commands."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VaceRun:
    repo: Path
    checkpoint: Path
    source_video: Path
    mask_video: Path
    anchor: Path
    output: Path
    prompt: str
    frame_num: int = 41
    sample_steps: int = 30
    guide_scale: float = 7.0
    seed: int = 2025

    def command(self) -> list[str]:
        return [
            "python", "vace/vace_wan_inference.py",
            "--ckpt_dir", str(self.checkpoint),
            "--src_video", str(self.source_video),
            "--src_mask", str(self.mask_video),
            "--src_ref_images", str(self.anchor),
            "--save_file", str(self.output),
            "--frame_num", str(self.frame_num),
            "--sample_steps", str(self.sample_steps),
            "--sample_guide_scale", str(self.guide_scale),
            "--base_seed", str(self.seed),
            "--prompt", self.prompt,
        ]

    def printable_command(self) -> str:
        return shlex.join(self.command())

    def execute(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.setdefault("TMPDIR", "/root/pip-tmp")
        environment.setdefault("PIP_CACHE_DIR", "/root/pip-cache")
        subprocess.run(self.command(), cwd=self.repo, env=environment, check=True)
