"""Interchangeable manual and external-FLUX anchor providers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class AnchorEditor(Protocol):
    def create(
        self, source: Path, reference: Path, mask: Path | None, prompt: str, output: Path
    ) -> Path: ...


@dataclass(frozen=True)
class ManualAnchorEditor:
    anchor: Path

    def create(
        self, source: Path, reference: Path, mask: Path | None, prompt: str, output: Path
    ) -> Path:
        if not self.anchor.is_file():
            raise FileNotFoundError(f"Manual anchor not found: {self.anchor}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.anchor, output)
        return output


@dataclass(frozen=True)
class FluxAnchorEditor:
    python: Path
    worker: Path

    def create(
        self, source: Path, reference: Path, mask: Path | None, prompt: str, output: Path
    ) -> Path:
        if mask is None:
            raise ValueError("FLUX anchor generation requires an anchor edit mask")
        command = [
            str(self.python), str(self.worker),
            "--source", str(source),
            "--reference", str(reference),
            "--mask", str(mask),
            "--prompt", prompt,
            "--output", str(output),
        ]
        subprocess.run(command, check=True)
        if not output.is_file():
            raise RuntimeError(f"FLUX worker did not create {output}")
        return output
