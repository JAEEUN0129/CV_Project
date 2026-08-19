"""Bridge from the main environment to the isolated hairstyle-transfer environment.

HairFastGAN needs library versions that conflict with the rest of this project, so it
lives in its own virtual environment and cannot be imported here. Running it as a
subprocess keeps that isolation intact: the worker's progress lines are parsed as they
arrive so a caller can report per-frame progress during a run that takes minutes a frame.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class HairstyleEnvironment:
    interpreter: Path = Path(".venv_hair/Scripts/python.exe")
    repo: Path = Path("external/HairFastGAN")
    weights: Path = Path("external/HairFastGAN_weights/pretrained_models")
    presets: Path = Path("data/jaeeun/hairstyles")

    def missing_parts(self) -> list[str]:
        """Name whatever is absent, so the UI can explain the gap instead of crashing."""
        required = {
            "격리 실행 환경": self.interpreter,
            "모델 코드": self.repo,
            "모델 가중치": self.weights,
        }
        return [name for name, path in required.items() if not path.exists()]

    @property
    def is_available(self) -> bool:
        return not self.missing_parts()

    def available_presets(self) -> dict[str, Path]:
        if not self.presets.is_dir():
            return {}
        return {path.stem: path for path in sorted(self.presets.iterdir()) if path.suffix.lower() != ".md"}


class HairstyleTransfer:
    def __init__(self, environment: HairstyleEnvironment | None = None) -> None:
        self.environment = environment or HairstyleEnvironment()

    def restyle_folder(
        self,
        faces_dir: Path,
        reference: Path,
        output_dir: Path,
        on_progress: Callable[[int, int], None] | None = None,
        color_reference: Path | None = None,
    ) -> list[Path]:
        """Restyle every face image in ``faces_dir``; returns the results in frame order.

        ``reference`` supplies the hair shape. ``color_reference`` supplies the colour when
        given; without it the shape photo's colour comes along too, which is why choosing a
        bob used to force the reference's blonde.

        Images the face detector cannot handle are skipped rather than failing the run,
        because a subject who turns away mid-clip is normal input, not a fault.
        """
        missing = self.environment.missing_parts()
        if missing:
            raise RuntimeError(f"헤어스타일 변경을 쓸 수 없습니다. 준비되지 않은 항목: {', '.join(missing)}")

        output_dir.mkdir(parents=True, exist_ok=True)
        # The model resolves several of its own weight paths relative to the working
        # directory, so the worker has to run from inside the repository. Everything the
        # caller supplies is therefore passed as an absolute path.
        command = [
            str(self.environment.interpreter.resolve()), "-u", str(Path("jaeeun/hair_runner.py").resolve()),
            "--faces", str(faces_dir.resolve()),
            "--shape", str(reference.resolve()),
            "--color", str((color_reference or reference).resolve()),
            "--output", str(output_dir.resolve()),
            "--repo", str(self.environment.repo.resolve()),
            "--weights", str(self.environment.weights.resolve()),
        ]
        skipped: list[str] = []
        transcript: list[str] = []
        with subprocess.Popen(
            command,
            cwd=str(self.environment.repo.resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as worker:
            for line in worker.stdout:
                line = line.strip()
                transcript.append(line)
                if line.startswith("PROGRESS ") and on_progress:
                    done, total = line.split()[1:3]
                    on_progress(int(done), int(total))
                elif line.startswith("SKIP "):
                    skipped.append(line.split()[1])
            code = worker.wait()

        if code == 2:
            raise RuntimeError("얼굴을 찾을 수 있는 프레임이 없습니다. 인물이 정면을 향하는 화면을 사용해 주세요.")
        if code != 0:
            # Surface the worker's own last words; without them a failure inside the
            # separate environment is invisible from here.
            detail = " / ".join(line for line in transcript[-3:] if line) or "출력 없음"
            print("\n".join(transcript), file=sys.stderr)
            raise RuntimeError(f"헤어스타일 변경에 실패했습니다 (종료 코드 {code}): {detail}")
        self.last_skipped = skipped
        # The worker writes an aligned copy and a hair mask beside each result; neither is
        # a result itself.
        return sorted(
            path
            for path in output_dir.glob("*.png")
            if not path.stem.endswith(("_aligned", "_mask"))
        )
