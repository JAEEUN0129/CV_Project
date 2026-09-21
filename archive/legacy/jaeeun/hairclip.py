"""Isolated text-to-hair-colour reference generation with HairCLIP."""

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class HairClipEnvironment:
    """Paths used by the standalone HairCLIP worker.

    HairCLIP predates the rest of this project and imports packages with names that
    collide with HairFastGAN/e4e.  Keeping it in a subprocess prevents those imports
    and its CUDA compatibility shims from leaking into the Streamlit process.
    """

    interpreter: Path | None = None
    repo: Path = Path("external/HairCLIP")
    checkpoint: Path = Path("external/HairCLIP/pretrained_models/hairclip.pt")
    clip_checkpoint: Path = Path("external/HairCLIP/pretrained_models/ViT-B-32.pt")
    e4e_checkpoint: Path = Path(
        "external/HairFastGAN_weights/pretrained_models/encoder4editing/e4e_ffhq_encode.pt"
    )
    landmark_model: Path = Path(
        "external/HairFastGAN_weights/pretrained_models/ShapeAdaptor/"
        "shape_predictor_68_face_landmarks.dat"
    )

    def resolve_interpreter(self) -> Path:
        override = os.environ.get("HAIRCLIP_PYTHON")
        if override:
            return Path(override)
        if self.interpreter is not None:
            return self.interpreter
        for candidate in (
            Path(".venv_clip/Scripts/python.exe"),
            Path(".venv_clip/bin/python"),
            Path(".venv_hair/Scripts/python.exe"),
            Path(".venv_hair/bin/python"),
        ):
            if candidate.exists():
                return candidate
        return Path(sys.executable)

    def missing_parts(self) -> list[str]:
        required = {
            "HairCLIP 실행 환경": self.resolve_interpreter(),
            "HairCLIP 코드": self.repo,
            "HairCLIP 가중치": self.checkpoint,
            "CLIP 텍스트 인코더": self.clip_checkpoint,
            "e4e 가중치": self.e4e_checkpoint,
            "얼굴 정렬 모델": self.landmark_model,
            "e4e 코드": self.repo / "encoder4editing",
        }
        return [name for name, path in required.items() if not path.exists()]


class HairClipColorEditor:
    def __init__(self, environment: HairClipEnvironment | None = None) -> None:
        self.environment = environment or HairClipEnvironment()

    def edit(self, source: Path, prompt: str, output: Path) -> Path:
        """Create a colour-reference portrait from ``source`` and an English prompt."""
        prompt = normalize_color_prompt(prompt)
        missing = self.environment.missing_parts()
        if missing:
            raise RuntimeError("HairCLIP을 사용할 수 없습니다: " + ", ".join(missing))
        output.parent.mkdir(parents=True, exist_ok=True)
        # dlib on Windows cannot open a path containing Korean characters.  Stage the
        # reference under the ASCII-only system temp directory before face alignment.
        staged_dir = Path(tempfile.mkdtemp(prefix="hairclip_"))
        staged_source = staged_dir / ("source" + (source.suffix.lower() or ".png"))
        shutil.copy(source, staged_source)
        staged_models = Path(tempfile.gettempdir()) / "hairclip_models"
        staged_models.mkdir(exist_ok=True)
        staged_landmark = staged_models / "shape_predictor_68_face_landmarks.dat"
        if (
            not staged_landmark.exists()
            or staged_landmark.stat().st_size != self.environment.landmark_model.stat().st_size
        ):
            shutil.copy(self.environment.landmark_model, staged_landmark)
        command = [
            str(self.environment.resolve_interpreter().resolve()),
            "-u",
            str(Path("jaeeun/hairclip_runner.py").resolve()),
            "--source", str(staged_source.resolve()),
            "--prompt", prompt,
            "--output", str(output.resolve()),
            "--repo", str(self.environment.repo.resolve()),
            "--checkpoint", str(self.environment.checkpoint.resolve()),
            "--clip-checkpoint", str(self.environment.clip_checkpoint.resolve()),
            "--e4e-checkpoint", str(self.environment.e4e_checkpoint.resolve()),
            "--landmark-model", str(staged_landmark.resolve()),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.environment.repo.resolve()),
                capture_output=True,
                text=True,
            )
        finally:
            shutil.rmtree(staged_dir, ignore_errors=True)
        if completed.returncode != 0:
            detail = " / ".join(
                line.strip() for line in (completed.stdout + completed.stderr).splitlines()[-15:]
                if line.strip()
            )
            raise RuntimeError("HairCLIP 색상 참조 생성에 실패했습니다: " + (detail or "출력 없음"))
        if not output.exists():
            raise RuntimeError("HairCLIP이 결과 파일을 만들지 않았습니다.")
        return output


_KOREAN_COLORS = {
    "애쉬 브라운": "gray brown", "애쉬브라운": "gray brown",
    "초코 브라운": "dark brown", "초코브라운": "dark brown",
    "초콜릿 브라운": "dark brown", "초콜릿브라운": "dark brown",
    "카키 브라운": "green brown", "카키브라운": "green brown",
    "레드 브라운": "red brown", "레드브라운": "red brown",
    "오렌지 브라운": "orange brown", "오렌지브라운": "orange brown",
    "밀크티 브라운": "light brown", "밀크티브라운": "light brown",
    "다크 브라운": "dark brown", "다크브라운": "dark brown",
    "로즈 골드": "pink blond", "로즈골드": "pink blond",
    "블루 블랙": "blue black", "블루블랙": "blue black",
    "와인 레드": "burgundy red", "와인레드": "burgundy red",
    "검정": "black", "블랙": "black", "갈색": "brown", "브라운": "brown",
    "플래티넘 블론드": "white blond", "플래티넘": "white blond",
    "금발": "blond", "블론드": "blond", "백금발": "white blond",
    "빨강": "red", "빨간색": "red", "레드": "red", "와인": "burgundy red",
    "주황": "orange", "오렌지": "orange", "노랑": "yellow", "옐로우": "yellow",
    "초록": "green", "그린": "green", "파랑": "blue", "블루": "blue",
    "실버": "gray", "회색": "gray", "그레이": "gray", "은색": "gray",
    "버건디": "burgundy red", "코랄": "pink orange", "베이지": "light blond",
    "네이비": "dark blue", "핑크": "pink",
    "분홍": "pink", "보라": "purple", "퍼플": "purple",
    "애쉬": "gray", 
}

_KOREAN_MODIFIERS = {
    "차가운": "cool", "쿨한": "cool", "따뜻한": "warm", "웜한": "warm",
    "자연스러운": "natural", "밝은": "light", "어두운": "dark",
    "선명한": "vivid", "은은한": "soft", "빛이 도는": "tinted",
}

_KOREAN_STYLE_WORDS = (
    "앞머리 웨이브", "긴 웨이브", "짧은 웨이브", "긴 생머리", "짧은 머리",
    "레이어드 컷", "레이어드컷", "허쉬 컷", "허쉬컷", "보브 컷", "보브컷",
    "헤어스타일", "스타일", "앞머리", "생머리", "웨이브", "단발", "장발",
)

_KOREAN_FILLERS = (
    "머리카락", "머리 색상", "머리색", "머리", "색상", "색깔", "색",
    "으로 염색해 주세요", "으로 염색해주세요", "로 염색해 주세요", "로 염색해주세요",
    "으로 바꿔 주세요", "으로 바꿔주세요", "로 바꿔 주세요", "로 바꿔주세요",
    "으로 해 주세요", "으로 해주세요", "로 해 주세요", "로 해주세요",
    "으로 해줘", "로 해줘", "하고 싶어요", "하고 싶어", "부탁해요", "부탁해",
    "해주세요", "해 주세요", "해줘", " 에 ", " 은 ", " 는 ", " 을 ", " 를 ",
)


def normalize_color_prompt(prompt: str) -> str:
    """Map common Korean colour descriptions to HairCLIP's English text domain."""
    normalized = " ".join(prompt.strip().lower().split())
    if not normalized:
        raise ValueError("머리색을 텍스트로 입력해주세요.")
    for phrase in sorted(_KOREAN_STYLE_WORDS, key=len, reverse=True):
        normalized = normalized.replace(phrase, " ")
    for korean in sorted(_KOREAN_COLORS, key=len, reverse=True):
        normalized = normalized.replace(korean, _KOREAN_COLORS[korean])
    for korean in sorted(_KOREAN_MODIFIERS, key=len, reverse=True):
        normalized = normalized.replace(korean, _KOREAN_MODIFIERS[korean])
    for phrase in sorted(_KOREAN_FILLERS, key=len, reverse=True):
        normalized = normalized.replace(phrase, " ")
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,.")
    if not normalized:
        raise ValueError("텍스트에 원하는 머리색을 함께 입력해주세요.")
    if any("가" <= char <= "힣" for char in normalized):
        raise ValueError(
            "HairCLIP이 이해할 수 없는 한국어 표현입니다. 색 이름을 단순하게 쓰거나 영어로 입력해주세요."
        )
    return normalized.removesuffix(" hair").strip() + " hair"
