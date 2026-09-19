"""Inference wrapper for the Korean personal-colour classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LABEL_ORDER = ("spring_warm", "summer_cool", "autumn_warm", "winter_cool")
LABEL_KO = {"spring_warm": "봄웜", "summer_cool": "여름쿨", "autumn_warm": "가을웜", "winter_cool": "겨울쿨"}


@dataclass(frozen=True)
class PersonalColorResult:
    label: str
    label_ko: str
    confidence: float
    probabilities: dict[str, float]


def _face_crop(image: Path, face_bbox: tuple[int, int, int, int] | None):
    from PIL import Image

    with Image.open(image).convert("RGB") as source:
        if face_bbox is None:
            return source.copy()
        x, y, width, height = face_bbox
        padding_x, padding_y = round(width * 0.25), round(height * 0.25)
        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(source.width, x + width + padding_x)
        bottom = min(source.height, y + height + padding_y)
        return source.crop((left, top, right, bottom))


def classify_personal_color(
    image: Path,
    checkpoint: Path,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> PersonalColorResult:
    """Classify a representative face crop using the published checkpoint."""
    import torch
    import timm
    from torchvision import transforms

    from PIL import Image

    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=False, num_classes=4)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    face = _face_crop(image, face_bbox)
    with face:
        tensor = transform(face).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).tolist()
    scores = dict(zip(LABEL_ORDER, (float(value) for value in probabilities)))
    label = max(scores, key=scores.get)
    return PersonalColorResult(label, LABEL_KO[label], scores[label], scores)