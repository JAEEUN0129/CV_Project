"""Manual colour-mask correction without rerunning hairstyle generation."""

import base64
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def apply_strokes(base: np.ndarray, strokes: list) -> np.ndarray:
    if len(strokes) > 500:
        raise ValueError("브러시 횟수가 너무 많습니다. 초기화 후 다시 보정해주세요.")
    image = Image.fromarray(base.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for stroke in strokes:
        radius = float(stroke["radius"])
        points = stroke["points"]
        if not 0 < radius <= .1 or not 1 <= len(points) <= 10000 or stroke["mode"] not in {"add", "erase"}:
            raise ValueError("올바르지 않은 브러시 입력입니다.")
        coordinates = []
        for x, y in points:
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError("브러시가 이미지 범위를 벗어났습니다.")
            coordinates.append((round(x * (width - 1)), round(y * (height - 1))))
        r = max(1, round(radius * min(width, height)))
        fill = 255 if stroke["mode"] == "add" else 0
        draw.line(coordinates, fill=fill, width=2 * r + 1)
        for x, y in coordinates:
            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill)
    return np.array(image) > 0


def render_mask_editor(source: Path, mask: Path, strokes: list, key: str):
    import streamlit.components.v1 as components

    editor = components.declare_component("hair_mask_brush", path=str(Path(__file__).parent / "mask_brush"))
    def uri(path):
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return editor(image=uri(source), mask=uri(mask), strokes=strokes,
                  identity=str(source), key=key, default=None)
