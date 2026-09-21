"""Adapters for human parsing, SAM 2 video propagation and inpainting."""

from pathlib import Path

import numpy as np


class HumanParser:
    def __init__(self, model_id: str = "fashn-ai/fashn-human-parser") -> None:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict(self, image: Path, *, return_probabilities: bool = False):
        """Return a class-ID mask at the original image resolution and ID labels."""
        import torch
        from PIL import Image

        source = Image.open(image).convert("RGB")
        inputs = self.processor(images=source, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(source.height, source.width), mode="bilinear", align_corners=False
        )
        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        labels = {int(key): value for key, value in self.model.config.id2label.items()}
        if return_probabilities:
            return mask, labels, logits.softmax(dim=1)[0].float().cpu().numpy()
        return mask, labels

    def save_results(self, image: Path, output_dir: Path) -> dict[str, Path]:
        """Save the class map, color overlay, and one binary mask per detected class."""
        from PIL import Image

        output_dir.mkdir(parents=True, exist_ok=True)
        class_map, labels = self.predict(image)
        source = np.asarray(Image.open(image).convert("RGB"))
        rng = np.random.default_rng(42)
        palette = rng.integers(40, 256, size=(max(labels) + 1, 3), dtype=np.uint8)
        palette[0] = 0
        color_mask = palette[class_map]
        overlay = (source.astype(np.float32) * 0.55 + color_mask * 0.45).astype(np.uint8)

        paths = {
            "class_map": output_dir / "class_map.png",
            "overlay": output_dir / "overlay.png",
        }
        Image.fromarray(class_map).save(paths["class_map"])
        Image.fromarray(overlay).save(paths["overlay"])
        for class_id in np.unique(class_map):
            label = labels.get(int(class_id), f"class_{class_id}").replace(" ", "_")
            path = output_dir / f"mask_{int(class_id):02d}_{label}.png"
            Image.fromarray(np.where(class_map == class_id, 255, 0).astype(np.uint8)).save(path)
            paths[label] = path
        return paths


class Sam2MaskPropagator:
    def __init__(self, config_path: str, checkpoint: Path) -> None:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = build_sam2_video_predictor(config_path, str(checkpoint), device=device)

    def propagate(self, frames_dir: Path, prompts: list[dict]):
        state = self.predictor.init_state(video_path=str(frames_dir))
        for prompt in prompts:
            self.predictor.add_new_points_or_box(inference_state=state, **prompt)
        yield from self.predictor.propagate_in_video(state)

    def propagate_from_masks(
        self, frames_dir: Path, seed_masks: dict[int, np.ndarray], frame_index: int = 0
    ):
        """Propagate parser masks through the clip, one object id per body part.

        Seeding with the parser mask rather than clicks keeps the boundary the parser
        already found, so the tracker only has to follow it instead of re-deriving it
        from a point that may land on the wrong side of a thin region such as hair.

        All parts are tracked in a single pass because the expensive per-frame image
        encoding is shared across objects; tracking them separately would repeat it.
        """
        state = self.predictor.init_state(video_path=str(frames_dir))
        for object_id, seed_mask in seed_masks.items():
            self.predictor.add_new_mask(
                inference_state=state, frame_idx=frame_index, obj_id=object_id, mask=seed_mask.astype(bool)
            )
        yield from self.predictor.propagate_in_video(state)


class InpaintingEditor:
    def __init__(self, model_id: str) -> None:
        import torch
        from diffusers import AutoPipelineForInpainting

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        # Only the half-precision weights are downloaded (see download_models.py), so the
        # variant must be requested explicitly. On CPU they are upcast to float32, which
        # costs nothing at load time and avoids half-precision math the CPU handles badly.
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16"
        ).to(device)

    def edit(self, image: Path, mask: Path, prompt: str, output: Path, seed: int = 42) -> Path:
        import torch
        from PIL import Image

        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            image=Image.open(image).convert("RGB"),
            mask_image=Image.open(mask).convert("L"),
            generator=generator,
        ).images[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        return output


def recolor_masked_region(
    image: Path, mask: Path, color: str, output: Path, strength: float = 0.72
) -> Path:
    """Recolor a region while retaining its original luminance and texture."""
    import cv2

    source = cv2.imread(str(image), cv2.IMREAD_COLOR)
    region = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if source is None or region is None:
        raise ValueError("Image or mask could not be read.")
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    target = np.full_like(source, rgb[::-1], dtype=np.uint8)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    target_lab[..., 0] = source_lab[..., 0]
    recolored = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
    alpha = cv2.GaussianBlur(region, (0, 0), 1.2).astype(np.float32) / 255.0
    alpha = (alpha * strength)[..., None]
    result = (source * (1 - alpha) + recolored * alpha).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result)
    return output
