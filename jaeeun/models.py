"""Human and portrait segmentation adapter used by the current VACE workflow."""

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
