"""Configuration and end-to-end virtual-fitting orchestration."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

from .data import extract_frames, preprocess_image, sampled_fps
from .hairstyle import HairstyleTransfer
from .models import HumanParser, Sam2MaskPropagator, recolor_masked_region
from .video import frames_to_mp4, smooth_frames


@dataclass(frozen=True)
class PipelineConfig:
    workspace: Path = Path("artifacts/jaeeun")
    image_size: int = 512
    fps: int = 8
    parsing_model: str = "fashn-ai/fashn-human-parser"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_checkpoint: Path = Path("checkpoints/sam2/sam2.1_hiera_small.pt")
    inpainting_model: str = "checkpoints/diffusion/stable-diffusion-2-inpainting"

    def prepare(self) -> None:
        for name in ("inputs", "frames", "masks", "outputs", "reports"):
            (self.workspace / name).mkdir(parents=True, exist_ok=True)


class VirtualFittingPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.config.prepare()

    def prepare_image(self, source: Path) -> Path:
        return preprocess_image(source, self.config.workspace / "inputs" / "normalized.png", self.config.image_size)

    def prepare_video(self, source: Path) -> list[Path]:
        return extract_frames(source, self.config.workspace / "frames", self.config.fps)

    def parse_image(self, source: Path) -> dict[str, Path]:
        normalized = self.prepare_image(source)
        parser = HumanParser(self.config.parsing_model)
        return parser.save_results(normalized, self.config.workspace / "masks")

    def recolor_image(self, source: Path, part: str, color: str) -> Path:
        results = self.parse_image(source)
        if part not in results:
            raise ValueError(f"Detected masks do not contain {part!r}.")
        output = self.config.workspace / "outputs" / f"{part}_recolored.png"
        return recolor_masked_region(
            self.config.workspace / "inputs" / "normalized.png", results[part], color, output
        )

    def recolor_hair_and_outfit(self, source: Path, hair_color: str, outfit_color: str) -> Path:
        """Parse once, then apply hair and top colors to one combined result."""
        results = self.parse_image(source)
        missing = [part for part in ("hair", "top") if part not in results]
        if missing:
            raise ValueError(f"Detected masks do not contain: {', '.join(missing)}")
        normalized = self.config.workspace / "inputs" / "normalized.png"
        intermediate = self.config.workspace / "outputs" / "hair_step.png"
        output = self.config.workspace / "outputs" / "hair_and_outfit_recolored.png"
        recolor_masked_region(normalized, results["hair"], hair_color, intermediate)
        return recolor_masked_region(intermediate, results["top"], outfit_color, output)

    def _seed_masks(self, frame: Path, parts: Iterable[str]) -> dict[str, np.ndarray]:
        """Parse one reference frame and return a boolean mask per requested part.

        The frame is parsed at its own resolution rather than through ``prepare_image``,
        because the video masks must line up with the extracted frames pixel for pixel.
        """
        parser = HumanParser(self.config.parsing_model)
        class_map, labels = parser.predict(frame)
        by_name = {name.replace(" ", "_"): id for id, name in labels.items()}
        seeds = {}
        for part in parts:
            if part not in by_name:
                raise ValueError(f"The parser has no {part!r} class.")
            mask = class_map == by_name[part]
            if not mask.any():
                raise ValueError(f"{part!r} is not visible in the first frame of the video.")
            seeds[part] = mask
        return seeds

    def recolor_video(
        self,
        source: Path,
        colors: dict[str, str],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Recolor body parts across a clip and reassemble it into an MP4.

        ``colors`` maps a parsed part name such as ``"hair"`` to its target hex color.
        Each part is located once on the first frame and then tracked, so a mask stays
        the same object across the clip instead of being re-guessed frame by frame.
        """
        import cv2

        frames = self.prepare_video(source)
        if not frames:
            raise ValueError("No frames could be read from the video.")

        seeds = self._seed_masks(frames[0], colors)
        parts_by_id = {index: part for index, part in enumerate(seeds, start=1)}
        masks_dir = self.config.workspace / "video_masks"
        regions_dir = self.config.workspace / "video_regions"
        edited_dir = self.config.workspace / "video_edited"
        for directory in (masks_dir, regions_dir, edited_dir):
            directory.mkdir(parents=True, exist_ok=True)

        propagator = Sam2MaskPropagator(self.config.sam2_config, self.config.sam2_checkpoint)
        tracked = propagator.propagate_from_masks(
            self.config.workspace / "frames",
            {index: seeds[part] for index, part in parts_by_id.items()},
        )
        for frame_index, object_ids, logits in tracked:
            edited = edited_dir / f"frame_{frame_index:06d}.png"
            working = frames[frame_index]
            edited_region = None
            for position, object_id in enumerate(object_ids):
                part = parts_by_id[int(object_id)]
                mask = ((logits[position] > 0).cpu().numpy().squeeze().astype(np.uint8)) * 255
                mask_path = masks_dir / f"{part}_frame_{frame_index:06d}.png"
                cv2.imwrite(str(mask_path), mask)
                working = recolor_masked_region(working, mask_path, colors[part], edited)
                edited_region = mask if edited_region is None else np.maximum(edited_region, mask)
            cv2.imwrite(str(regions_dir / f"frame_{frame_index:06d}.png"), edited_region)
            if on_progress:
                on_progress(frame_index + 1, len(frames))

        smoothed_dir = self.config.workspace / "video_smoothed"
        smooth_frames(
            sorted(edited_dir.glob("frame_*.png")),
            smoothed_dir,
            region_paths=sorted(regions_dir.glob("frame_*.png")),
        )
        output = self.config.workspace / "outputs" / f"{'_'.join(colors)}_recolored.mp4"
        return frames_to_mp4(
            smoothed_dir, output, sampled_fps(source, self.config.fps), audio_source=source
        )

    def restyle_image(self, source: Path, reference: Path) -> Path:
        """Replace the hair in one photo with the hairstyle shown in ``reference``.

        Unlike recoloring, this regenerates the hair, so the result is the model's square
        aligned face crop rather than the original framing.
        """
        staging = self.config.workspace / "hairstyle_input"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        shutil.copy(source, staging / f"photo{source.suffix or '.png'}")

        results = HairstyleTransfer().restyle_folder(
            staging, reference, self.config.workspace / "hairstyle_output"
        )
        if not results:
            raise ValueError("사진에서 얼굴을 찾지 못했습니다.")
        # Named after the reference so trying a second hairstyle keeps the first result.
        output = self.config.workspace / "outputs" / f"hairstyle_{reference.stem}.png"
        shutil.copy(results[0], output)
        return output

    def restyle_video(
        self,
        source: Path,
        reference: Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Apply one hairstyle across a clip and reassemble it into an MP4.

        Every frame is restyled independently rather than tracked, because the hairstyle
        has to be redrawn for each head angle; measurements on this footage showed the
        results drifting slightly less than the source frames already do, so the sequence
        holds together without extra smoothing.
        """
        frames = self.prepare_video(source)
        if not frames:
            raise ValueError("영상에서 프레임을 읽지 못했습니다.")

        restyled_dir = self.config.workspace / "hairstyle_frames"
        if restyled_dir.exists():
            shutil.rmtree(restyled_dir)
        results = HairstyleTransfer().restyle_folder(
            self.config.workspace / "frames", reference, restyled_dir, on_progress
        )
        if not results:
            raise ValueError("얼굴이 보이는 프레임이 없습니다.")

        # Frames where the face was not found leave gaps, and ffmpeg needs an unbroken
        # sequence, so renumber what survived into a fresh folder.
        ordered_dir = self.config.workspace / "hairstyle_sequence"
        if ordered_dir.exists():
            shutil.rmtree(ordered_dir)
        ordered_dir.mkdir(parents=True)
        for index, path in enumerate(results):
            shutil.copy(path, ordered_dir / f"frame_{index:06d}.png")

        output = self.config.workspace / "outputs" / f"hairstyle_{reference.stem}.mp4"
        # No audio: dropped frames make the surviving run shorter than the original sound.
        return frames_to_mp4(ordered_dir, output, sampled_fps(source, self.config.fps))
