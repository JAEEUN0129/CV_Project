"""CLI entry point for anchor-driven VACE comparison experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchor_editor import FluxAnchorEditor, ManualAnchorEditor
from .anchor_mask import build_anchor_mask
from .anchor_selector import select_anchor
from .runner import VaceRun
from .short_hair_mask import build_short_hair_mask_video
from .specs import EDIT_SPECS
from ..prepare_vace_mask_video import build_mask_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--style-reference", type=Path, required=True)
    parser.add_argument("--edit-type", choices=EDIT_SPECS, required=True)
    parser.add_argument("--anchor-provider", choices=("manual", "flux"), default="manual")
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--anchor-mask", type=Path)
    parser.add_argument("--mask-video", type=Path)
    parser.add_argument("--masked-video", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path("/root/vace-experiment"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vace-repo", type=Path, default=Path("/root/VACE"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/models/Wan2.1-VACE-1.3B"))
    parser.add_argument("--flux-python", type=Path, default=Path("/root/flux-env/bin/python"))
    parser.add_argument("--flux-worker", type=Path)
    parser.add_argument(
        "--flux-edit-mode", choices=("masked", "reference_only"), default="masked"
    )
    parser.add_argument("--frame-num", type=int, default=41)
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--guide-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (args.source, args.style_reference):
        if not path.exists():
            parser.error(f"required input does not exist: {path}")
    if args.frame_num < 1 or (args.frame_num - 1) % 4:
        parser.error("--frame-num must have the form 4n+1")

    spec = EDIT_SPECS[args.edit_type]
    args.workspace.mkdir(parents=True, exist_ok=True)
    selected_frame = args.workspace / "source-anchor-frame.png"
    selected = select_anchor(args.source, selected_frame)
    final_anchor = args.workspace / f"anchor-{args.edit_type}.png"
    anchor_mask = args.anchor_mask or args.workspace / f"anchor-mask-{args.edit_type}.png"

    if args.anchor_provider == "manual":
        if args.anchor is None:
            parser.error("--anchor is required for the manual provider")
        editor = ManualAnchorEditor(args.anchor)
    else:
        if args.flux_worker is None:
            parser.error("--flux-worker is required for the flux provider")
        if args.flux_edit_mode == "masked" and args.anchor_mask is None:
            build_anchor_mask(selected_frame, anchor_mask, args.edit_type)
        editor = FluxAnchorEditor(
            args.flux_python, args.flux_worker, edit_mode=args.flux_edit_mode
        )
    editor_mask = anchor_mask
    if args.anchor_provider == "flux" and args.flux_edit_mode == "reference_only":
        editor_mask = None
    editor.create(selected_frame, args.style_reference, editor_mask, spec.prompt, final_anchor)

    mask_video = args.mask_video or args.workspace / f"mask-{args.edit_type}.mp4"
    masked_video = args.masked_video or args.workspace / f"source-masked-{args.edit_type}.mp4"
    if args.mask_video is None or args.masked_video is None:
        if args.edit_type == "short_hair":
            build_short_hair_mask_video(
                args.source,
                final_anchor,
                mask_video,
                masked_video,
                temporal_window=3,
                target_output=args.workspace / "target-mask-short_hair.mp4",
                target_anchor_output=args.workspace / "anchor-target-mask-short_hair.png",
            )
        else:
            ratio = {
                "see_through_bangs": 0.32,
                "remove_bangs": 0.25,
            }.get(args.edit_type, 0.28)
            bangs_style = "curtain" if args.edit_type == "curtain_bangs" else "straight"
            build_mask_video(
                args.source,
                mask_video,
                masked_video,
                forehead_ratio=ratio,
                temporal_window=3,
                bangs_style=bangs_style,
                edit_type=args.edit_type,
            )

    run = VaceRun(
        repo=args.vace_repo,
        checkpoint=args.checkpoint,
        source_video=masked_video,
        mask_video=mask_video,
        anchor=final_anchor,
        output=args.output,
        prompt=spec.prompt,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )
    metadata = {
        "edit_type": spec.name,
        "mask_strategy": spec.mask_strategy,
        "anchor_provider": args.anchor_provider,
        "flux_edit_mode": args.flux_edit_mode if args.anchor_provider == "flux" else None,
        "anchor_frame_index": selected.frame_index,
        "anchor_time_seconds": selected.timestamp_seconds,
        "anchor_face_bbox": selected.face_bbox,
        "anchor_sharpness": selected.sharpness,
        "source_fps": selected.fps,
        "seed": args.seed,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "guide_scale": args.guide_scale,
        "command": run.command(),
    }
    (args.workspace / "experiment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(run.printable_command())
    if not args.dry_run:
        run.execute()


if __name__ == "__main__":
    main()
