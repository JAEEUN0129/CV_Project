# Anchor-driven VACE experiments

This package keeps anchor creation separate from video generation so the image
editor can be changed without rewriting the VACE stage.

## Current status

- `manual`: ready. Copies an already approved anchor image.
- `flux`: FLUX.1 Kontext reference-guided inpainting worker included. It requires
  a separately installed environment and accepted Hugging Face model access.
- VACE: ready for prebuilt frame-aligned `mask-video` and `masked-video` inputs.
- Final hair-only compositing: not implemented yet.

## Select an anchor candidate

```bash
python -m jaeeun.vace.anchor_selector \
  --source /root/vace-test/source.mp4 \
  --output /root/vace-experiment/source-anchor-frame.png \
  --metadata /root/vace-experiment/anchor-frame.json
```

The first implementation ranks frames by Laplacian sharpness. A human should
still reject a frame with a turned or occluded face before generating an anchor.

## Run a manual-anchor experiment

The experiment creates the frame-aligned mask and gray-masked source when their
paths are omitted:

```bash
python -m jaeeun.vace.experiment \
  --source /root/vace-test/source.mp4 \
  --style-reference /root/vace-test/style2.png \
  --edit-type wave \
  --anchor-provider manual \
  --anchor /root/vace-test/anchor-wave.png \
  --output /root/vace-experiment/output-wave.mp4
```

Add `--dry-run` to create the anchor copy and `experiment.json` without starting
GPU inference. Supported edit contracts are `see_through_bangs`, `short_hair`,
`remove_bangs`, and `wave`.

## FLUX adapter contract

```bash
python -m jaeeun.vace.experiment \
  --source /root/vace-test/source.mp4 \
  --style-reference /root/vace-test/style2.png \
  --edit-type wave \
  --anchor-provider flux \
  --flux-python /root/flux-env/bin/python \
  --flux-worker /root/CV_Project/jaeeun/vace/flux_worker.py \
  --output /root/vace-experiment/output-wave.mp4
```

When `--anchor-mask` is omitted, the pipeline creates one from the selected frame
for the requested edit type. Install the worker environment separately:

```bash
python -m venv /root/flux-env
source /root/flux-env/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
pip install -r /root/CV_Project/jaeeun/requirements-flux.txt
```

The model is gated; authenticate with Hugging Face and accept its license before
the first download. The worker belongs in the isolated FLUX environment, not the
working VACE Python environment.
