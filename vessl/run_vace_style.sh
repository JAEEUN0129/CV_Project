#!/usr/bin/env bash
set -euo pipefail

STYLE_ID="${1:-}"
BANGS_STYLE="${2:-straight}"
TEST_DIR="${VACE_TEST_DIR:-/root/vace-test}"
VACE_DIR="${VACE_DIR:-/root/VACE}"
MODEL_DIR="${VACE_MODEL_DIR:-/root/models/Wan2.1-VACE-1.3B}"
PROJECT_DIR="${PROJECT_DIR:-/root/CV_Project}"
DIRECT_INFERENCE=0

case "$STYLE_ID" in
  1)
    STYLE_IMAGE="$TEST_DIR/style1.jpg"
    EDIT_MASK="$TEST_DIR/hair-mask.png"
    PROMPT="A photorealistic video of the same person from the source video with long straight hair matching the reference image. Preserve her identity, eyes, eyebrows, nose, mouth, facial expression, clothing, pose, lighting, camera motion, and background. Change only the hair, with natural strands and temporally consistent motion."
    ;;
  2)
    STYLE_IMAGE="$TEST_DIR/style2.png"
    EDIT_MASK="$TEST_DIR/hair-mask-wavy.png"
    cd "$PROJECT_DIR"
    python -m jaeeun.prepare_vace_mask \
      --source "$TEST_DIR/source.mp4" \
      --base-mask "$TEST_DIR/hair-mask.png" \
      --output "$EDIT_MASK" \
      --wavy \
      --side-pixels 40 \
      --down-pixels 30
    PROMPT="A photorealistic video of the same person from the source video with long voluminous wavy hair matching the reference image. The hair has clearly defined soft S-shaped waves from the mid-lengths continuously through the lower lengths and ends; the bottom sections and tips remain visibly wavy and full, never straight or flat. Preserve her identity, eyes, eyebrows, nose, mouth, facial expression, clothing, pose, lighting, camera motion, and background. Change only the hair, with natural strands and temporally consistent waves."
    ;;
  3)
    case "$BANGS_STYLE" in
      straight)
        STYLE_PHRASE="full straight bangs"
        ;;
      choppy)
        STYLE_PHRASE="short choppy bangs"
        ;;
      side)
        STYLE_PHRASE="side-swept bangs"
        ;;
      *)
        echo "Unknown bangs style: $BANGS_STYLE" >&2
        echo "Usage: bash $0 3 {straight|choppy|side}" >&2
        exit 2
        ;;
    esac
    STYLE_IMAGE="$TEST_DIR/style3.jpg"
    EDIT_MASK="$TEST_DIR/hair-mask-bangs.mp4"
    MASKED_VIDEO="$TEST_DIR/source-bangs-masked.mp4"
    cd "$PROJECT_DIR"
    python -m jaeeun.prepare_vace_mask_video \
      --source "$TEST_DIR/source.mp4" \
      --output "$EDIT_MASK" \
      --masked-video-output "$MASKED_VIDEO" \
      --forehead-ratio 0.28 \
      --temporal-window 3
    DIRECT_INFERENCE=1
    PROMPT="Edit this video so that the person has $STYLE_PHRASE. Change only the bangs area in every frame. Keep the rest of the hairstyle, overall hair length, hair color, face identity, skin, glasses, clothing, pose, lighting, camera motion, and background unchanged. Make the new bangs clearly recognizable, natural-looking, and temporally consistent throughout the video."
    ;;
  4)
    STYLE_IMAGE="$TEST_DIR/style4.png"
    EDIT_MASK="$TEST_DIR/hair-mask.png"
    PROMPT="A photorealistic video of the same person from the source video with a short hairstyle matching the reference image. Remove the original long hair inside the editable region and naturally reconstruct the revealed background, neck, and clothing. Preserve her identity, eyes, eyebrows, nose, mouth, facial expression, clothing, pose, lighting, camera motion, and background. Change only the hair, with temporally consistent motion."
    ;;
  *)
    echo "Usage: bash $0 {1|2|3|4}" >&2
    exit 2
    ;;
esac

for required in "$TEST_DIR/source.mp4" "$TEST_DIR/hair-mask.png" "$STYLE_IMAGE" "$MODEL_DIR"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p /root/pip-tmp /root/pip-cache
export TMPDIR=/root/pip-tmp
export PIP_CACHE_DIR=/root/pip-cache

cd "$VACE_DIR"
if [[ "$DIRECT_INFERENCE" == "1" ]]; then
  python vace/vace_wan_inference.py \
    --ckpt_dir "$MODEL_DIR" \
    --src_video "$MASKED_VIDEO" \
    --src_mask "$EDIT_MASK" \
    --src_ref_images "$STYLE_IMAGE" \
    --prompt "$PROMPT"
else
  python vace/vace_pipeline.py \
    --base wan \
    --task swap_anything \
    --mode masktrack,plain \
    --video "$TEST_DIR/source.mp4" \
    --mask "$EDIT_MASK" \
    --image "$STYLE_IMAGE" \
    --ckpt_dir "$MODEL_DIR" \
    --prompt "$PROMPT"
fi

echo "Style $STYLE_ID finished. Results are under $VACE_DIR/results."
