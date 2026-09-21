"""Consumer-facing Streamlit editor for the three-stage hair workflow."""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import uuid
import subprocess
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaeeun.personal_color import classify_personal_color
from jaeeun.vace.anchor_mask import build_anchor_mask
from jaeeun.vace.anchor_selector import select_anchor
from jaeeun.vace.color_candidates import PERSONAL_COLOR_PALETTES, build_color_candidates
from jaeeun.vace.runner import VaceRun
from jaeeun.vace.style_options import STYLE_OPTIONS, COLOR_OPTIONS, inputs_ready, hairstyle_prompt, requested_style_label
from jaeeun.prepare_vace_mask_video import build_mask_video


COLOR_LABELS = {
    "spring_warm": "봄 웜",
    "summer_cool": "여름 쿨",
    "autumn_warm": "가을 웜",
    "winter_cool": "겨울 쿨",
}
COLOR_META = {
    "requested": ("요청한 스타일", "요청한 스타일", "#6758d8"),
    "honey_brown": ("Honey Brown", "허니 브라운", "#B8794E"),
    "caramel_brown": ("Caramel Brown", "캐러멜 브라운", "#A96C43"),
    "peach_brown": ("Peach Brown", "피치 브라운", "#C88467"),
    "ash_brown": ("Ash Brown", "애쉬 브라운", "#82746F"),
    "rose_brown": ("Rose Brown", "로즈 브라운", "#9A6F78"),
    "blue_black": ("Blue Black", "블루 블랙", "#202A3A"),
    "chocolate_brown": ("Chocolate Brown", "초콜릿 브라운", "#633F32"),
    "copper_brown": ("Copper Brown", "코퍼 브라운", "#A75E3B"),
    "chestnut_brown": ("Chestnut Brown", "체스트넛 브라운", "#875035"),
    "deep_black": ("Deep Black", "딥 블랙", "#17191D"),
    "burgundy": ("Burgundy", "버건디", "#713B4E"),
    "violet_black": ("Violet Black", "바이올렛 블랙", "#30263B"),
}


def save_upload(upload, suffix: str | None = None) -> Path:
    extension = suffix or Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as handle:
        handle.write(upload.getbuffer())
        return Path(handle.name)


def image_uri(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def color_meta(color_id: str) -> tuple[str, str, str]:
    if color_id == "requested":
        label = requested_style_label(st.session_state.get("style_options", {}), bool(st.session_state.get("reference_path")))
        return label, label, "#6758d8"
    custom = {"black": ("Black", "블랙", "#17191D"), "brown": ("Brown", "브라운", "#633F32"), "gray": ("Gray", "그레이", "#969696"), "red": ("Red", "레드", "#A53737")}
    return COLOR_META.get(color_id, custom.get(color_id, (color_id, color_id, "#8A8F98")))


def job_workspace() -> Path:
    return Path(st.session_state.job_workspace)


def active_palette() -> tuple:
    return PERSONAL_COLOR_PALETTES[st.session_state.personal_color_result["label"]]


def clear_from_upload_change() -> None:
    for key in ("representative_frame", "personal_color_result", "vace_candidates", "selected_anchor", "selected_preview", "video_result"):
        st.session_state.pop(key, None)
    st.session_state.phase = "upload"


def phase_number() -> int:
    return {"upload": 1, "color": 2, "video": 3}[st.session_state.phase]


def get_representative_frame() -> Path:
    frame_path = job_workspace() / "source-anchor-frame.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_path.exists():
        select_anchor(Path(st.session_state.source_path), frame_path)
    st.session_state.representative_frame = str(frame_path)
    return frame_path


def create_anchor_candidates() -> None:
    workspace = job_workspace()
    frame = get_representative_frame()
    edit_type = st.session_state.get("edit_type", "wave")
    mask = workspace / f"anchor-mask-{edit_type}.png"
    build_anchor_mask(frame, mask, edit_type, target_length=st.session_state.get("style_options", {}).get("length"))
    generated = build_color_candidates(
        frame,
        Path(st.session_state.reference_path) if st.session_state.get("reference_path") else None,
        mask,
        workspace / "anchors",
        st.session_state.personal_color_result["label"],
        Path(os.environ.get("FLUX_PYTHON", "/root/flux-env/bin/python")),
        Path(os.environ.get("FLUX_WORKER", str(Path(__file__).parent / "vace" / "flux_worker.py"))),
        st.session_state.style_prompt,
        palette=active_palette(),
        include_requested=True,
        requested_color=st.session_state.get("style_options", {}).get("color"),
        requested_options=st.session_state.get("style_options", {}),
    )
    st.session_state.vace_candidates = [
        {"id": item.color_id, "name": item.name, "prompt_color": item.prompt_color, "path": str(item.anchor)}
        for item in generated
    ]
    st.session_state.selected_anchor = "requested"
    st.session_state.selected_preview = st.session_state.selected_anchor


def generate_video() -> None:
    workspace = job_workspace()
    selected = next(item for item in st.session_state.vace_candidates if item["id"] == st.session_state.selected_anchor)
    mask_video = workspace / "hair-mask.mp4"
    masked_video = workspace / "source-masked.mp4"
    build_mask_video(Path(st.session_state.source_path), mask_video, masked_video, edit_type=st.session_state.edit_type, forehead_ratio=0.32)
    prompt = (
        f"Match the hairstyle shape, length, texture and bangs of the supplied anchor image. The final hair colour is {selected['prompt_color']}. "
        "Preserve the person's identity, face, skin, clothing, lighting, pose, camera motion, and background. "
        "Change only the hair inside the mask and keep the colour temporally consistent."
    )
    output = workspace / f"result-{selected['id']}.mp4"
    VaceRun(
        repo=Path(os.environ.get("VACE_REPO", "/root/VACE")),
        checkpoint=Path(os.environ.get("VACE_MODEL_DIR", "/root/vace-model")),
        source_video=masked_video,
        mask_video=mask_video,
        anchor=Path(selected["path"]),
        output=output,
        prompt=prompt,
    ).execute()
    st.session_state.video_result = str(output)


st.set_page_config(page_title="AI Hair Studio", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      :root{--ink:#18181b;--muted:#71717a;--line:#e4e4e7;--canvas:#eef0f3;--accent:#6758d8;--studio-header-height:3.5rem}
    html,
    body,
    [data-testid="stAppViewContainer"]{
      margin:0!important;
      padding:0!important;
    }

    .stApp{
      background:#f7f7f8;
      color:var(--ink);
      height:100vh;
      overflow:hidden;
      margin:0!important;
      padding:0!important;
    }

    /* Streamlit 기본 상단 여백 제거 */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer,
    .main,
    section.main,
    .block-container{
      margin-top:0!important;
      padding-top:0!important;
    }

    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer,
    .block-container{
      max-width:none!important;
      width:100%!important;
      height:100vh!important;
      min-height:0!important;
      margin:0!important;
      padding:0!important;
      overflow:hidden;
    }

    /* Streamlit 기본 header/toolbar 공간 제거 */
    header[data-testid="stHeader"]{
      display:none!important;
      height:0!important;
      min-height:0!important;
    }

    [data-testid="stToolbar"]{
      display:none!important;
    }

    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]{gap:0!important}
    /* Keep both panels inside one viewport below the header. */
    .st-key-studio_body{
      position:fixed!important;
      top:var(--studio-header-height);
      bottom:0;
      left:0;
      right:0;
      overflow:hidden;
      min-height:0!important;
      padding-block:1.25rem!important;
      box-sizing:border-box!important;
    }
    .st-key-studio_body > [data-testid="stHorizontalBlock"]{
      height:100%;
      min-height:0;
      gap:0!important;
      align-items:stretch;
      overflow:hidden;
    }
    .st-key-studio_body > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
      height:100%;
      min-height:0!important;
      overflow-y:auto;
      overflow-x:hidden;
      overscroll-behavior-y:contain;
      scrollbar-gutter:stable;
      padding:0 1rem!important;
      box-sizing:border-box;
    }
    .st-key-studio_body > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child{
      background:var(--canvas);
      border-left:1px solid var(--line);
    }
    /* 왼쪽 패널에만 여백 적용 */
    [data-testid="stColumn"]:has(.st-key-left_rail){
      padding-left:1.4rem!important;
      padding-right:1rem!important;
      box-sizing:border-box!important;
    }
    .st-key-studio_body > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-left_rail){
      padding-top:0!important;
      padding-bottom:0!important;
    }
    [data-testid="stColumn"]:has(.st-key-left_rail) > [data-testid="stVerticalBlock"]{
      min-height:100%;
      gap:0!important;
    }
    .st-key-left_rail{
      flex-shrink:0;
      margin-block:auto;
    }
    .st-key-left_rail .reference-upload-label,
    .st-key-left_rail [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p{
      font-size:.875rem!important;
      font-weight:400;
      line-height:1.5;
      margin:0;
    }
    .st-key-left_rail .reference-upload-help{
      font-size:.7rem!important;
      line-height:1.4;
      color:var(--muted);
      margin:.2rem 0 0;
    }

    /* 왼쪽 컨트롤 패널 전용 여백 */
    .st-key-left_rail{
      padding-left:0!important;
      padding-right:0!important;
      padding-top:0!important;
      padding-bottom:0;
      box-sizing:border-box!important;
    }
    .studio-header{
      position:fixed;
      top:0;
      left:0;
      z-index:100;
      height:var(--studio-header-height);
      width:100%;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:0;
      border-bottom:1px solid var(--line);
      background:#fff;
      padding:0 1.25rem;
      box-sizing:border-box;
      margin-top:0!important;
    }.brand{display:flex;align-items:center;gap:.7rem}.brand-name{font-size:1rem;font-weight:750}.brand-subtitle{color:var(--muted);font-size:.68rem;margin-left:.25rem}
    .stepper{display:flex;align-items:center;gap:.5rem;color:#a1a1aa;font-size:.68rem;transform:none;margin-left:auto}.step{display:flex;align-items:center;gap:.3rem;white-space:nowrap}.step-dot{width:1.2rem;height:1.2rem;border-radius:50%;display:grid;place-items:center;background:#e4e4e7;color:#71717a;font-size:.6rem;font-weight:700}.step.active{color:var(--ink);font-weight:650}.step.active .step-dot{background:var(--accent);color:#fff}.step-line{width:1rem;height:1px;background:var(--line)}
    .rail{background:#fff;border-right:1px solid var(--line);padding:.7rem .9rem;min-height:calc(100vh - 3.5rem);overflow:visible}.rail-kicker{color:var(--accent);text-transform:uppercase;letter-spacing:.1em;font-size:.58rem;font-weight:800}.rail-copy{color:var(--muted);font-size:.76rem;line-height:1.45;margin:.42rem 0 .7rem}.section{border-top:1px solid #f0f0f2;padding:.45rem 0}.section-title{font-size:.68rem;font-weight:750;margin-bottom:.3rem}.color-section-title{margin-top:.1rem}.section-number{color:var(--accent);margin-right:.3rem}.helper{color:var(--muted);font-size:.62rem;line-height:1.35}.upload-summary{background:#f7f7f8;border:1px solid var(--line);border-radius:.6rem;padding:.4rem .5rem;font-size:.68rem}.upload-summary strong{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.upload-summary span{color:#35a06b;font-size:.62rem}
      .color-result{background:#f5f3ff;border:1px solid #ded9ff;border-radius:.75rem;padding:.8rem}.color-result strong{display:block;font-size:1rem}.confidence{color:var(--muted);font-size:.68rem;margin-top:.2rem}.confidence b{color:var(--accent)}.swatch{display:inline-block;width:.85rem;height:.85rem;border-radius:50%;vertical-align:-.08rem;margin-right:.45rem;border:1px solid rgba(0,0,0,.12)}.recommendation-row{display:flex;align-items:center;gap:.55rem;padding:.3rem .1rem;font-size:.75rem}
    .preview-zone{background:var(--canvas);padding:.7rem .95rem .55rem;min-height:calc(100vh - 3.5rem);max-height:calc(100vh - 3.5rem);overflow:hidden}.preview-toolbar{display:flex;align-items:center;justify-content:space-between;min-height:1.8rem}.preview-label{color:var(--muted);text-transform:uppercase;letter-spacing:.11em;font-size:.58rem;font-weight:800}.preview-title{font-size:.76rem;font-weight:700;margin-top:.15rem}.canvas{height:calc(100vh - 13rem);min-height:16rem;display:flex;align-items:center;justify-content:center;padding:.8rem;background:#e5e7eb;border-radius:.7rem;overflow:hidden}.canvas img,.canvas video{max-height:100%;max-width:100%;object-fit:contain;border-radius:.3rem}.empty-canvas{text-align:center;color:var(--muted);font-size:.72rem;line-height:1.45}.empty-icon{width:2.3rem;height:2.3rem;border-radius:.65rem;display:grid;place-items:center;background:#fff;margin:0 auto .5rem;color:var(--accent);font-size:1rem}
      .thumb-heading{display:flex;justify-content:space-between;align-items:center;margin:.9rem 0 .5rem}.thumb-heading strong{font-size:.78rem}.thumb-heading span{color:var(--muted);font-size:.68rem}.thumb-card{border:2px solid transparent;border-radius:.7rem;background:#fff;padding:.25rem}.thumb-card.selected{border-color:var(--accent);box-shadow:0 0 0 3px rgba(103,88,216,.12)}.thumb-card img{width:100%;aspect-ratio:1.15;object-fit:cover;border-radius:.45rem}.thumb-name{font-size:.66rem;font-weight:700;padding:.3rem .15rem}.thumb-check{color:var(--accent);font-size:.62rem;font-weight:750}.selected-strip{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-top:.75rem;padding:.7rem .85rem;background:#fff;border:1px solid var(--line);border-radius:.7rem}.selected-strip strong{font-size:.8rem}.selected-strip span{display:block;color:var(--muted);font-size:.68rem;margin-top:.15rem}.stButton>button{border-radius:.55rem;min-height:2.45rem;font-weight:700}.stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent)}
    div[data-testid="stFileUploader"]{margin:0!important} div[data-testid="stFileUploader"] section{padding:.35rem .5rem!important;min-height:2.65rem!important} div[data-testid="stFileUploader"] small{font-size:.58rem!important} div[data-testid="stTextInput"]{margin:0!important} div[data-testid="stTextInput"] input{font-size:.75rem!important;padding:.45rem .55rem!important;height:2.2rem!important} .stCaption{font-size:.6rem!important;margin:.15rem 0!important} .stButton>button{min-height:2.05rem!important;font-size:.7rem!important;padding:.3rem .5rem!important}
    .preview-zone{background:transparent;padding:0;min-height:0;max-height:none;overflow:visible}
    .upload-generating{display:flex;align-items:center;justify-content:center;gap:.5rem;
      min-height:2.05rem;padding:.3rem .5rem;border-radius:.55rem;background:var(--accent);
      color:white;font-size:.7rem;font-weight:700}
    .upload-generating-ring{width:.85rem;height:.85rem;border:2px solid #ffffff66;
      border-top-color:white;border-radius:50%;animation:upload-spin .8s linear infinite}
    @keyframes upload-spin{to{transform:rotate(360deg)}}
    [data-testid="stColumn"]:has(.preview-zone){background:#f7f7f8!important}
    .canvas.color-canvas{
      height:clamp(10rem, calc(100dvh - 23rem), 65dvh);
      min-height:0!important;
      box-sizing:border-box;
    }
    .thumb-heading{margin:.25rem 0}
    .st-key-studio_body [class*="st-key-color_thumb_"] button{
      height:clamp(5rem, 12dvh, 8rem);
      padding:.3rem!important;
      border-radius:.65rem;
      background-repeat:no-repeat!important;
      background-position:center!important;
      background-size:contain!important;
      align-items:flex-end;
    }
    .st-key-studio_body [class*="st-key-color_thumb_"] button p{
      background:rgba(255,255,255,.94);
      color:#18181b;
      border-radius:.25rem;
      padding:.1rem .4rem;
      font-size:.65rem;
    }
    @media(max-width:900px){.stepper{display:none}.rail{min-height:auto;border-right:0;border-bottom:1px solid var(--line)}.canvas{min-height:18rem}}
    @media(max-width:640px){
      .st-key-studio_body{overflow-y:auto;overscroll-behavior-y:contain}
      .st-key-studio_body > [data-testid="stHorizontalBlock"]{height:auto;overflow:visible}
      .st-key-studio_body > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{height:auto;overflow:visible}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in (("phase", "upload"), ("vace_candidates", []), ("selected_anchor", None), ("selected_preview", None), ("video_result", None)):
    st.session_state.setdefault(key, default)

phase = phase_number()
step_markup = "".join(
    f'<div class="step {"active" if index <= phase else ""}"><span class="step-dot">{index}</span>{label}</div>'
    + ('<span class="step-line"></span>' if index < 3 else '')
    for index, label in ((1, "Upload"), (2, "Color"), (3, "Video"))
)
st.markdown(f'<div class="studio-header"><div class="brand"><div><span class="brand-name">AI Hair Studio</span><span class="brand-subtitle">Personal Color · Hairstyle</span></div></div><div class="stepper">{step_markup}</div></div>', unsafe_allow_html=True)

with st.container(key="studio_body"):
    left, main = st.columns([0.29, 0.71], gap="small")
    with left:
        with st.container(key="left_rail"):
            if st.session_state.phase == "upload":
                st.markdown('<div class="section-title"><span class="section-number">01</span>Upload</div>', unsafe_allow_html=True)
                options = {}
                for field, label in (("bangs", "앞머리"), ("length", "길이"), ("wave", "웨이브"), ("color", "컬러")):
                    choices = COLOR_OPTIONS if field == "color" else STYLE_OPTIONS[field]
                    options[field] = st.selectbox(label, list(choices), index=None, placeholder="선택 안 함", key=f"style_{field}")
                st.markdown(
                    '<div class="reference-upload-label">원하는 스타일 사진 (선택)</div>'
                    '<div class="reference-upload-help">옵션에 없는 스타일은 사진으로 지정하세요.</div>',
                    unsafe_allow_html=True,
                )
                reference_upload = st.file_uploader("원하는 스타일 사진 (선택)", type=["jpg", "jpeg", "png", "webp"], key="reference_upload", label_visibility="collapsed")
                source_upload = st.file_uploader("사용자 영상 (필수)", type=["mp4", "mov"], key="source_upload")
                ready = inputs_ready(source_upload is not None, options, reference_upload is not None)
                action_slot = st.empty()
                upload_error = st.session_state.pop("upload_error", None)
                if upload_error:
                    st.error(upload_error)
                if action_slot.button("완료", type="primary", use_container_width=True, disabled=not ready):
                    action_slot.markdown(
                        '<div class="upload-generating" role="status" aria-live="polite">'
                        '<span class="upload-generating-ring" aria-hidden="true"></span>'
                        '미리보기 생성 중...</div>', unsafe_allow_html=True,
                    )
                    clear_from_upload_change()
                    root = Path(os.environ.get("VACE_WORKSPACE", "artifacts/jaeeun/vace-ui")).resolve()
                    st.session_state.job_workspace = str(root / uuid.uuid4().hex)
                    st.session_state.source_path = str(save_upload(source_upload))
                    st.session_state.reference_path = str(save_upload(reference_upload)) if reference_upload is not None else None
                    st.session_state.style_options = options
                    st.session_state.style_prompt = hairstyle_prompt(options, reference_upload is not None)
                    try:
                        frame = get_representative_frame()
                        checkpoint = os.environ.get("PERSONAL_COLOR_CHECKPOINT")
                        if checkpoint:
                            result = classify_personal_color(frame, Path(checkpoint))
                            st.session_state.personal_color_result = {"label": result.label, "label_ko": result.label_ko, "confidence": result.confidence, "source": "AI 분석"}
                        else:
                            st.session_state.personal_color_result = {"label": "summer_cool", "label_ko": COLOR_LABELS["summer_cool"], "confidence": None, "source": "기본 추천"}
                        st.session_state.edit_type = "custom"
                        create_anchor_candidates()
                        st.session_state.phase = "color"
                        st.rerun()
                    except (RuntimeError, ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
                        st.session_state.upload_error = str(error)
                        st.session_state.phase = "upload"
                        st.rerun()
            else:
                result = st.session_state.personal_color_result
                st.markdown('<div class="section-title color-section-title"><span class="section-number">02</span>Color</div>', unsafe_allow_html=True)
                confidence = f'<div class="confidence">Confidence <b>{result["confidence"]:.0%}</b></div>' if result.get("confidence") is not None else ''
                source_label = '<div class="confidence">AI 분석</div>' if result.get("source") == "AI 분석" else ''
                st.markdown(f'<div class="color-result"><strong>당신의 퍼스널컬러는 {result["label_ko"]}입니다.</strong>{source_label}{confidence}</div>', unsafe_allow_html=True)
                st.caption("요청한 스타일과 추천 컬러를 비교해보세요.")
                for color_index, (color_id, _, prompt_color) in enumerate((("requested", "요청한 스타일", ""),) + active_palette()):
                    english, korean, hex_color = color_meta(color_id)
                    label = korean if color_id == "requested" else f"추천 {color_index}.  {korean}  ·  {english}"
                    if st.button(label, key=f"color_{color_id}", use_container_width=True):
                        st.session_state.selected_anchor = color_id
                        st.session_state.selected_preview = color_id

    with main:
        if st.session_state.phase == "upload":
            st.markdown('<div class="preview-zone"><div class="preview-toolbar"><div><div class="preview-label">Upload</div><div class="preview-title">입력을 완료하면 Color 단계가 시작됩니다.</div></div></div><div class="canvas"><div class="empty-canvas"><div class="empty-icon">✦</div><strong>왼쪽에서 원하는 스타일과 영상을 입력하세요.</strong><br>사용자 영상과 스타일 옵션 또는 참조 사진을 넣고 완료를 눌러주세요.</div></div></div>', unsafe_allow_html=True)
        elif st.session_state.phase == "color":
            selected = next((item for item in st.session_state.vace_candidates if item["id"] == st.session_state.selected_anchor), None)
            preview_path = selected["path"] if selected else st.session_state.get("representative_frame")
            title = color_meta(selected["id"])[0] if selected else "Original Frame"
            st.markdown(f'<div class="preview-toolbar"><div><div class="preview-label">Color Preview</div><div class="preview-title">{title}</div></div><div class="helper">추천 색상을 선택해보세요.</div></div>', unsafe_allow_html=True)
            if preview_path and Path(preview_path).exists(): st.markdown(f'<div class="canvas color-canvas"><img src="{image_uri(preview_path)}" alt="{title}"></div>', unsafe_allow_html=True)
            else: st.markdown('<div class="canvas color-canvas"><div class="empty-canvas">색상 버튼을 누르면 Preview가 생성됩니다.</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="thumb-heading"><strong>요청한 스타일 · 추천 컬러</strong><span>사진을 눌러 비교해보세요.</span></div>', unsafe_allow_html=True)
            thumbs = st.session_state.get("vace_candidates", [])
            if thumbs:
                columns = st.columns(len(thumbs))
                for column, item in zip(columns, thumbs):
                    with column:
                        is_selected = item["id"] == st.session_state.selected_anchor
                        button_key = f"color_thumb_{item['id']}"
                        border_color = "var(--accent)" if is_selected else "transparent"
                        st.markdown(
                            f'<style>.st-key-{button_key} button{{'
                            f'background-image:url("{image_uri(item["path"])}")!important;'
                            f'border:2px solid {border_color}!important;'
                            '}</style>', unsafe_allow_html=True,
                        )
                        label = ("✓ " if is_selected else "") + color_meta(item["id"])[1]
                        if st.button(label, key=button_key, use_container_width=True):
                            st.session_state.selected_anchor = item["id"]
                            st.session_state.selected_preview = item["id"]
                            st.rerun()
            else:
                st.caption("왼쪽에서 색상을 선택하면 헤어스타일 미리보기가 생성됩니다.")
            if st.session_state.vace_candidates and st.button("이 색상으로 동영상 생성", type="primary", use_container_width=True, key="generate_video"):
                st.session_state.phase = "video"
                try:
                    model_path = Path(os.environ.get("VACE_MODEL_DIR", "/root/vace-model"))
                    if not any(model_path.rglob("*")):
                        st.rerun()
                    with st.spinner("동영상을 생성하는 중..."):
                        generate_video()
                    st.rerun()
                except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
                    st.error(str(error))
        else:
            result_path = st.session_state.get("video_result")
            st.markdown('<div class="preview-toolbar"><div><div class="preview-label">Video Result</div><div class="preview-title">선택한 헤어컬러 영상</div></div><div class="helper">03 Video</div></div>', unsafe_allow_html=True)
            if result_path and Path(result_path).exists():
                st.video(result_path)
                with open(result_path, "rb") as result_file:
                    st.download_button("결과 영상 다운로드", result_file, file_name=Path(result_path).name, mime="video/mp4", use_container_width=True)
            else:
                st.markdown('<div class="canvas"><div class="empty-canvas"><div class="empty-icon">◌</div><strong>영상 생성 기능을 준비 중입니다.</strong><br>Wan/VACE 모델이 연결되면 이곳에 최종 영상이 표시됩니다.</div></div>', unsafe_allow_html=True)
