"""Consumer-facing Streamlit editor for personal-colour hairstyle anchors."""

from __future__ import annotations

import base64
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaeeun.personal_color import classify_personal_color
from jaeeun.vace.anchor_mask import build_anchor_mask
from jaeeun.vace.anchor_selector import select_anchor
from jaeeun.vace.color_candidates import PERSONAL_COLOR_PALETTES, build_color_candidates
from jaeeun.vace.specs import EDIT_SPECS


COLOR_LABELS = {
    "spring_warm": "봄 웜",
    "summer_cool": "여름 쿨",
    "autumn_warm": "가을 웜",
    "winter_cool": "겨울 쿨",
}
COLOR_META = {
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


def upload_digest(upload) -> str:
    return hashlib.sha256(upload.getvalue()).hexdigest()


def image_uri(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def color_meta(color_id: str) -> tuple[str, str, str]:
    return COLOR_META.get(color_id, (color_id, color_id, "#8A8F98"))


def reset_state() -> None:
    for key in (
        "source_path", "source_name", "source_digest", "reference_path", "reference_name",
        "reference_digest", "representative_frame", "personal_color_result",
        "vace_candidates", "vace_generation_key", "selected_preview", "selected_anchor",
        "manual_personal_color", "edit_type", "source_upload", "reference_upload",
    ):
        st.session_state.pop(key, None)


def choose_preview(item_id: str) -> None:
    st.session_state.selected_preview = item_id
    if item_id != "original":
        st.session_state.selected_anchor = item_id
    else:
        st.session_state.selected_anchor = None


def generation_key(color_result: dict, edit_type: str) -> str:
    values = "|".join(
        (
            st.session_state.get("source_digest", ""),
            st.session_state.get("reference_digest", ""),
            color_result["label"],
            edit_type,
            EDIT_SPECS[edit_type].prompt,
        )
    )
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def candidates_are_available(candidates: list[dict]) -> bool:
    return len(candidates) == 3 and all(Path(item["path"]).is_file() for item in candidates)


st.set_page_config(page_title="AI Hair Studio", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      :root{--ink:#18181b;--muted:#71717a;--line:#e4e4e7;--canvas:#eef0f3;--accent:#6758d8}
      .stApp{background:#f7f7f8;color:var(--ink)} .block-container{max-width:1500px;padding:0 1.25rem 2rem}
      header[data-testid="stHeader"]{background:#fff;border-bottom:1px solid var(--line)}
      .studio-header{height:4.25rem;display:flex;align-items:center;justify-content:space-between}
      .brand{display:flex;align-items:center;gap:.7rem}.brand-mark{width:2rem;height:2rem;border-radius:.65rem;background:var(--accent);color:#fff;display:grid;place-items:center;font-weight:800}.brand-name{font-size:1.05rem;font-weight:750;letter-spacing:-.02em}.brand-subtitle{color:var(--muted);font-size:.72rem;margin-left:.25rem}
      .stepper{display:flex;align-items:center;gap:.55rem;color:#a1a1aa;font-size:.72rem}.step{display:flex;align-items:center;gap:.35rem;white-space:nowrap}.step-dot{width:1.35rem;height:1.35rem;border-radius:50%;display:grid;place-items:center;background:#e4e4e7;color:#71717a;font-size:.65rem;font-weight:700}.step.active{color:var(--ink);font-weight:650}.step.active .step-dot{background:var(--accent);color:#fff}.step-line{width:1.2rem;height:1px;background:var(--line)}
      .editor-shell{min-height:calc(100vh - 6.2rem);background:#fff;border:1px solid var(--line);border-radius:1rem;overflow:hidden}.rail{background:#fff;border-right:1px solid var(--line);padding:1.3rem 1.15rem;min-height:calc(100vh - 6.2rem)}.rail-kicker{color:var(--accent);text-transform:uppercase;letter-spacing:.1em;font-size:.65rem;font-weight:800}.rail-title{font-size:1.35rem;font-weight:760;letter-spacing:-.04em;margin:.3rem 0 .2rem}.rail-copy{color:var(--muted);font-size:.76rem;line-height:1.55;margin-bottom:1.3rem}.section{border-top:1px solid #f0f0f2;padding:1rem 0}.section-title{font-size:.76rem;font-weight:750;margin-bottom:.55rem}.section-number{color:var(--accent);margin-right:.35rem}.helper{color:var(--muted);font-size:.69rem;line-height:1.45}.upload-summary{background:#f7f7f8;border:1px solid var(--line);border-radius:.7rem;padding:.65rem .7rem;font-size:.75rem}.upload-summary strong{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.upload-summary span{color:#35a06b;font-size:.68rem}
      .color-result{display:flex;align-items:center;justify-content:space-between;background:#f5f3ff;border:1px solid #ded9ff;border-radius:.75rem;padding:.75rem}.color-result strong{display:block;font-size:.93rem}.confidence{color:var(--muted);font-size:.68rem;margin-top:.2rem}.confidence b{color:var(--accent)}.swatch{display:inline-block;width:.85rem;height:.85rem;border-radius:50%;vertical-align:-.08rem;margin-right:.45rem;border:1px solid rgba(0,0,0,.12)}.recommendation-row{display:flex;align-items:center;gap:.55rem;padding:.38rem .1rem;font-size:.75rem}.recommendation-row small{color:var(--muted);margin-left:auto}
      .preview-zone{background:var(--canvas);padding:1.25rem 1.35rem 1rem;min-height:calc(100vh - 6.2rem)}.preview-toolbar{display:flex;align-items:center;justify-content:space-between;min-height:2.2rem}.preview-label{color:var(--muted);text-transform:uppercase;letter-spacing:.11em;font-size:.64rem;font-weight:800}.preview-title{font-size:.85rem;font-weight:700;margin-top:.2rem}.canvas{min-height:28rem;display:flex;align-items:center;justify-content:center;padding:1.5rem;background:#e5e7eb;border-radius:.85rem;overflow:hidden}.canvas img{max-height:58vh;max-width:100%;object-fit:contain;border-radius:.35rem}.empty-canvas{text-align:center;color:var(--muted);font-size:.85rem;line-height:1.6}.empty-icon{width:2.8rem;height:2.8rem;border-radius:.8rem;display:grid;place-items:center;background:#fff;margin:0 auto .7rem;color:var(--accent);font-size:1.25rem}
      .thumb-heading{display:flex;justify-content:space-between;align-items:center;margin:.9rem 0 .5rem}.thumb-heading strong{font-size:.78rem}.thumb-heading span{color:var(--muted);font-size:.68rem}.thumb-card{border:2px solid transparent;border-radius:.7rem;background:#fff;padding:.25rem}.thumb-card.selected{border-color:var(--accent);box-shadow:0 0 0 3px rgba(103,88,216,.12)}.thumb-card img{width:100%;aspect-ratio:1.15;object-fit:cover;border-radius:.45rem}.thumb-name{font-size:.66rem;font-weight:700;padding:.3rem .15rem .1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.thumb-check{color:var(--accent);font-size:.62rem;font-weight:750}.selected-strip{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-top:.75rem;padding:.7rem .85rem;background:#fff;border:1px solid var(--line);border-radius:.7rem}.selected-strip strong{font-size:.8rem}.selected-strip span{display:block;color:var(--muted);font-size:.68rem;margin-top:.15rem}.disabled-note{color:var(--muted);font-size:.68rem;text-align:right}.stButton>button{border-radius:.55rem;min-height:2.45rem;font-weight:700}.stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent)}
      @media(max-width:900px){.stepper{display:none}.rail{min-height:auto;border-right:0;border-bottom:1px solid var(--line)}.canvas{min-height:22rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in (("selected_preview", "original"), ("selected_anchor", None), ("vace_candidates", []), ("personal_color_result", None)):
    st.session_state.setdefault(key, default)

st.markdown('<div class="studio-header"><div class="brand"><div class="brand-mark">AH</div><div><span class="brand-name">AI Hair Studio</span><span class="brand-subtitle">Personal Color · Hairstyle</span></div></div><div class="stepper"><div class="step active"><span class="step-dot">1</span>Upload</div><span class="step-line"></span><div class="step active"><span class="step-dot">2</span>Color</div><span class="step-line"></span><div class="step active"><span class="step-dot">3</span>Anchor</div><span class="step-line"></span><div class="step"><span class="step-dot">4</span>Video</div></div></div>', unsafe_allow_html=True)
header_left, header_right = st.columns([8, 1])
with header_right:
    if st.button("Reset", key="reset_top"):
        reset_state()
        st.rerun()

left, main = st.columns([0.29, 0.71], gap="small")
with left:
    st.markdown('<div class="rail"><div class="rail-kicker">Beauty AI editor</div><div class="rail-title">AI Hair Color Studio</div><div class="rail-copy">내 얼굴과 원하는 헤어스타일에 어울리는 컬러를 미리 확인해보세요.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section"><div class="section-title"><span class="section-number">01</span>Original Video</div>', unsafe_allow_html=True)
    source_upload = st.file_uploader("영상 업로드", type=["mp4", "mov"], label_visibility="collapsed", key="source_upload")
    if source_upload:
        source_digest = upload_digest(source_upload)
        if st.session_state.get("source_digest") != source_digest:
            st.session_state.source_path = str(save_upload(source_upload, ".mp4")); st.session_state.source_name = source_upload.name
            st.session_state.source_digest = source_digest
            for key in ("representative_frame", "vace_candidates", "personal_color_result", "vace_generation_key"): st.session_state.pop(key, None)
        st.markdown(f'<div class="upload-summary"><strong>{source_upload.name}</strong><span>✓ 업로드 완료</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get("source_path"):
        st.markdown('<div class="section"><div class="section-title"><span class="section-number">Frame</span>Representative Frame</div>', unsafe_allow_html=True)
        if not st.session_state.get("representative_frame"):
            try:
                with st.spinner("대표 프레임을 고르는 중..."):
                    frame_path = Path(os.environ.get("VACE_WORKSPACE", "artifacts/jaeeun/vace-ui")) / "source-anchor-frame.png"; frame_path.parent.mkdir(parents=True, exist_ok=True)
                    select_anchor(Path(st.session_state.source_path), frame_path); st.session_state.representative_frame = str(frame_path)
            except (RuntimeError, ValueError, OSError) as error:
                st.error(f"대표 프레임을 만들 수 없습니다: {error}")
        st.image(st.session_state.representative_frame, use_container_width=True); st.caption("퍼스널컬러 분석에 사용되는 대표 프레임")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-title"><span class="section-number">02</span>Hairstyle Reference</div>', unsafe_allow_html=True)
    reference_upload = st.file_uploader("헤어스타일 이미지 업로드", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="reference_upload")
    if reference_upload:
        reference_digest = upload_digest(reference_upload)
        if st.session_state.get("reference_digest") != reference_digest:
            st.session_state.reference_path = str(save_upload(reference_upload)); st.session_state.reference_name = reference_upload.name; st.session_state.pop("vace_candidates", None)
            st.session_state.reference_digest = reference_digest
            st.session_state.pop("vace_generation_key", None)
        st.image(st.session_state.reference_path, use_container_width=True); st.markdown(f'<div class="upload-summary"><strong>{reference_upload.name}</strong><span>헤어 형태 참조 준비 완료</span></div>', unsafe_allow_html=True)
    st.caption("헤어스타일의 형태는 유지하고 추천된 컬러를 적용합니다."); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-title"><span class="section-number">03</span>Personal Color</div>', unsafe_allow_html=True)
    checkpoint = os.environ.get("PERSONAL_COLOR_CHECKPOINT")
    color_result = st.session_state.get("personal_color_result")
    if color_result:
        confidence = f'<div class="confidence">Confidence <b>{color_result["confidence"]:.0%}</b></div>' if color_result.get("confidence") is not None else ""
        source_label = color_result.get("source", "분석 결과")
        st.markdown(f'<div class="color-result"><div><strong>{color_result["label_ko"]}</strong><div class="confidence">{source_label}</div>{confidence}</div><span class="swatch" style="background:#6758d8"></span></div>', unsafe_allow_html=True)
    else:
        manual_color = st.selectbox("퍼스널컬러 선택", list(COLOR_LABELS), format_func=lambda value: COLOR_LABELS[value], label_visibility="collapsed", key="manual_personal_color")
        st.caption("모델이 있으면 대표 프레임을 자동 분석합니다.")
        if st.button("Analyze Personal Color", use_container_width=True, key="analyze_color"):
            if checkpoint and st.session_state.get("representative_frame"):
                result = classify_personal_color(Path(st.session_state.representative_frame), Path(checkpoint)); confidence_value = result.confidence; label = result.label; label_ko = result.label_ko
                source_label = "AI 분석"
            else:
                confidence_value = None; label = manual_color; label_ko = COLOR_LABELS[manual_color]; source_label = "수동 선택"
            st.session_state.personal_color_result = {"label": label, "label_ko": label_ko, "confidence": confidence_value, "source": source_label}; st.session_state.pop("vace_generation_key", None); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    color_result = st.session_state.get("personal_color_result")
    if color_result:
        st.markdown('<div class="section"><div class="section-title"><span class="section-number">04</span>Recommended Hair Colors</div>', unsafe_allow_html=True)
        for color_id, _, _ in PERSONAL_COLOR_PALETTES[color_result["label"]]:
            english, korean, hex_color = color_meta(color_id)
            st.markdown(f'<div class="recommendation-row"><span class="swatch" style="background:{hex_color}"></span><b>{english}</b><small>{korean}</small></div>', unsafe_allow_html=True)
        edit_type = st.selectbox("헤어스타일 변경", list(EDIT_SPECS), format_func=lambda value: value.replace("_", " ").title(), key="edit_type")
        if st.button("3가지 컬러 미리보기 생성", type="primary", use_container_width=True, key="generate_anchors"):
            if not st.session_state.get("source_path") or not st.session_state.get("reference_path"):
                st.error("원본 영상과 헤어스타일 참조 이미지를 먼저 업로드하세요.")
            else:
                try:
                    workspace = Path(os.environ.get("VACE_WORKSPACE", "artifacts/jaeeun/vace-ui")); frame = Path(st.session_state.representative_frame); anchor_mask = workspace / f"anchor-mask-{edit_type}.png"; current_key = generation_key(color_result, edit_type)
                    if st.session_state.get("vace_generation_key") == current_key and candidates_are_available(st.session_state.get("vace_candidates", [])):
                        st.info("이미 생성된 AI Preview를 다시 사용합니다.")
                        st.rerun()
                    with st.status("AI Preview를 생성하는 중...", expanded=True) as status:
                        build_anchor_mask(frame, anchor_mask, edit_type); status.write("대표 프레임과 헤어 영역을 준비했습니다.")
                        generated = build_color_candidates(frame, Path(st.session_state.reference_path), anchor_mask, workspace / "anchors", color_result["label"], Path(os.environ.get("FLUX_PYTHON", "/root/flux-env/bin/python")), Path(os.environ.get("FLUX_WORKER", str(Path(__file__).parent / "vace" / "flux_worker.py"))), EDIT_SPECS[edit_type].prompt)
                        st.session_state.vace_candidates = [{"id": item.color_id, "name": item.name, "prompt_color": item.prompt_color, "path": str(item.anchor)} for item in generated]; st.session_state.vace_generation_key = current_key; st.session_state.selected_preview = "original"; st.session_state.selected_anchor = None; status.update(label="AI Preview 준비 완료", state="complete")
                    st.rerun()
                except (RuntimeError, ValueError, FileNotFoundError) as error: st.error(str(error))
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with main:
    candidates = st.session_state.get("vace_candidates", []); preview_id = st.session_state.get("selected_preview", "original")
    selected = next((item for item in candidates if item["id"] == preview_id), None); preview_path = st.session_state.get("representative_frame") if preview_id == "original" else (selected["path"] if selected else st.session_state.get("representative_frame")); preview_title = "Original Frame" if preview_id == "original" else (color_meta(selected["id"])[0] if selected else "Selected Anchor")
    st.markdown('<div class="preview-zone">', unsafe_allow_html=True)
    st.markdown(f'<div class="preview-toolbar"><div><div class="preview-label">AI Preview</div><div class="preview-title">{preview_title}</div></div><div class="helper">Anchor selection preview</div></div>', unsafe_allow_html=True)
    if preview_path and Path(preview_path).exists(): st.markdown(f'<div class="canvas"><img src="{image_uri(preview_path)}" alt="{preview_title}"></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="canvas"><div class="empty-canvas"><div class="empty-icon">✦</div><strong>영상을 업로드하면 대표 프레임이 여기에 표시됩니다.</strong><br>왼쪽에서 입력과 분석을 시작하세요.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="thumb-heading"><strong>Preview selection</strong><span>이미지를 선택하면 큰 화면에 표시됩니다.</span></div>', unsafe_allow_html=True)
    thumbs = [("original", "Original", st.session_state.get("representative_frame"))] + [(item["id"], color_meta(item["id"])[0], item["path"]) for item in candidates]
    columns = st.columns(max(1, len(thumbs)))
    for column, (item_id, label, path) in zip(columns, thumbs):
        with column:
            selected_class = "selected" if item_id == preview_id else ""
            if path and Path(path).exists():
                check = '<div class="thumb-check">✓ SELECTED</div>' if item_id == preview_id else ''
                st.markdown(f'<div class="thumb-card {selected_class}"><img src="{image_uri(path)}" alt="{label}"><div class="thumb-name">{label}</div>{check}</div>', unsafe_allow_html=True)
            if st.button(f"{label} 선택", key=f"preview_{item_id}", use_container_width=True): choose_preview(item_id); st.rerun()
    selected_anchor = next((item for item in candidates if item["id"] == st.session_state.get("selected_anchor")), None)
    if selected_anchor:
        st.markdown(f'<div class="selected-strip"><div><strong>Selected Anchor · {color_meta(selected_anchor["id"])[0]}</strong><span>이 anchor는 향후 AI Video Generation에 사용됩니다.</span></div><div class="disabled-note">영상 생성 기능을 준비 중입니다.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="selected-strip"><div><strong>Selected Anchor</strong><span>하단 썸네일에서 원하는 컬러를 선택하세요.</span></div><div class="disabled-note">Wan/VACE 연결 예정</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)