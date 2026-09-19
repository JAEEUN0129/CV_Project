"""Streamlit UI for personal-colour anchors followed by Wan/VACE generation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from jaeeun.personal_color import classify_personal_color
from jaeeun.vace.anchor_mask import build_anchor_mask
from jaeeun.vace.anchor_selector import select_anchor
from jaeeun.vace.color_candidates import build_color_candidates
from jaeeun.vace.runner import VaceRun
from jaeeun.vace.specs import EDIT_SPECS
from jaeeun.prepare_vace_mask_video import build_mask_video


def _save_upload(upload, suffix: str | None = None) -> Path:
    extension = suffix or Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as handle:
        handle.write(upload.getbuffer())
        return Path(handle.name)


def _config_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


st.set_page_config(page_title="VACE 헤어 컬러 추천", layout="wide")
st.markdown(
        """
        <style>
            .block-container { max-width: 1440px; padding-top: 2rem; }
            h1 { font-size: 1.7rem !important; }
            .candidate-label { color: #667085; font-size: 0.8rem; margin-top: -0.35rem; }
            .result-empty {
                min-height: 28rem; display: flex; align-items: center; justify-content: center;
                background: #f3f5f7; color: #667085; border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
)
st.title("퍼스널컬러 기반 Wan 헤어스타일 영상")
st.caption("Flux가 색상별 anchor 3장을 만들고, 선택한 1장을 Wan/VACE에 전달합니다.")

source_upload = st.file_uploader("원본 영상", type=["mp4", "mov", "avi"])
reference_upload = st.file_uploader("헤어스타일 참조 이미지", type=["jpg", "jpeg", "png", "webp"])
edit_type = st.selectbox("헤어스타일 변경", list(EDIT_SPECS))
personal_color = st.selectbox(
    "퍼스널컬러",
    ["spring_warm", "summer_cool", "autumn_warm", "winter_cool"],
    format_func=lambda value: {
        "spring_warm": "봄웜",
        "summer_cool": "여름쿨",
        "autumn_warm": "가을웜",
        "winter_cool": "겨울쿨",
    }[value],
)

if source_upload and reference_upload:
    workspace = Path(os.environ.get("VACE_WORKSPACE", "artifacts/jaeeun/vace-ui"))
    workspace.mkdir(parents=True, exist_ok=True)
    source = _save_upload(source_upload, ".mp4")
    reference = _save_upload(reference_upload)
    flux_python = _config_path("FLUX_PYTHON", "/root/flux-env/bin/python")
    flux_worker = _config_path("FLUX_WORKER", "jaeeun/vace/flux_worker.py")
    vace_repo = _config_path("VACE_REPO", "/root/VACE")
    checkpoint = _config_path("VACE_MODEL_DIR", "/root/models/Wan2.1-VACE-1.3B")

    if st.button("퍼스널컬러 후보 3장 생성", type="primary"):
        try:
            selected_frame = workspace / "source-anchor-frame.png"
            selected = select_anchor(source, selected_frame)
            anchor_mask = workspace / f"anchor-mask-{edit_type}.png"
            build_anchor_mask(selected_frame, anchor_mask, edit_type)
            detected = None
            classifier_checkpoint = os.environ.get("PERSONAL_COLOR_CHECKPOINT")
            if classifier_checkpoint:
                detected = classify_personal_color(
                    selected_frame, Path(classifier_checkpoint), selected.face_bbox
                )
                personal_color = detected.label
            candidates = build_color_candidates(
                selected_frame,
                reference,
                anchor_mask,
                workspace / "anchors",
                personal_color,
                flux_python,
                flux_worker,
                EDIT_SPECS[edit_type].prompt,
            )
            st.session_state["vace_candidates"] = [
                {
                    "id": candidate.color_id,
                    "name": candidate.name,
                    "prompt_color": candidate.prompt_color,
                    "path": str(candidate.anchor),
                }
                for candidate in candidates
            ]
            st.session_state["vace_context"] = {
                "source": str(source),
                "workspace": str(workspace),
                "reference": str(reference),
                "selected_frame": str(selected_frame),
                "anchor_mask": str(anchor_mask),
                "vace_repo": str(vace_repo),
                "checkpoint": str(checkpoint),
                "edit_type": edit_type,
                "personal_color": personal_color,
                "classifier_result": (
                    {
                        "label": detected.label,
                        "confidence": detected.confidence,
                        "probabilities": detected.probabilities,
                    }
                    if detected
                    else None
                ),
            }
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            st.error(str(error))

if st.session_state.get("vace_candidates"):
    candidates = st.session_state["vace_candidates"]
    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        st.subheader("추천 헤어컬러")
        st.markdown('<p class="candidate-label">마음에 드는 anchor를 고르면 오른쪽에서 영상을 생성합니다.</p>', unsafe_allow_html=True)
        selected_id = st.radio(
            "Wan에 전달할 색상",
            [candidate["id"] for candidate in candidates],
            format_func=lambda value: next(
                candidate["name"] for candidate in candidates if candidate["id"] == value
            ),
            label_visibility="collapsed",
        )
        for candidate in candidates:
            st.image(candidate["path"], caption=candidate["name"], use_container_width=True)
        if st.button("선택한 스타일로 생성", type="primary", use_container_width=True):
            context = st.session_state["vace_context"]
            chosen = next(candidate for candidate in candidates if candidate["id"] == selected_id)
            try:
                mask_video = Path(context["workspace"]) / f"mask-{context['edit_type']}.mp4"
                masked_video = Path(context["workspace"]) / f"source-masked-{context['edit_type']}.mp4"
                build_mask_video(
                    Path(context["source"]),
                    mask_video,
                    masked_video,
                    edit_type=context["edit_type"],
                )
                spec = EDIT_SPECS[context["edit_type"]]
                prompt = (
                    f"{spec.prompt} The final hair colour is {chosen['prompt_color']}. "
                    "Match that colour consistently throughout the video."
                )
                output = Path(context["workspace"]) / f"result-{selected_id}.mp4"
                VaceRun(
                    repo=Path(context["vace_repo"]),
                    checkpoint=Path(context["checkpoint"]),
                    source_video=masked_video,
                    mask_video=mask_video,
                    anchor=Path(chosen["path"]),
                    output=output,
                    prompt=prompt,
                ).execute()
                st.session_state["vace_result"] = str(output)
            except (RuntimeError, ValueError, FileNotFoundError) as error:
                st.error(str(error))
    with right:
        st.subheader("최종 생성 결과")
        result = st.session_state.get("vace_result")
        if result and Path(result).is_file():
            st.video(result)
            with open(result, "rb") as result_file:
                st.download_button(
                    "결과 영상 다운로드",
                    result_file,
                    file_name=Path(result).name,
                    mime="video/mp4",
                    use_container_width=True,
                )
        else:
            st.markdown(
                '<div class="result-empty">왼쪽에서 anchor를 선택하고 생성을 시작하세요.</div>',
                unsafe_allow_html=True,
            )