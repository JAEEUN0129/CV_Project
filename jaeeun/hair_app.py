"""Hair-only Streamlit page used by :mod:`jaeeun.app`."""

from pathlib import Path
import tempfile

import streamlit as st

from .hairclip import HairClipEnvironment
from .hairstyle import HairstyleEnvironment
from .pipeline import VirtualFittingPipeline


def _save_upload(upload) -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(upload.name).suffix.lower()
    ) as temporary:
        temporary.write(upload.getbuffer())
        return Path(temporary.name)


def render() -> None:
    st.set_page_config(page_title="AI 헤어스타일 생성", layout="wide")
    st.markdown(
        """
        <style>
          .block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 4rem; }
          h1 { font-size: 2rem !important; letter-spacing: -0.04em; }
          .subtitle { color: #667085; margin: -0.5rem 0 2rem; }
          .stButton > button { min-height: 3rem; border-radius: 10px; font-weight: 700; }
          div[data-testid="stDownloadButton"] > button {
            min-height: 2.8rem; border-radius: 10px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    hair_environment = HairstyleEnvironment()
    clip_environment = HairClipEnvironment()
    missing = list(dict.fromkeys(
        hair_environment.missing_parts() + clip_environment.missing_parts()
    ))

    st.title("AI 헤어스타일 생성")
    st.markdown(
        '<p class="subtitle">내 사진이나 영상에 원하는 헤어스타일과 머리색을 적용해 보세요.</p>',
        unsafe_allow_html=True,
    )

    input_column, style_column = st.columns(2, gap="large")
    with input_column:
        st.markdown("### 내 사진 또는 영상")
        source_upload = st.file_uploader(
            "변경할 인물 사진이나 영상을 올려주세요.",
            type=["jpg", "jpeg", "png", "webp", "mp4"],
            key="source_upload",
        )
        if source_upload:
            if Path(source_upload.name).suffix.lower() == ".mp4":
                st.video(source_upload)
            else:
                st.image(source_upload, use_container_width=True)

    with style_column:
        st.markdown("### 원하는 스타일 사진")
        style_upload = st.file_uploader(
            "머리 모양이 잘 보이는 정면 사진을 올려주세요.",
            type=["jpg", "jpeg", "png", "webp"],
            key="style_upload",
        )
        if style_upload:
            st.image(style_upload, use_container_width=True)

    st.markdown("### 원하는 헤어스타일 텍스트")
    style_prompt = st.text_input(
        "원하는 헤어스타일 텍스트",
        placeholder="예: 차가운 애쉬 브라운으로 해주세요",
        help="한글로 입력하면 HairCLIP 실행 전에 영어 색상 표현으로 자동 변환합니다.",
        label_visibility="collapsed",
    )
    st.caption("머리 모양은 스타일 사진을 사용하고, 한글 텍스트는 영어로 변환해 색상 참조를 생성합니다.")

    if missing:
        st.warning("서버에 준비되지 않은 항목이 있습니다: " + ", ".join(missing))

    generate = st.button("헤어스타일 생성", type="primary", use_container_width=True)
    if generate:
        try:
            if missing:
                raise RuntimeError("모델 실행 환경이 준비되지 않았습니다: " + ", ".join(missing))
            if source_upload is None:
                raise ValueError("변경할 내 사진이나 영상을 올려주세요.")
            if style_upload is None:
                raise ValueError("원하는 스타일 사진을 올려주세요.")
            if not style_prompt.strip():
                raise ValueError("원하는 머리색을 텍스트로 입력해주세요.")

            source = _save_upload(source_upload)
            style = _save_upload(style_upload)
            is_video = Path(source_upload.name).suffix.lower() == ".mp4"
            pipeline = VirtualFittingPipeline()

            progress = st.progress(0.03, text="텍스트로 색상 참조를 생성하는 중...")
            color_reference = pipeline.create_text_color_reference(style, style_prompt)
            progress.progress(0.1, text="헤어스타일 생성 모델을 준비하는 중...")

            if is_video:
                result_path = pipeline.restyle_video(
                    source,
                    style,
                    color_reference=color_reference,
                    on_progress=lambda done, total: progress.progress(
                        0.1 + 0.9 * done / total,
                        text=f"{done}/{total} 프레임 생성 중...",
                    ),
                )
                result_kind = "video"
            else:
                result_path = pipeline.restyle_image(source, style, color_reference)
                progress.progress(1.0, text="완료")
                result_kind = "image"

            progress.empty()
            st.session_state["hair_result"] = {
                "kind": result_kind,
                "path": str(result_path),
            }
        except (ValueError, RuntimeError) as error:
            st.session_state.pop("hair_result", None)
            st.error(str(error))

    result = st.session_state.get("hair_result")
    if result:
        st.divider()
        st.markdown("## 생성 결과")
        if result["kind"] == "video":
            st.video(result["path"])
            download_label = "결과 영상 다운로드"
            mime = "video/mp4"
        else:
            st.image(result["path"], use_container_width=True)
            download_label = "결과 사진 다운로드"
            mime = "image/png"

        with open(result["path"], "rb") as result_file:
            st.download_button(
                download_label,
                data=result_file,
                file_name=Path(result["path"]).name,
                mime=mime,
                use_container_width=True,
            )
