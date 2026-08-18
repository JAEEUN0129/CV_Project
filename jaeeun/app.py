"""Streamlit UI for image parsing, recoloring and video preprocessing."""

from pathlib import Path
import importlib
import tempfile

import streamlit as st

import jaeeun.data as data_module
import jaeeun.models as models_module
import jaeeun.video as video_module
import jaeeun.hairstyle as hairstyle_module
import jaeeun.pipeline as pipeline_module


# Streamlit keeps imported modules in memory between reruns. Reload dependencies
# before the pipeline; otherwise a fresh pipeline can import an old cached data
# module that does not yet contain newly added helpers such as sampled_fps.
importlib.invalidate_caches()
data_module = importlib.reload(data_module)
models_module = importlib.reload(models_module)
video_module = importlib.reload(video_module)
hairstyle_module = importlib.reload(hairstyle_module)
pipeline_module = importlib.reload(pipeline_module)
VirtualFittingPipeline = pipeline_module.VirtualFittingPipeline

HAIRSTYLE_ACTION = "헤어스타일 변경"
RECOLOR_ACTIONS = {
    "머리·상의 색상 동시 변경": {"hair": None, "top": None},
    "머리 색상만 변경": {"hair": None},
    "상의 색상만 변경": {"top": None},
}


def save_upload(upload) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix.lower()) as temporary:
        temporary.write(upload.getbuffer())
        return Path(temporary.name)


st.set_page_config(page_title="Hair & Outfit Virtual Fitting", layout="wide")
st.title("헤어·의상 가상 피팅")
uploaded = st.file_uploader("사진 또는 짧은 영상을 업로드하세요", type=["jpg", "jpeg", "png", "webp", "mp4"])
hair_color = st.color_picker("머리 색상", "#6A3520")
outfit_color = st.color_picker("상의 색상", "#2457C5")

hair_environment = hairstyle_module.HairstyleEnvironment()
presets = hair_environment.available_presets()

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix != ".mp4":
        st.image(uploaded, caption="업로드 이미지", width=400)
    input_path = save_upload(uploaded)

    action = st.radio(
        "처리 기능",
        [HAIRSTYLE_ACTION, *RECOLOR_ACTIONS, "분할 마스크 확인"],
    )
    video_colors = {
        "머리·상의 색상 동시 변경": {"hair": hair_color, "top": outfit_color},
        "머리 색상만 변경": {"hair": hair_color},
        "상의 색상만 변경": {"top": outfit_color},
    }

    reference_path = None
    if action == HAIRSTYLE_ACTION:
        missing = hair_environment.missing_parts()
        if missing:
            st.warning(f"헤어스타일 변경 준비가 끝나지 않았습니다. 없는 항목: {', '.join(missing)}")
        else:
            # The model copies a hairstyle from a photo rather than reading a description,
            # so the choice is always an image: either a prepared one or the user's own.
            choice = st.selectbox("원하는 헤어스타일", [*presets, "직접 올리기"])
            if choice == "직접 올리기":
                own = st.file_uploader(
                    "원하는 헤어스타일 사진", type=["jpg", "jpeg", "png", "webp"], key="reference"
                )
                reference_path = save_upload(own) if own else None
            else:
                reference_path = presets[choice]
            if reference_path:
                st.image(str(reference_path), caption="이 머리 모양과 색을 가져옵니다", width=220)
            st.caption(
                "이 컴퓨터에는 그래픽 가속이 없어 한 장에 2~5분이 걸립니다. "
                "영상은 프레임 수만큼 곱해서 걸리고, 얼굴이 보이지 않는 프레임은 건너뜁니다."
            )

    if st.button("실행", type="primary"):
        pipeline = VirtualFittingPipeline()
        normalized = pipeline.config.workspace / "inputs" / "normalized.png"
        try:
            if action == HAIRSTYLE_ACTION:
                if reference_path is None:
                    st.warning("원하는 헤어스타일 사진을 먼저 골라주세요.")
                    st.stop()
                progress = st.progress(0.0, text="헤어스타일 모델을 준비하는 중...")
                if suffix == ".mp4":
                    result = pipeline.restyle_video(
                        input_path,
                        reference_path,
                        on_progress=lambda done, total: progress.progress(
                            done / total, text=f"{done}/{total} 프레임 처리 중"
                        ),
                    )
                    kind = "video"
                else:
                    result = pipeline.restyle_image(input_path, reference_path)
                    kind = "image"
                progress.empty()
                st.session_state["result"] = {
                    "kind": kind,
                    "after": str(result),
                    "note": "결과는 모델 규격에 맞춰 얼굴을 정사각형으로 잘라낸 화면입니다.",
                }
            elif action == "분할 마스크 확인":
                with st.spinner("부위를 분할하는 중..."):
                    reference = pipeline.prepare_video(input_path)[0] if suffix == ".mp4" else input_path
                    results = pipeline.parse_image(reference)
                st.session_state["result"] = {"kind": "masks", "after": str(results["overlay"])}
            elif suffix == ".mp4":
                # Tracking runs on the CPU here, so a per-frame bar is the only way the
                # user can tell a minutes-long run apart from a hang.
                progress = st.progress(0.0, text="첫 프레임에서 부위를 찾는 중...")
                result = pipeline.recolor_video(
                    input_path,
                    video_colors[action],
                    on_progress=lambda done, total: progress.progress(
                        done / total, text=f"{done}/{total} 프레임 처리 중"
                    ),
                )
                progress.empty()
                st.session_state["result"] = {"kind": "video", "after": str(result)}
            else:
                with st.spinner("색상을 바꾸는 중..."):
                    if action == "머리·상의 색상 동시 변경":
                        result = pipeline.recolor_hair_and_outfit(input_path, hair_color, outfit_color)
                    else:
                        part, color = ("hair", hair_color) if action.startswith("머리") else ("top", outfit_color)
                        result = pipeline.recolor_image(input_path, part, color)
                st.session_state["result"] = {
                    "kind": "image", "before": str(normalized), "after": str(result)
                }
        except (ValueError, RuntimeError) as error:
            st.session_state.pop("result", None)
            st.error(f"처리할 수 없습니다: {error}")

# Rendered outside the button block so the result survives the rerun that any later
# widget change triggers, instead of vanishing the moment a color is adjusted.
result = st.session_state.get("result")
if result:
    if result["kind"] == "masks":
        st.success("부위별 마스크 생성 완료 — 아래 색은 편집 색상이 아닌 클래스 구분용입니다.")
        st.image(result["after"], caption="분할 마스크 시각화(편집 결과 아님)", width=512)
    elif result["kind"] == "video":
        st.success("영상 편집 완료")
        st.video(result["after"])
        with open(result["after"], "rb") as video_file:
            st.download_button("결과 영상 내려받기", video_file, file_name=Path(result["after"]).name)
    elif "before" in result:
        st.success("색상 편집 완료")
        col1, col2 = st.columns(2)
        col1.image(result["before"], caption="Before")
        col2.image(result["after"], caption="After")
    else:
        # Hairstyle results come back as the model's own square crop, so there is no
        # matching "before" at the same framing to sit beside them.
        st.success("헤어스타일 변경 완료")
        st.image(result["after"], caption="After", width=512)
    if result.get("note"):
        st.caption(result["note"])

st.caption(
    "색상 변경은 사진·영상 모두 지원합니다. 헤어스타일 변경은 정면을 향한 얼굴에서만 동작하며, "
    "고개를 옆이나 뒤로 크게 돌린 프레임은 건너뜁니다."
)
