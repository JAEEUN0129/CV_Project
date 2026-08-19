"""Streamlit UI for hair and outfit virtual fitting.

Two editing paths sit behind one screen and they behave very differently, which the layout
has to make obvious. Recolouring keeps the original framing, sound and back-of-head shots
and finishes in seconds; restyling regenerates the hair, needs a face in view, returns a
square crop and costs minutes a frame. Presenting them as equal options would mislead.
"""

from pathlib import Path
import importlib
import tempfile

import streamlit as st

import jaeeun.data as data_module
import jaeeun.models as models_module
import jaeeun.video as video_module
import jaeeun.hairstyle as hairstyle_module
import jaeeun.pipeline as pipeline_module

# Streamlit keeps imported modules in memory between reruns. Reload dependencies before the
# pipeline; otherwise a fresh pipeline can import a stale module that lacks new helpers.
importlib.invalidate_caches()
for _module in (data_module, models_module, video_module, hairstyle_module):
    importlib.reload(_module)
pipeline_module = importlib.reload(pipeline_module)
VirtualFittingPipeline = pipeline_module.VirtualFittingPipeline
HairstyleEnvironment = hairstyle_module.HairstyleEnvironment
HairstyleTransfer = hairstyle_module.HairstyleTransfer

RESTYLE = "헤어스타일 바꾸기"
RECOLOR_HAIR = "머리 색 바꾸기"
RECOLOR_OUTFIT = "상의 색 바꾸기"
RECOLOR_BOTH = "머리·상의 함께"
INSPECT = "부위 확인"

KEEP_MY_COLOR = "지금 내 머리색 그대로"
UPLOAD_OWN = "사진 직접 올리기"

st.set_page_config(page_title="헤어·의상 가상 피팅", layout="wide", initial_sidebar_state="expanded")

# Flat separation only: background, spacing and type carry the grouping, so nothing here
# draws boxes or shadows around content.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1180px; }
      h1 { font-size: 1.65rem !important; font-weight: 700; letter-spacing: -0.02em; }
      h2 { font-size: 1.0rem !important; font-weight: 600; margin: 1.5rem 0 0.4rem !important;
           color: #6b7280; }
      .lede { color: #6b7280; font-size: 0.9rem; margin: -0.4rem 0 1.2rem; }
      .stButton>button { border-radius: 8px; font-weight: 600; padding: 0.55rem 1.4rem; }
      section[data-testid="stSidebar"] { background: #f7f8fa; }
      div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def save_upload(upload) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix.lower()) as temporary:
        temporary.write(upload.getbuffer())
        return Path(temporary.name)


def pick_reference(label: str, presets: dict, key: str, extra: list | None = None):
    """Choose a reference photo from the prepared set, or let the user supply one."""
    options = (extra or []) + list(presets) + [UPLOAD_OWN]
    choice = st.selectbox(label, options, key=key + "_choice")
    if choice == UPLOAD_OWN:
        upload = st.file_uploader(
            label + " 사진", type=["jpg", "jpeg", "png", "webp"], key=key + "_upload"
        )
        return (save_upload(upload) if upload else None), choice
    return presets.get(choice), choice


environment = HairstyleEnvironment()
presets = environment.available_presets()
missing = environment.missing_parts()

st.title("헤어·의상 가상 피팅")
st.markdown(
    '<p class="lede">사진이나 짧은 영상을 올리면 머리와 상의를 바꿔 봅니다.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 입력")
    uploaded = st.file_uploader(
        "사진 또는 영상", type=["jpg", "jpeg", "png", "webp", "mp4"], label_visibility="collapsed"
    )
    if uploaded:
        st.caption(uploaded.name)
    st.markdown("### 무엇을 바꿀까요")
    action = st.radio(
        "기능", [RESTYLE, RECOLOR_HAIR, RECOLOR_OUTFIT, RECOLOR_BOTH, INSPECT],
        label_visibility="collapsed",
    )
    if action == RESTYLE and missing:
        st.warning("준비되지 않음: " + ", ".join(missing))

if not uploaded:
    st.info("왼쪽에서 사진이나 영상을 올려 시작하세요.")
    st.markdown("## 두 가지 방식")
    intro_left, intro_right = st.columns(2)
    with intro_left:
        st.markdown("**머리·상의 색 바꾸기**")
        st.caption(
            "원래 머리의 밝기와 결을 남기고 색만 옮깁니다. 원본 화면과 소리가 그대로 유지되고, "
            "뒤통수를 보이는 장면도 처리됩니다. 몇 초면 끝납니다."
        )
    with intro_right:
        st.markdown("**헤어스타일 바꾸기**")
        st.caption(
            "머리를 새로 그립니다. 길이와 모양이 실제로 바뀌지만 얼굴이 정면에 가까워야 하고, "
            "결과는 얼굴을 잘라낸 정사각형으로 나옵니다. 한 장에 수 분 걸립니다."
        )
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
is_video = suffix == ".mp4"
input_path = save_upload(uploaded)
pipeline = VirtualFittingPipeline()

shape_reference = color_reference = None
hair_color = outfit_color = None

with st.sidebar:
    st.markdown("### 설정")
    if action == RESTYLE:
        shape_reference, _ = pick_reference("머리 모양", presets, "shape")
        color_reference, color_choice = pick_reference(
            "머리 색", presets, "color", extra=[KEEP_MY_COLOR]
        )
        if color_choice == KEEP_MY_COLOR:
            color_reference = None
            st.caption("색을 따로 고르지 않으면 모양 사진의 색이 함께 옵니다.")
    else:
        if action in (RECOLOR_HAIR, RECOLOR_BOTH):
            hair_color = st.color_picker("머리 색", "#6A3520")
        if action in (RECOLOR_OUTFIT, RECOLOR_BOTH):
            outfit_color = st.color_picker("상의 색", "#2457C5")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("## 입력")
    if is_video:
        st.video(str(input_path))
    else:
        st.image(str(input_path))

    if action == RESTYLE and shape_reference:
        st.markdown("## 참조")
        reference_columns = st.columns(2)
        reference_columns[0].image(str(shape_reference), caption="모양", width=140)
        if color_reference:
            reference_columns[1].image(str(color_reference), caption="색", width=140)
        else:
            reference_columns[1].caption("색: 모양 사진과 동일")

    if action == RESTYLE and is_video and not missing:
        st.markdown("## 사용 가능 여부")
        st.caption("얼굴이 보이는 프레임만 처리됩니다. 오래 걸리는 작업 전에 먼저 확인하세요.")
        if st.button("미리 확인", use_container_width=True):
            progress = st.progress(0.0, text="프레임을 검사하는 중...")
            pipeline.prepare_video(input_path)
            usable, unusable = HairstyleTransfer().check_frames(
                pipeline.config.workspace / "frames",
                on_progress=lambda done, total: progress.progress(
                    done / total, text=str(done) + "/" + str(total) + " 검사"
                ),
            )
            progress.empty()
            st.session_state["check"] = (len(usable), len(unusable))
        if "check" in st.session_state:
            usable_count, unusable_count = st.session_state["check"]
            metrics = st.columns(3)
            metrics[0].metric("전체", str(usable_count + unusable_count) + "장")
            metrics[1].metric("처리 가능", str(usable_count) + "장")
            metrics[2].metric("예상 시간", str(usable_count * 3) + "~" + str(usable_count * 5) + "분")
            if unusable_count:
                st.warning(
                    str(unusable_count) + "장은 얼굴을 찾지 못해 건너뜁니다. 고개를 크게 돌린 "
                    "구간을 잘라내거나 정면을 보는 영상으로 다시 찍으면 결과가 좋아집니다."
                )

    run = st.button("실행", type="primary", use_container_width=True)

with right:
    st.markdown("## 결과")
    if run:
        try:
            if action == RESTYLE:
                if missing:
                    raise RuntimeError("헤어스타일 변경을 쓸 수 없습니다: " + ", ".join(missing))
                if shape_reference is None:
                    raise ValueError("머리 모양 사진을 먼저 골라주세요.")
                progress = st.progress(0.0, text="모델을 준비하는 중...")
                if is_video:
                    produced = pipeline.restyle_video(
                        input_path, shape_reference,
                        on_progress=lambda done, total: progress.progress(
                            done / total, text=str(done) + "/" + str(total) + " 프레임"
                        ),
                        color_reference=color_reference,
                    )
                else:
                    produced = pipeline.restyle_image(input_path, shape_reference, color_reference)
                progress.empty()
                st.session_state["result"] = {
                    "kind": "video" if is_video else "image",
                    "path": str(produced),
                    "note": "모델 규격에 맞춰 얼굴을 정사각형으로 잘라낸 화면입니다.",
                }
            elif action == INSPECT:
                with st.spinner("부위를 나누는 중..."):
                    reference = pipeline.prepare_video(input_path)[0] if is_video else input_path
                    results = pipeline.parse_image(reference)
                st.session_state["result"] = {
                    "kind": "image", "path": str(results["overlay"]),
                    "note": "색은 편집 결과가 아니라 부위 구분용입니다.",
                }
            else:
                colors = {}
                if hair_color:
                    colors["hair"] = hair_color
                if outfit_color:
                    colors["top"] = outfit_color
                if is_video:
                    progress = st.progress(0.0, text="첫 프레임에서 부위를 찾는 중...")
                    produced = pipeline.recolor_video(
                        input_path, colors,
                        on_progress=lambda done, total: progress.progress(
                            done / total, text=str(done) + "/" + str(total) + " 프레임"
                        ),
                    )
                    progress.empty()
                    kind = "video"
                else:
                    with st.spinner("색을 바꾸는 중..."):
                        if len(colors) == 2:
                            produced = pipeline.recolor_hair_and_outfit(
                                input_path, hair_color, outfit_color
                            )
                        else:
                            part, value = next(iter(colors.items()))
                            produced = pipeline.recolor_image(input_path, part, value)
                    kind = "image"
                st.session_state["result"] = {"kind": kind, "path": str(produced), "note": None}
        except (ValueError, RuntimeError) as error:
            st.session_state.pop("result", None)
            st.error(str(error))

    # Rendered outside the run block so a result survives the rerun that any later widget
    # change triggers, instead of vanishing the moment a colour is adjusted.
    result = st.session_state.get("result")
    if not result:
        st.caption("실행하면 여기에 결과가 나타납니다.")
    else:
        if result["kind"] == "video":
            st.video(result["path"])
        else:
            st.image(result["path"])
        with open(result["path"], "rb") as handle:
            st.download_button(
                "결과 내려받기", handle, file_name=Path(result["path"]).name,
                use_container_width=True,
            )
        if result.get("note"):
            st.caption(result["note"])
