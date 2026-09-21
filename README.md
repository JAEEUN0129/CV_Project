# AI Hair Studio

사용자 영상과 스타일 옵션 또는 참조 사진으로 요청한 스타일을 생성하고,
퍼스널컬러 추천색을 비교한 뒤 VACE 영상을 생성합니다.

## 실행

```bash
python -m streamlit run jaeeun/vace_app.py --server.port 8501
```

`jaeeun/app.py`도 같은 화면을 실행합니다. [설치 안내](jaeeun/SETUP.md)를 참고하세요.

## 구조

선택한 앞머리는 앞머리 영역에서, 길이·웨이브는 옆·아래 머리 영역에서 별도로 보정합니다.
부분 보정에는 참조 사진을 다시 넣지 않으며, 편집 영역 밖 픽셀은 복원합니다.
옵션에 따라 요청 스타일 생성에 FLUX 추론이 최대 3회 필요합니다.
추천색은 최종 요청 이미지의 머리색만 변경합니다.
자동 분할이 머리끝을 놓치면 Color 화면의 **염색 영역 보정**에서 브러시로
영역을 추가·제거한 뒤 **보정 적용**을 누르세요. 세 추천색에 함께 적용됩니다.

| 경로 | 역할 |
|---|---|
| `jaeeun/vace_app.py` | Upload → Color → Video UI |
| `jaeeun/personal_color.py` | 퍼스널컬러 분류 |
| `jaeeun/models.py` | 사람·얼굴 분할 어댑터 |
| `jaeeun/vace/style_options.py` | 옵션·프롬프트·표시 문구 |
| `jaeeun/vace/anchor_selector.py`, `anchor_mask.py` | 대표 프레임·편집 마스크 |
| `jaeeun/vace/anchor_editor.py`, `flux_worker.py` | FLUX 호출·추론 |
| `jaeeun/vace/color_candidates.py` | 요청 이미지 생성·추천색 변경 |
| `jaeeun/vace/regional_edit.py` | 앞머리·옆머리 분리 편집 및 보호 영역 복원 |
| `jaeeun/vace/mask_editor.py`, `mask_brush/` | 수동 염색 마스크 보정 |
| `jaeeun/prepare_vace_mask_video.py`, `jaeeun/vace/runner.py` | 영상 마스크·VACE 호출 |
| `jaeeun/vace/experiment.py`, `specs.py`, `short_hair_mask.py` | 별도 CLI 실험 |
| `vessl/` | GPU 실행 명세·실험 스크립트 |
| `tests/` | 회귀 검사 |
| `archive/legacy/` | 현재 웹에서 사용하지 않는 이전 구현 |

FLUX로 요청 이미지를 만든 뒤 추천색은 같은 이미지의 머리 영역에서만 변경합니다.
마스크 누락과 생성 모델의 옵션 반영은 실제 GPU 결과로 검증해야 합니다.

[파일 정리 목록](archive/legacy/README.md)에 이동 파일과 보존 파일을 기록했습니다.
모델 가중치, 가상환경, 사용자 사진, 생성 결과는 삭제하지 않았습니다.

```bash
python -m unittest discover -s tests
```
