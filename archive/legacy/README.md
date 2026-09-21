# 이전 구현 보관

현재 웹에서 호출하지 않는 이전 구현을 이 폴더로 옮겼습니다.
원래 상대 경로를 유지한 보관본이며 이 위치에서 실행하는 패키지가 아닙니다.
이전 동작을 재현하려면 정리 전 Git 커밋을 별도 작업 폴더로 체크아웃하세요.

## 이동한 파일 17개

- HairFastGAN/HairCLIP: `jaeeun/hair_app.py`, `hair_runner.py`, `hairclip.py`,
  `hairclip_runner.py`, `hairstyle.py`, `patch_hairfastgan.py`, `requirements-hair.txt`
- 이전 파이프라인·도구: `jaeeun/pipeline.py`, `data.py`, `video.py`,
  `frames_to_video.py`, `evaluate.py`, `download_models.py`
- 이전 문서·배포: `jaeeun/PROJECT_PLAN.md`, `SETUP.md`,
  `vessl/hairstyle.yaml`, `vessl/README.md`

## 정리 전 스냅샷

원래 README는 `PROJECT_README.md`로 보관했습니다.
`jaeeun/app.py`, `models.py`, `requirements.txt`, `requirements-server.txt`도 보관했습니다.
현재 파일에는 새 UI 진입점·공통 분할 어댑터·필요한 UI 의존성만 남겼습니다.

## 유지한 파일

- `jaeeun/vace/experiment.py`, `specs.py`, `short_hair_mask.py`,
  `jaeeun/prepare_vace_mask.py`: 별도 CLI 실험에서 사용
- `color_candidates.py`의 `write_manifest`, `select_candidate`: 실험 CLI에서 사용
- `external/`, `checkpoints/`, `.venv*`, `artifacts/`, `data/`: 모델·환경·사용자 자료 보존

## 파일 내부 정리

- 구형 `app.py`의 실행되지 않던 UI 제거; 현재 UI 호환 진입점으로 교체
- `models.py`의 구형 SAM 2, Stable Diffusion, 색 변경 어댑터 분리
- 미사용 `digest`, `candidate_palette`와 사용하지 않는 인자 제거

이번 정리는 저장 공간 확보가 아니라 활성 코드와 이전 실험의 구분을 위한 정리입니다.
