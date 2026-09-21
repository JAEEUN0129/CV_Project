# 현재 웹 실행 환경

## 웹·분할 환경

CUDA에 맞는 PyTorch와 Torchvision을 먼저 준비한 뒤 저장소 루트에서 실행합니다.

```bash
python -m pip install -r jaeeun/requirements-server.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m streamlit run jaeeun/vace_app.py --server.address 0.0.0.0 --server.port 8501
```

현재 서버처럼 PyTorch 2.3.1을 유지할 때는 `transformers==4.45.2` 조합을 사용합니다.
서버 목록에는 HairFastGAN/HairCLIP/SAM 2 의존성을 포함하지 않습니다.

## FLUX 환경

별도 환경에서 `requirements-flux.txt`를 설치합니다.
`FLUX_PYTHON`은 해당 Python을 가리켜야 합니다. 예: `/root/flux-env/bin/python`.
FLUX의 라이브러리를 웹 환경에 덮어 설치하지 않습니다.

## VACE 환경

외부 VACE 저장소와 Wan 모델 가중치가 필요합니다.
현재 `runner.py`는 PATH의 `python`을 호출하므로 그 Python에서 `import wan`이 가능해야 합니다.
다른 VACE 환경을 사용하려면 실행기의 인터프리터 지정 기능을 먼저 연결해야 합니다.
Wan/FlashAttention 설치 문제는 이번 파일 정리로 해결되는 범위가 아닙니다.

## 설정

| 환경 변수 | 용도 |
|---|---|
| `FLUX_PYTHON` | FLUX 전용 Python |
| `FLUX_WORKER` | FLUX worker 경로 |
| `VACE_REPO` | 외부 VACE 저장소 |
| `VACE_MODEL_DIR` | Wan/VACE 가중치 |
| `VACE_WORKSPACE` | 작업별 결과 저장 루트 |
| `PERSONAL_COLOR_CHECKPOINT` | 퍼스널컬러 체크포인트; 없으면 기본 팔레트 사용 |

모델은 첫 실행 시 다운로드될 수 있습니다.
기존 설치 기록은 `archive/legacy/jaeeun/SETUP.md`에 보관했습니다.
