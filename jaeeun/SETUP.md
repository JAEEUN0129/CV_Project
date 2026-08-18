# Jaeeun 실행 환경

Python 3.11, Git, FFmpeg가 필요합니다.

환경은 **두 개**입니다. 헤어스타일 변경 모델이 요구하는 라이브러리 버전이 나머지 기능과
충돌하기 때문에 섞어 설치하면 양쪽 다 망가집니다.

| 환경 | 담당 | 없으면 |
|---|---|---|
| `.venv` | 부위 분할, 영역 추적, 색 변경, 영상 처리, 웹 화면 | 아무것도 못 함 |
| `.venv_hair` | 헤어스타일 변경 | 나머지 기능은 정상, 헤어스타일만 비활성 |

## 1. 주 환경 (`.venv`)

```powershell
cd C:\yai_18_여름방학\CV_Project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r .\jaeeun\requirements.txt
.\.venv\Scripts\python.exe -m pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
```

SAM 2와 Diffusion 가중치를 받습니다. 오래 걸리므로 별도 창에서 돌리는 편이 좋습니다.

```powershell
.\.venv\Scripts\python.exe .\jaeeun\download_models.py
```

웹 화면 실행:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\jaeeun\app.py
```

## 2. 헤어스타일 환경 (`.venv_hair`)

모델 코드와 가중치를 받습니다. 가중치가 7.3GB이므로 시간이 걸립니다.

```powershell
git clone --depth 1 https://github.com/AIRI-Institute/HairFastGAN external\HairFastGAN
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('AIRI-Institute/HairFastGAN', local_dir='external/HairFastGAN_weights')"
```

모델 코드는 가중치를 자기 폴더 안에서 찾으므로 연결해 줍니다.

```powershell
New-Item -ItemType Junction -Path external\HairFastGAN\pretrained_models -Target external\HairFastGAN_weights\pretrained_models
New-Item -ItemType Junction -Path external\HairFastGAN\input -Target external\HairFastGAN_weights\input
```

격리 환경을 만들고 라이브러리를 설치합니다.

```powershell
python -m venv .venv_hair
.\.venv_hair\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv_hair\Scripts\python.exe -m pip install -r .\jaeeun\requirements-hair.txt
```

### 그래픽카드가 없는 환경에서만 필요한 수정

HairFastGAN은 NVIDIA 그래픽카드를 전제로 작성되어 있어, 없는 환경에서는 세 군데를 고쳐야
합니다. 계산 로직은 건드리지 않고 장치 관련 부분만 바꿉니다.

1. `models/**/op/fused_act.py`와 `upfirdn2d.py` (각 3개) — 파일 맨 위의 `load(...)` 호출을
   `try` / `except`로 감싸 실패해도 넘어가게 합니다. 함수 안쪽에는 이미 CPU 경로가 있습니다.
2. 소스 전체의 `device='cuda'`를 `device=('cuda' if torch.cuda.is_available() else 'cpu')`로
   바꿉니다.
3. `utils/bicubic.py`의 `__init__` 기본값 `cuda=True`를 실제 장치에 따르도록 바꾸고,
   `utils/image_utils.py`의 `hair_from_mask`에서 `F.interpolate` 인자를 `mask.float()`로
   바꿉니다. CPU에는 정수 자료형용 최근접 보간 커널이 없습니다.

`.venv_hair`가 없어도 웹 화면은 실행되며, 헤어스타일 기능만 "준비되지 않음"으로 표시됩니다.

## 3. 동작 확인

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from jaeeun.pipeline import VirtualFittingPipeline; print(VirtualFittingPipeline().parse_image(Path('data/jaeeun/hairstyles/앞머리 웨이브.jpg')))"
```

## 참고

- 헤어스타일 참조 사진은 `data/jaeeun/hairstyles/`에 넣습니다. 파일을 넣으면 웹 화면의
  선택지가 늘어납니다. 현재 들어 있는 사진의 사용 조건은 해당 폴더의 README를 확인하세요.
- 그래픽 가속이 없는 환경에서 헤어스타일 변경은 사진 한 장에 2~5분 걸립니다.
- 모델과 데이터셋의 라이선스는 사용 전에 다시 확인합니다.
