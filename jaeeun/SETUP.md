# Jaeeun 실행 환경

Python 3.11, Git, FFmpeg가 필요합니다.

> 이 문서는 **내 컴퓨터에서 실행하는 방법**입니다. 그래픽 가속이 없으면 헤어스타일
> 변경이 사진 한 장에 2~5분 걸립니다. GPU 서버에서 돌리는 방법은
> [../vessl/README.md](../vessl/README.md)를 참고하세요.

환경은 **두 개**입니다. 헤어스타일 변경 모델이 요구하는 라이브러리 버전이 나머지 기능과
충돌하기 때문에 섞어 설치하면 양쪽 다 망가집니다.

| 환경 | 담당 | 없으면 |
|---|---|---|
| `.venv` | 부위 분할, 영역 추적, 색 변경, 영상 처리, 웹 화면 | 아무것도 못 함 |
| `.venv_hair` | 헤어스타일 변경, HairCLIP 텍스트 색상 참조 | 나머지 기능은 정상, 해당 기능만 비활성 |

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

### 받은 모델 코드 수정 (필수)

HairFastGAN은 NVIDIA 그래픽카드를 전제로 작성되어 있어 그대로는 이 프로젝트에서 실행되지
않습니다. 수정은 스크립트로 자동 적용합니다.

```powershell
.\.venv\Scripts\python.exe .\jaeeun\patch_hairfastgan.py
```

고치는 곳은 네 가지이며, 계산 로직은 건드리지 않고 장치 관련 부분만 바꿉니다. 여러 번
실행해도 안전합니다.

| 대상 | 내용 |
|---|---|
| StyleGAN2 연산 (`fused_act`, `upfirdn2d`) | CUDA 커널 컴파일 실패를 허용하고, 실패 시 순수 PyTorch 경로를 쓰도록 분기 조건 수정 |
| 소스 전체의 `device='cuda'` | 실제 사용 가능한 장치를 따르도록 변경 |
| `utils/bicubic.py` | 축소 필터의 장치 기본값을 자동 감지로 변경 |
| `utils/image_utils.py` | 마스크 보간 시 자료형 변환 (CPU에 정수형 최근접 보간 커널이 없음) |

`external/`은 저장소에 포함되지 않으므로, **모델 코드를 새로 받을 때마다 이 스크립트를 다시
실행해야 합니다.**

`.venv_hair`가 없어도 웹 화면은 실행되며, 헤어스타일 기능만 "준비되지 않음"으로 표시됩니다.

## 3. HairCLIP 텍스트 머리색

HairCLIP은 선택한 헤어 프리셋의 색상을 텍스트로 바꾸고, 그 결과를 HairFastGAN의
`color_reference`로 전달합니다. e4e와 얼굴 정렬 가중치는 HairFastGAN에 받은 파일을
재사용합니다.

```powershell
git clone --depth 1 https://github.com/wtybest/HairCLIP.git external\HairCLIP
git clone --depth 1 https://github.com/omertov/encoder4editing.git external\HairCLIP\encoder4editing
.\.venv_hair\Scripts\python.exe -m gdown 1hqZT6ZMldhX3M_x378Sm4Z2HMYr-UwQ4 -O external\HairCLIP\pretrained_models\hairclip.pt
curl.exe -L "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" -o external\HairCLIP\pretrained_models\ViT-B-32.pt
```

기본적으로 `.venv_hair`를 재사용합니다. 별도 환경을 쓰려면 `HAIRCLIP_PYTHON` 환경 변수에
해당 Python 실행 파일을 지정하거나 `.venv_clip`을 만드세요. GPU가 없으면 CPU 호환 경로로
실행되지만 e4e와 StyleGAN2 추론이 매우 느립니다.

## 4. 동작 확인

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from jaeeun.pipeline import VirtualFittingPipeline; print(VirtualFittingPipeline().parse_image(Path('data/jaeeun/hairstyles/앞머리 웨이브.jpg')))"
```

## 참고

- 헤어스타일 참조 사진은 `data/jaeeun/hairstyles/`에 넣습니다. 파일을 넣으면 웹 화면의
  선택지가 늘어납니다. 현재 들어 있는 사진의 사용 조건은 해당 폴더의 README를 확인하세요.
- 그래픽 가속이 없는 환경에서 헤어스타일 변경은 사진 한 장에 2~5분 걸립니다.
- 모델과 데이터셋의 라이선스는 사용 전에 다시 확인합니다.
