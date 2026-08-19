# GPU 서버(VESSL)에서 실행하기

이 프로젝트의 성능 병목은 전부 계산 장비에 있다. 같은 코드가 그래픽 가속 없는 노트북에서는
사진 한 장에 2~5분, GPU 서버에서는 약 10초 걸린다. 코드는 장치를 자동으로 감지하므로
옮길 때 수정할 것은 없다.

## 실측

| | 노트북 (CPU) | GPU 서버 |
|---|---|---|
| 사진 1장 | 2~5분 | 약 10초 |
| 영상 11프레임 | 33분 | 2~3분 |
| 영상 80프레임(10초) | 4~7시간 | 약 15분 |

두 환경의 결과물은 흔들림 지표가 소수점 둘째 자리까지 일치했다. 프레임별 픽셀 차이는
15/255 수준으로, 부동소수점 연산 순서 차이에서 오는 것이며 눈으로는 구분되지 않는다.

---

## A. 최초 1회 — 내 컴퓨터

```powershell
pip install vessl
vessl configure
vessl ssh-key add
```

## B. 워크스페이스 만들기 (웹 콘솔)

| 항목 | 값 |
|---|---|
| 리소스 | GPU 1장이면 충분. 저렴한 것부터 고른다 |
| 이미지 | **CUDA가 포함된 PyTorch 이미지** (예: `quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5`) |
| 디스크 | 50 GiB (실제 사용량은 약 20GB) |
| Volumes | **비워둔다.** `/root`가 기본으로 영구 저장된다 |
| 시작 스크립트 | 비워둔다. 첫 실행은 터미널에서 눈으로 확인하는 편이 낫다 |
| 포트 | **추가하지 않아도 된다.** 웹 화면은 SSH 통로로 연결한다 |

**사용 가능한 노드를 전부 체크하면 실패할 수 있다.** 디스크가 가득 찬 노드에 배정되면
워크스페이스가 시작조차 못 하고 `No space left on device`로 재시작을 반복한다. 이때는
**노드를 하나씩만 체크해서 성공하는 것을 찾는다.**

## C. 접속

워크스페이스를 새로 만들면 **주소와 포트가 바뀐다.** 콘솔의 SSH 항목에서 확인한다.

### 작업용 — VS Code

`Remote - SSH` 확장을 설치한 뒤:

```
F1 → "Remote-SSH: Connect to Host..." → 주소 선택
왼쪽 아래 "SSH: <주소>" 표시 확인
File → Open Folder → /root
Terminal → New Terminal
```

이 터미널이 곧 서버다. 폴더를 열어두면 결과 이미지를 내려받지 않고 클릭만으로 확인할 수 있고,
파일을 끌어다 놓아 업로드할 수 있다.

### 웹 화면용 — SSH 통로

브라우저로 화면을 열려면 서버의 8501번을 내 컴퓨터로 끌어와야 한다. **내 컴퓨터**에서:

```powershell
ssh -L 8501:localhost:8501 -p <포트> root@<주소>
```

워크스페이스에 포트를 추가하는 방법도 있지만, 포트 설정은 **중지 상태에서만** 바꿀 수 있고
중지에 몇 분이 걸린다. 통로 방식이 빠르고, 화면이 외부에 노출되지 않아 더 안전하다.

VS Code를 쓴다면 하단 패널의 **Ports** 탭에서 8501을 추가해도 같은 효과가 난다.

## D. 서버 설치 — 워크스페이스를 새로 만들 때마다

```bash
# 상태 확인. GPU가 안 보이면 여기서 중단한다.
df -h /root && nvidia-smi

# 화면 없는 서버 이미지에 빠져 있는 것들. ffmpeg은 영상 재조합에 쓴다.
apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg

# 코드
cd /root
git clone https://github.com/JAEEUN0129/CV_Project
cd CV_Project
git checkout jaeeun
git clone --depth 1 https://github.com/AIRI-Institute/HairFastGAN external/HairFastGAN
python jaeeun/patch_hairfastgan.py

# 헤어스타일 모델 가중치 7.3GB (10~20분). 끊기면 같은 명령을 다시 치면 이어받는다.
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('AIRI-Institute/HairFastGAN', local_dir='/root/CV_Project/external/HairFastGAN_weights', max_workers=4)"

# 모델 코드가 가중치를 자기 폴더 안에서 상대 경로로 찾는다.
ln -s /root/CV_Project/external/HairFastGAN_weights/pretrained_models external/HairFastGAN/pretrained_models

# 라이브러리. 이미지에 이미 있는 torch 계열은 목록에서 빠져 있다 —
# 다시 설치하면 CPU 판으로 덮여 가속이 사라진다.
pip install -r jaeeun/requirements-server.txt
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"

# 색 변경이 쓰는 가중치. 디퓨전은 보류 상태이므로 오래 걸리면 중간에 끊어도 된다.
python jaeeun/download_models.py
```

부위 분할 모델(250MB)은 첫 실행 때 자동으로 받아진다.

**이미 설치를 마친 워크스페이스라면 최신 코드만 받는다:**

```bash
cd /root/CV_Project && git pull
```

## E. 동작 확인

```bash
mkdir -p /root/test_in
cp /root/CV_Project/external/HairFastGAN_weights/input/6.png /root/test_in/
cd /root/CV_Project/external/HairFastGAN
time python /root/CV_Project/jaeeun/hair_runner.py \
  --faces /root/test_in \
  --reference /root/CV_Project/external/HairFastGAN_weights/input/2.png \
  --output /root/test_out \
  --repo /root/CV_Project/external/HairFastGAN \
  --weights /root/CV_Project/external/HairFastGAN_weights/pretrained_models
```

**`DEVICE cuda`가 찍혀야 한다.** `cpu`가 나오면 가속 없이 도는 것이다. 첫 실행은 부가 가중치
두 개(합계 380MB)를 자동으로 받느라 1분쯤 걸리고, 이후로는 30초 안팎이다.

## F. 웹 화면 실행

```bash
cd /root/CV_Project
streamlit run jaeeun/app.py --server.port 8501
```

C에서 만든 SSH 통로가 열려 있으면 브라우저에서 **`http://localhost:8501`** 로 접속한다.

터미널을 닫으면 화면도 꺼진다. 계속 띄워두려면:

```bash
nohup streamlit run jaeeun/app.py --server.port 8501 > /root/app.log 2>&1 &
```

서버에는 별도 격리 환경이 없다. 코드가 가상환경을 찾지 못하면 현재 인터프리터를 쓰므로,
D까지 마쳤다면 헤어스타일 기능도 그대로 동작한다.

## G. 명령줄로 영상 처리 (선택)

웹 화면으로 다 되지만, 여러 스타일을 한 번에 돌릴 때는 명령이 편하다.

**프레임 올리기** — VS Code 파일 목록으로 끌어다 놓거나, **내 컴퓨터**에서:

```powershell
scp -P <포트> -r "artifacts\jaeeun\frames" root@<주소>:/root/video_in
```

**실행** — 참조 사진 4종은 저장소에 포함되어 있어 따로 올릴 필요가 없다. 모양과 색을 다른
사진에서 가져오려면 `--shape`와 `--color`를 쓴다.

```bash
cd /root/CV_Project/external/HairFastGAN
time python /root/CV_Project/jaeeun/hair_runner.py \
  --faces /root/video_in \
  --shape "/root/CV_Project/data/jaeeun/hairstyles/앞머리 웨이브.jpg" \
  --color /root/video_in/000000.jpg \
  --output /root/video_out \
  --repo /root/CV_Project/external/HairFastGAN \
  --weights /root/CV_Project/external/HairFastGAN_weights/pretrained_models
```

**미리 확인만** 하려면 `--check-only`를 붙인다. 모델을 올리지 않고 얼굴 검출만 돌려,
11프레임을 45초에 판정한다.

**결과 받아 영상으로 합치기** — 내 컴퓨터에서:

```powershell
scp -P <포트> -r root@<주소>:/root/video_out "artifacts\jaeeun\gpu_out"
.\.venv\Scripts\python.exe .\jaeeun\frames_to_video.py --frames artifacts\jaeeun\gpu_out --output artifacts\jaeeun\outputs\gpu_result.mp4 --fps 7.5
```

`--fps`는 프레임을 뽑은 실제 비율이다. 30fps 영상을 4프레임마다 뽑았으면 8이 아니라 7.5다.
잘못 넣으면 재생 속도가 어긋난다.

**받는 위치가 이미 있으면 그 안에 폴더째 들어간다.** `gpu_out`이 있는 상태로 받으면
`gpu_out/video_out/`이 된다.

## H. 마칠 때

**워크스페이스를 중지한다.** 켜둔 시간만큼 과금된다. 중지에는 1~3분 걸린다.

껐다 켜면 `/root` 바깥이 초기화되므로 라이브러리를 다시 설치해야 한다. 코드와 가중치는
`/root` 안이라 남는다.

```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg
cd /root/CV_Project && git pull && pip install -r jaeeun/requirements-server.txt
```

**"종료하기"는 삭제에 가깝다.** 되돌릴 수 없고 `/root`가 통째로 사라져 7.3GB를 다시 받아야
한다. 중지는 목록의 정지 아이콘이고, 종료는 점 세 개 메뉴 안에 있다.

---

## 걸렸던 문제와 원인

| 증상 | 원인 | 대응 |
|---|---|---|
| 워크스페이스가 시작 직후 `No space left on device`로 반복 재시작 | 배정된 노드의 디스크가 가득 참 | 노드를 하나씩 지정해 성공하는 것을 찾는다 |
| `ImportError: libGL.so.1` | 화면 없는 이미지에 OpenCV가 요구하는 라이브러리가 없음 | `libgl1`, `libglib2.0-0` 설치 |
| numpy가 2.x로 올라가며 모델 코드가 깨짐 | OpenCV 최신 판이 numpy 2를 끌어옴 | 버전을 고정해 설치 (목록에 반영됨) |
| `scp`가 `Could not resolve hostname c` | 서버 터미널에서 실행함 | 파일 전송 명령은 **내 컴퓨터**에서 실행한다 |
| 포트 추가하려는데 중지가 오래 걸림 | 정상. 컨테이너 정리에 시간이 걸린다 | SSH 통로를 쓰면 포트 설정 자체가 필요 없다 |
