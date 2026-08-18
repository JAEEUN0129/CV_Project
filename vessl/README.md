# GPU 서버(VESSL)에서 헤어스타일 변경 실행하기

이 프로젝트의 성능 병목은 전부 계산 장비에 있다. 같은 코드가 그래픽 가속 없는 노트북에서는
사진 한 장에 2~5분, GPU 서버에서는 약 10초 걸린다. 코드는 장치를 자동으로 감지하므로
옮길 때 수정할 것은 없다.

## 실측 (11프레임 영상 기준)

| | 노트북 (CPU) | GPU 서버 |
|---|---|---|
| 사진 1장 | 2~5분 | 약 10초 |
| 영상 11프레임 | 33분 | 2~3분 |
| 영상 80프레임(10초) | 4~7시간 | 약 15분 |

두 환경의 결과물은 흔들림 지표가 소수점 둘째 자리까지 일치한다. 프레임별 픽셀 차이는
15/255 수준으로, 부동소수점 연산 순서 차이에서 오는 것이며 눈으로는 구분되지 않는다.

---

## 1. 최초 1회 — 내 컴퓨터

```powershell
pip install vessl
vessl configure
vessl ssh-key add
```

## 2. 워크스페이스 만들기 (웹 콘솔)

| 항목 | 값 |
|---|---|
| 리소스 | GPU 1장이면 충분. 저렴한 것부터 고른다 |
| 이미지 | **CUDA가 포함된 PyTorch 이미지** (예: `quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5`) |
| 디스크 | 50 GiB (실제 사용량은 약 14GB) |
| Volumes | **비워둔다.** `/root`가 기본으로 영구 저장된다 |
| 시작 스크립트 | 비워둔다. 첫 실행은 터미널에서 눈으로 확인하는 편이 낫다 |

**사용 가능한 노드를 전부 체크하면 실패할 수 있다.** 디스크가 가득 찬 노드에 배정되면
워크스페이스가 시작조차 못 하고 `No space left on device`로 재시작을 반복한다. 이때는
**노드를 하나씩만 체크해서 성공하는 것을 찾는다.**

## 3. 접속 — VS Code

`Remote - SSH` 확장을 설치한 뒤:

```
F1 → "Remote-SSH: Connect to Host..." → 서버 주소 선택
왼쪽 아래 "SSH: <주소>" 표시 확인
File → Open Folder → /root
Terminal → New Terminal
```

이 터미널이 곧 서버다. 따로 `ssh` 명령을 칠 필요가 없다. 폴더를 열어두면 결과 이미지를
내려받지 않고 클릭만으로 확인할 수 있다.

## 4. 서버 설치 — 워크스페이스를 새로 만들 때마다

```bash
# 상태 확인. GPU가 안 보이면 여기서 중단한다.
df -h /root && nvidia-smi

# 화면이 없는 서버 이미지에는 OpenCV가 요구하는 그래픽 라이브러리가 없다.
apt-get update && apt-get install -y libgl1 libglib2.0-0

# 코드
cd /root
git clone https://github.com/JAEEUN0129/CV_Project
cd CV_Project
git checkout jaeeun
git clone --depth 1 https://github.com/AIRI-Institute/HairFastGAN external/HairFastGAN
python jaeeun/patch_hairfastgan.py

# 가중치 7.3GB (10~20분). 끊기면 같은 명령을 다시 치면 이어받는다.
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('AIRI-Institute/HairFastGAN', local_dir='/root/CV_Project/external/HairFastGAN_weights', max_workers=4)"

# 모델 코드가 가중치를 자기 폴더 안에서 상대 경로로 찾는다.
ln -s /root/CV_Project/external/HairFastGAN_weights/pretrained_models external/HairFastGAN/pretrained_models
pip install -r jaeeun/requirements-hair.txt
```

## 5. 동작 확인

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

**`DEVICE cuda`가 찍혀야 한다.** `cpu`가 나오면 GPU를 못 쓰는 상태이므로 그대로 두면
가속 없이 돈다. 첫 실행은 부가 가중치 두 개(합계 380MB)를 자동으로 받느라 1분쯤 걸리고,
이후로는 30초 안팎이다.

## 6. 영상 프레임 처리

프레임은 저장소에 없으므로 직접 올린다. VS Code 파일 목록으로 끌어다 놓는 것이 가장 쉽다.
명령으로 할 경우 **내 컴퓨터**에서 실행한다(서버 터미널이 아니다).

```powershell
scp -P <포트> -r "artifacts\jaeeun\frames" root@<주소>:/root/video_in
```

서버에서 실행. 참조 사진 4종은 저장소에 포함되어 있어 따로 올릴 필요가 없다.

```bash
cd /root/CV_Project/external/HairFastGAN
time python /root/CV_Project/jaeeun/hair_runner.py \
  --faces /root/video_in \
  --reference "/root/CV_Project/data/jaeeun/hairstyles/앞머리 웨이브.jpg" \
  --output /root/video_out \
  --repo /root/CV_Project/external/HairFastGAN \
  --weights /root/CV_Project/external/HairFastGAN_weights/pretrained_models
```

결과를 내 컴퓨터로 가져와 영상으로 합친다.

```powershell
scp -P <포트> -r root@<주소>:/root/video_out "artifacts\jaeeun\gpu_out"
.\.venv\Scripts\python.exe .\jaeeun\frames_to_video.py --frames artifacts\jaeeun\gpu_out --output artifacts\jaeeun\outputs\gpu_result.mp4 --fps 7.5
```

`--fps`는 프레임을 뽑은 실제 비율이다. 30fps 영상을 4프레임마다 뽑았으면 8이 아니라 7.5다.
잘못 넣으면 재생 속도가 어긋난다.

## 7. 마칠 때

**워크스페이스를 중지한다.** 켜둔 시간만큼 과금된다.

껐다 켜면 `/root` 바깥이 초기화되므로 라이브러리를 다시 설치해야 한다. 코드와 가중치는
`/root` 안이라 남는다.

```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0
cd /root/CV_Project && pip install -r jaeeun/requirements-hair.txt
```

계속 쓸 예정이라면 `/root` 안에 가상환경을 만들어 두면 이 과정이 없어진다.

---

## 걸렸던 문제와 원인

| 증상 | 원인 | 대응 |
|---|---|---|
| 워크스페이스가 시작 직후 `No space left on device`로 반복 재시작 | 배정된 노드의 디스크가 가득 참 | 노드를 하나씩 지정해 성공하는 것을 찾는다 |
| `ImportError: libGL.so.1` | 화면 없는 이미지에 OpenCV가 요구하는 라이브러리가 없음 | `libgl1`, `libglib2.0-0` 설치 |
| numpy가 2.x로 올라가며 모델 코드가 깨짐 | OpenCV 최신 판이 numpy 2를 끌어옴 | 버전을 고정해 설치 (`requirements-hair.txt`에 반영됨) |
| `scp`가 `Could not resolve hostname c` | 서버 터미널에서 실행함 | 파일 전송 명령은 **내 컴퓨터**에서 실행한다 |
