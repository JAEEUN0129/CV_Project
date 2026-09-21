# GPU 실행 파일

- `vace-ui.yaml`: 현재 Streamlit 웹 배포 명세
- `vace-smoke.yaml`: VACE 단독 스모크 실험
- `run_vace_style.sh`: 스타일별 VACE 실험

실행 환경은 [설치 안내](../jaeeun/SETUP.md)를 참고하세요.
명세의 Dataset·GPU·모델 경로는 실제 서버에 맞게 지정해야 합니다.
외부 VACE/Wan 의존성 설치는 기존 PyTorch와 호환되는 환경을 먼저 준비해야 합니다.
이전 HairFastGAN 배포 명세와 기록은 `archive/legacy/vessl/`에 보관했습니다.
