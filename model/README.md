# RightTurn Weather + Camera (우회전 보행자 감지 연구·교육용 시제품)

## 1. 목표
YOLO11n을 추가 학습하고 웹캠, USB 카메라, 영상 파일 또는 RTSP/IP 카메라에 적용하는 시연 프로젝트입니다. 현재 저장소의 Roboflow 이미지 61장과 라벨 61개를 model/data/pedestrian으로 가져왔습니다. 주간/야간/우천별 현장 데이터는 별도로 추가해야 합니다.

**중요:** 보행자 신호는 **시연자가 수동으로 입력**합니다(`g`: 녹색, `r`: 적색, `u`: 알 수 없음). 적색 신호 진입은 법적 판단이 아닌 *의심 상황*입니다. 이 프로젝트는 **인증된 교통 안전 장치가 아니며**, 화면 표시를 우회전해도 된다는 허가로 해석해서는 안 됩니다.

## 2. 설치

```powershell
cd RightTurn-Weather-Camera
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

카메라 화면의 한글은 Pillow로 표시합니다. Windows에서는 기본 제공되는 맑은 고딕을 사용합니다. 다른 운영체제에서는 나눔고딕, Noto Sans CJK 또는 AppleGothic 글꼴이 필요합니다.

## 3. 학습 데이터

명령은 이 `model` 폴더에서 필요한 패키지가 설치된 Python 환경으로 실행합니다.
원본 `pedestrian crosswalk detection.v2i.yolov11` 폴더는 보존하고 아래 구조로 복사했습니다.
원본의 `valid` 분할은 `val`로 맞췄습니다.

```text
data/pedestrian/
  data.yaml
  images/train/   # 43장
  images/val/     # 12장
  images/test/    # 6장
  labels/train/  # 43개
  labels/val/    # 12개
  labels/test/   # 6개
```

클래스는 **0: bicycle, 1: humans**입니다. 사람을 0으로 바꾸면 자전거와 섞이므로 원본 번호를 유지합니다.
YOLO 라벨 한 줄은 `클래스번호 중심_x 중심_y 너비 높이`이며 좌표는 0`1로 정규화합니다.
객체가 없는 이미지는 같은 이름의 빈 `.txt` 파일을 둡니다.
카메라 시연은 학습한 `humans`를 보행자로 추적하며 기본 COCO 모델의 `person`도 지원합니다.

라이선스와 출처는 `data/pedestrian/SOURCE.md`, `README.dataset.txt`, `README.roboflow.txt`에 보존했습니다.
학습 이미지와 라벨은 Git 제외 규칙을 제거해 소스트리에서 관리할 수 있습니다.
원본 영상, `runs/`, 모델 `*.pt`, 라벨 캐시 `*.cache`는 계속 제외합니다.

이 데이터는 61장의 소규모 샘플이며 파일명상 같은 영상의 프레임이 여러 분할에 포함됩니다.
최종 성능은 독립된 촬영 영상 및 주간/야간/우천별 데이터로 평가해야 합니다.
직접 데이터를 추가할 때는 촬영 영상 단위로 train/val/test를 먼저 분리하고 같은 클래스 정의로 라벨링하세요.

```powershell
python scripts/extract_frames.py --video data/raw/night/night01.mp4 --out data/staging/night01 --every-sec 2
```

## 4. 데이터 검사 및 추가 학습

```powershell
# 이미지 읽기, 라벨 대응, 클래스 및 좌표 검사
python scripts/check_dataset.py

# 학습하지 않고 학습 코드의 데이터 연결만 확인
python scripts/train.py --check-only

# CPU로 짧은 실행 확인
python scripts/train.py --epochs 1 --batch 2 --imgsz 320 --device cpu --name dataset_smoke

# NVIDIA GPU 환경이 준비된 경우 본 학습
python scripts/train.py --epochs 40 --batch 8 --device 0 --name weather_person_v1
```

GPU가 없으면 본 학습에도 `--device cpu --batch 2`를 사용할 수 있지만 오래 걸릴 수 있습니다.
기본 데이터는 `data/pedestrian/data.yaml`이고 다른 YAML은 `--data 경로/data.yaml`로 지정합니다.
검사기는 `images/<분할>`, `labels/<분할>` 구조를 사용합니다.
YAML의 `path`는 생략하거나 절대 경로로 지정하세요. 기본 설정에는 절대 경로가 없어 프로젝트를 옮겨도 동작합니다.
기존 모델에서 더 학습하려면 `--weights 경로/best.pt`, 학습 그래프를 저장하려면 `--plots`를 추가합니다.

완료 시 실제 결과 폴더가 출력됩니다. 같은 이름이 있으면 `weather_person_v12`처럼 새 이름이 만들어질 수 있습니다.
아래는 본 학습 결과가 `runs/weather_person_v1`에 생성된 경우입니다.

```powershell
python scripts/evaluate.py --weights runs/weather_person_v1/weights/best.pt
python run_demo.py --source data/raw/crosswalk.mp4 --model runs/weather_person_v1/weights/best.pt
python run_demo.py --source 0 --model runs/weather_person_v1/weights/best.pt
```

`dataset_smoke`는 학습 연결을 확인하기 위한 모델이며 성능 검증을 마친 최종 모델이 아닙니다.

## 5. 영상 및 카메라 실행

모든 명령은 `model` 폴더에서 실행합니다. 기본 모델은 `yolo11n.pt`입니다.

```powershell
# 저장 영상
python run_demo.py --source data/raw/crosswalk.mp4 --model yolo11n.pt
# 내장 카메라
python run_demo.py --source 0 --model yolo11n.pt
# USB 웹캠 (장치에 따라 0, 1, 2 등으로 변경)
python run_demo.py --source 1 --model yolo11n.pt
# 추가 학습 모델
python run_demo.py --source 1 --model runs/weather_person_v1-2/weights/best.pt
```

**실행할 때마다 현재 첫 프레임에서 횡단보도를 자동 탐색합니다.**
기존 ROI 파일이 있어도 새로 탐색하며, 성공하면 같은 파일을 갱신합니다.
자동 탐색은 주변 도로와의 명암 차이 및 반복되는 평행 도색 띠를 이용한 OpenCV 추정입니다. 화면 전체를 탐색하며 횡단보도 전용 학습 모델은 아닙니다.
실패하면 오래된 저장 영역을 불러오지 않고 전체 화면의 보행자 감지를 계속합니다.

- **a**: 현재 화면에서 자동 탐색만 다시 실행합니다. 영상과 모델을 다시 불러오지 않습니다.
- **z**: 경계에 점을 찍어 **다각형 영역**을 직접 지정합니다. 대각선은 모서리를, 굴곡은 점을 촘촘히 찍어 둘러쌉니다.
  왼쪽 클릭으로 점 추가, Enter/Space로 확정, 오른쪽 클릭/Backspace로 마지막 점 취소,
  X로 다시 그리기, C/Esc로 취소합니다. 취소하면 현재 영역을 유지합니다.
  선이 서로 교차하거나 영역 넓이가 0이면 확정하지 않고 수정할 수 있게 안내합니다.
  오목한 다각형도 지원하며, 보행자 발 위치가 실제 다각형 안에 있는지 판단합니다.
  새 다각형은 같은 ROI 파일의 roi 항목에 정규화된 [[x,y], ...] 좌표로 저장합니다.
  기존 [x,y,너비,높이] 사각형 형식과 자동 탐색 결과도 계속 지원합니다.
- a 재탐색에 실패하면 현재 사용 중인 영역과 저장 파일을 유지합니다.
- 영역을 성공적으로 갱신하면 분석을 재개하고 새 영역으로 횡단 여부를 판단합니다.
- **p**: 분석 일시 정지/재개, **q** 또는 창 닫기: 종료
- **g / r / u**: 보행자 신호를 녹색 / 적색 / 알 수 없음으로 수동 설정

기본 ROI 저장 위치는 카메라 `0`이면 `config/camera_0_roi.json`, 카메라 `1`이면 `config/camera_1_roi.json`,
영상 파일이면 `config/crosswalk_roi.json`, IP 스트림이면 `config/ip_roi.json`입니다.
첫 저장 이후에는 이 파일을 계속 덮어씁니다. 매 실행 또는 a 입력마다 새로운 파일을 만들지 않습니다.
다른 고정 경로를 사용하려면 `--roi-file config/my_camera_roi.json`을 지정하세요.
상대 ROI 경로는 현재 터미널 위치와 관계없이 `model` 폴더 기준으로 해석합니다.

```powershell
# 같은 ROI 파일을 계속 갱신
python run_demo.py --source 1 --model yolo11n.pt --roi-file config/my_camera_roi.json
# 카메라 연결 확인만 수행 (자동 탐색 및 모델 분석은 하지 않음)
python run_demo.py --source 0 --check-camera
# Windows 카메라 연결 방식 지정
python run_demo.py --source 0 --camera-backend dshow
# 필요할 때만 분석 화면을 영상으로 저장
python run_demo.py --source data/raw/crosswalk.mp4 --output recorded.mp4
```

영상 파일은 가능한 경우 하드웨어 디코딩을 사용하며, 지원하지 않으면 기본 디코더로 복귀합니다.
고해상도 영상의 기본 재생은 원본 시간에 맞춰 중간 프레임을 건너뜁니다.
원본 60 FPS 영상의 모든 프레임을 모델로 분석한다는 의미는 아닙니다. 실제 분석 FPS는 CPU/GPU에 따라 달라집니다.
기본 분석·그리기 크기는 가장 긴 변 1280픽셀로 제한하고 영상 비율과 정규화 ROI 좌표는 유지합니다.
`--max-frame-size 0`으로 원본 처리 크기를 유지할 수 있으나 느려질 수 있습니다.
저장 영상에서 모든 프레임을 분석하려면 `--video-mode all`을 사용하세요.
`--output`이 있으면 기본적으로 전체 프레임 처리로 전환하여 저장 영상의 시간축을 유지합니다.
명시적으로 `--video-mode realtime --output ...`을 함께 쓰면 분석된 프레임만 저장하므로 재생 시간이 짧아질 수 있습니다.
저장되는 영상의 해상도도 처리 크기를 따릅니다.

```powershell
# 4K/60 FPS 영상: 시간에 맞춰 프레임 선택 + 처리 크기 제한
python run_demo.py --source data/raw/crosswalk2.mp4 --model yolo11n.pt

# 원본 크기로 모든 프레임 분석 (빠른 재생보다 분석 보존이 필요한 경우)
python run_demo.py --source data/raw/crosswalk2.mp4 --model yolo11n.pt --video-mode all --max-frame-size 0
```

카메라 수신은 별도 스레드에서 계속 처리하며 최신 프레임을 분석합니다.
첫 프레임에 횡단보도가 없으면 카메라를 맞춘 뒤 a를 눌러 다시 찾을 수 있습니다.
카메라 위치가 바뀌었을 때도 a 또는 z를 사용하세요.
`--output`을 지정했을 때만 영상으로 저장하며, 기존 영상 경로 탐색과 한글 글꼴 지원은 동일합니다.
RTSP 접속 정보와 촬영 영상을 GitHub에 올리지 마세요.

## 6. 권장 시험 사례

- 주간: 보행자 없음 / 대기 중 / 횡단 중
- 야간: 어두운 환경의 보행자 / 역광 또는 전조등 눈부심
- 우천: 우산 / 반사 / 젖은 노면 / 일부 가림
- 신호: 시연자가 g/r/u 키로 수동 설정하고 상태 처리 로직**만** 확인
- 카메라: 웹캠과 저장 영상을 각각 시험하고 지연, 연결 끊김, 프레임 누락 확인

객체 감지 지표는 정밀도(P), 재현율(R), 평균 정밀도(mAP50/mAP50-95)입니다. 시스템 지표는 횡단 감지 재현율, 오경고 빈도, 경고 지연 시간입니다. 카메라 연결이 끊기거나 결과가 불확실할 때 화면에 우회전이 안전하다고 표시해서는 안 됩니다.

## 7. 한계 및 향후 과제

현재 보행자 신호는 영상에서 자동으로 판별하지 않습니다. 별도의 신호 인식 기능에는 전용 지도 학습 데이터셋과 평가가 필요합니다. 사각형 관심 영역과 발 위치를 이용한 추정은 특이한 촬영 각도에서 부정확할 수 있습니다. 이 프로젝트는 실제 도로에서 운영하는 안전 시스템이 아닌 *실험실 시연용*입니다.

공식 참고 자료:
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/modes/track/
- https://docs.ultralytics.com/datasets/detect/
