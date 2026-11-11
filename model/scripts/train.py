"""주간·야간·우천 환경의 보행자 감지를 위해 YOLO11n을 추가 학습합니다.
날씨별 클래스는 사용하지 않으며 모든 사람을 클래스 0으로 표시합니다.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='yolo11n.pt')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu', help='CPU는 cpu, CUDA GPU는 0을 지정합니다.')
    p.add_argument('--name', default='weather_person_v1')
    p.add_argument('--workers', type=int, default=0, help='Windows에서는 0을 권장합니다.')
    a = p.parse_args()
    data = ROOT / 'data/pedestrian/data.yaml'
    # Ultralytics는 데이터셋 YAML 파일의 디렉터리를 기준으로 YAML의 경로를 해석합니다.
    model = YOLO(a.weights)
    model.train(data=str(data), epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
                device=a.device, workers=a.workers, project=str(ROOT / 'runs'),
                name=a.name, seed=42, patience=12)
    print('결과 저장 위치:', ROOT / 'runs' / a.name)


if __name__ == '__main__':
    main()
