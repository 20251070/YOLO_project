"""학습에 사용하지 않은, 라벨이 있는 시험용 이미지로 평가합니다."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    metrics = YOLO(a.weights).val(data=str(ROOT / 'data/pedestrian/data.yaml'),
                                  split='test', device=a.device,
                                  project=str(ROOT / 'runs'), name='test_evaluation')
    print(f'정밀도={metrics.box.mp:.4f} 재현율={metrics.box.mr:.4f} '
          f'mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}')


if __name__ == '__main__':
    main()
