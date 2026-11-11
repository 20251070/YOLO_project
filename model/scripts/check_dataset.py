"""단일 클래스 YOLO 경계 상자 라벨과 이미지·라벨 파일의 대응 관계를 검사합니다."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(ROOT / 'data/pedestrian'))
    args = p.parse_args()
    base = Path(args.data_dir)
    errors = []
    counts = {}
    for split in ('train', 'val', 'test'):
        image_dir = base / 'images' / split
        label_dir = base / 'labels' / split
        images = sorted(x for x in image_dir.glob('*') if x.suffix.lower() in EXTS)
        labels = list(label_dir.glob('*.txt'))
        counts[split] = len(images)
        for label in labels:
            if not any((image_dir / (label.stem + ext)).exists() for ext in EXTS):
                errors.append(f'{label}: 대응하는 이미지가 없는 라벨입니다.')
        for image in images:
            label = label_dir / f'{image.stem}.txt'
            if not label.exists():
                errors.append(f'{image}: 라벨이 없습니다(사람이 없으면 빈 .txt 파일을 만드세요).')
                continue
            for line_number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
                parts = line.split()
                try:
                    if len(parts) != 5 or parts[0] != '0':
                        raise ValueError('클래스 0과 좌표값 네 개가 필요합니다.')
                    x, y, w, h = map(float, parts[1:])
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                            and x-w/2 >= -1e-5 and x+w/2 <= 1+1e-5
                            and y-h/2 >= -1e-5 and y+h/2 <= 1+1e-5):
                        raise ValueError('경계 상자가 정규화된 이미지 범위를 벗어났습니다.')
                except ValueError as e:
                    errors.append(f'{label}:{line_number}: {e}')
    print('이미지 수:', counts)
    if counts['train'] == 0 or counts['val'] == 0:
        errors.append('학습용과 검증용 데이터에는 이미지가 있어야 합니다.')
    if errors:
        print('\n'.join(errors[:50]))
        raise SystemExit(f'검사 실패: 문제 {len(errors)}건')
    print('데이터셋 형식이 올바릅니다(이미지 손상 여부와 데이터 누출 여부는 별도로 직접 확인해야 합니다).')


if __name__ == '__main__':
    main()
