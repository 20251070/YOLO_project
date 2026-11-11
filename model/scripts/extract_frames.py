"""수동 라벨 작성을 위해 프레임을 추출하며 파일명에 원본 영상 이름을 유지합니다.
촬영 영상별로 실행하고 라벨 작성 전에 원본 영상 단위로 데이터를 분할합니다.
"""
import argparse
from pathlib import Path
import cv2

if __package__:
    from .video_source import resolve_video_path
else:
    from video_source import resolve_video_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--every-sec', type=float, default=2.0)
    a = p.parse_args()
    if a.every_sec <= 0:
        p.error('--every-sec 값은 0보다 커야 합니다.')
    try:
        src = resolve_video_path(a.video)
    except FileNotFoundError as exc:
        p.error(str(exc))
    output = Path(a.out)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {src}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    step = max(1, round(fps * a.every_sec))
    index = count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            name = f'{src.stem}_f{index:08d}.jpg'
            if not cv2.imwrite(str(output / name), frame):
                raise SystemExit(f'파일 저장에 실패했습니다: {name}')
            count += 1
        index += 1
    cap.release()
    print(f'{count}개 프레임을 추출했습니다. 저장 위치: {output}')


if __name__ == '__main__':
    main()
