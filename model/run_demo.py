"""필수 패키지와 SSL이 정상인 파이썬 환경으로 횡단보도 영상을 실행합니다."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    home = Path.home()
    candidates = [
        ROOT / '.venv/Scripts/python.exe',
        ROOT / '.venv/bin/python',
    ]
    if os.environ.get('CONDA_PREFIX'):
        candidates.append(Path(os.environ['CONDA_PREFIX']) / 'python.exe')
    candidates.extend([
        home / 'anaconda3-3/envs/yolo_project/python.exe',
        home / 'anaconda3/envs/yolo_project/python.exe',
        Path(sys.executable),
    ])
    python_on_path = shutil.which('python')
    if python_on_path:
        candidates.append(Path(python_on_path))
    demo_python = None
    check = 'import ssl, cv2, lap; from PIL import Image; from ultralytics import YOLO'
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run(
                [str(candidate), '-B', '-c', check],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            demo_python = candidate
            break
    if demo_python is None:
        raise SystemExit('SSL과 필수 패키지가 정상인 파이썬 환경이 없습니다. 환경을 활성화하고 requirements.txt를 설치하세요.')
    print(f'사용할 파이썬: {demo_python}', flush=True)
    command = [
        str(demo_python), '-B', str(ROOT / 'scripts/demo_camera.py'),
        '--source', str(ROOT / 'data/raw/crosswalk.mp4'),
        '--model', str(ROOT / 'yolo11n.pt'), *sys.argv[1:],
    ]
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
