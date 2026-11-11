"""실행 위치와 관계없이 프로젝트의 영상 파일 경로를 찾습니다."""
from pathlib import Path
import os
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]


def resolve_video_path(value):
    """현재 폴더, 프로젝트 폴더, 원본 영상 폴더, 스크립트 폴더 순으로 파일을 찾습니다."""
    path = Path(value).expanduser()
    candidates = (path,) if path.is_absolute() else (
        Path.cwd() / path,
        ROOT / path,
        ROOT / 'data/raw' / path,
        Path(__file__).resolve().parent / path,
    )
    checked = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    for candidate in checked:
        if candidate.is_file():
            return candidate
    locations = '\n'.join(f'  - {candidate}' for candidate in checked)
    raise FileNotFoundError(
        f'영상 파일을 찾을 수 없습니다: {value}\n'
        f'확인한 위치:\n{locations}\n'
        '파일 이름과 경로를 확인하거나 절대 경로를 지정하세요.'
    )


def open_video_capture(source, camera_backend='auto'):
    """카메라는 Windows 연결 방식을 순서대로 시도하고 첫 정상 프레임까지 확인합니다."""
    is_camera = isinstance(source, int)
    backends = {
        'dshow': [('DirectShow', cv2.CAP_DSHOW)],
        'msmf': [('Media Foundation', cv2.CAP_MSMF)],
    }
    automatic = [('자동', cv2.CAP_ANY)]
    if os.name == 'nt':
        automatic = backends['dshow'] + backends['msmf'] + automatic
    attempts = backends.get(camera_backend, automatic) if is_camera else [('영상 입력', cv2.CAP_ANY)]
    failures = []
    for name, api in attempts:
        cap = None
        success = False
        if is_camera:
            print(f'카메라 {source} 연결 시도: {name}', flush=True)
        try:
            cap = cv2.VideoCapture(source, api) if is_camera else cv2.VideoCapture(source)
            if not cap.isOpened():
                failures.append(f'{name}: 장치를 열지 못했습니다.')
                continue
            # 일부 카메라는 초기화 직후 빈 프레임을 반환하므로 잠시 기다립니다.
            deadline = time.monotonic() + 2.0
            while True:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    success = True
                    if is_camera:
                        print(f'카메라 {source} 연결 완료: {name}', flush=True)
                    return cap, frame
                if not is_camera or time.monotonic() >= deadline:
                    failures.append(f'{name}: 화면을 읽지 못했습니다.')
                    break
                time.sleep(.05)
        except cv2.error:
            failures.append(f'{name}: OpenCV 연결 오류가 발생했습니다.')
        finally:
            if cap is not None and not success:
                cap.release()
    if is_camera:
        details = '\n'.join(failures)
        raise RuntimeError(
            f'카메라 {source}에서 화면을 읽을 수 없습니다.\n{details}\n'
            '1. Windows 카메라 앱과 화상회의 앱을 종료하세요.\n'
            '2. Windows 설정 > 개인 정보 및 보안 > 카메라에서 '
            '데스크톱 앱의 카메라 접근 허용 여부를 확인하세요.\n'
            '3. 일반 카메라와 적외선 카메라가 함께 있으면 --source 1도 확인하세요.'
        )
    raise RuntimeError(f'카메라 또는 영상을 열거나 첫 프레임을 읽을 수 없습니다: {source}')
