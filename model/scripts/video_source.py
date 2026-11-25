"""실행 위치와 관계없이 프로젝트의 영상 파일 경로를 찾습니다."""
from pathlib import Path
import os
import time
import threading
from dataclasses import dataclass

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
    if is_camera:
        attempts = backends.get(camera_backend, automatic)
    elif '://' not in str(source) and hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
        attempts = [('하드웨어 영상 디코딩', cv2.CAP_FFMPEG), ('영상 입력', cv2.CAP_ANY)]
    else:
        attempts = [('영상 입력', cv2.CAP_ANY)]
    failures = []
    for name, api in attempts:
        cap = None
        success = False
        if is_camera:
            print(f'카메라 {source} 연결 시도: {name}', flush=True)
        try:
            if not is_camera and api == cv2.CAP_FFMPEG:
                cap = cv2.VideoCapture(source, api, [
                    cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,
                ])
            else:
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
                    elif api == cv2.CAP_FFMPEG and cap.get(cv2.CAP_PROP_HW_ACCELERATION):
                        print('영상 하드웨어 디코딩 사용', flush=True)
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


@dataclass(frozen=True)
class LiveFrame:
    sequence: int
    image: object
    captured_at: float


class LatestFrameCapture:
    """계속 수신한 영상 중 가장 최근 프레임 하나만 보관합니다.

    캡처 스레드만 read/release를 호출합니다. 시간은 센서 시간이 아닌
    OpenCV 수신 완료 시점입니다.
    """

    def __init__(self, capture, first_frame, timeout=5.0):
        if timeout <= 0:
            raise ValueError('실시간 입력 제한 시간은 0보다 커야 합니다.')
        self._capture = capture
        self._timeout = timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = LiveFrame(1, first_frame, time.monotonic())
        self._error = None
        self._thread = threading.Thread(target=self._run, name='camera-capture', daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                ok, image = self._capture.read()
                now = time.monotonic()
                if ok and image is not None and image.size:
                    with self._lock:
                        self._frame = LiveFrame(self._frame.sequence + 1, image, now)
                else:
                    with self._lock:
                        expired = now - self._frame.captured_at > self._timeout
                    if expired:
                        raise RuntimeError('카메라 영상 수신이 끊겼습니다. 연결 상태를 확인하고 다시 실행하세요.')
                    self._stop.wait(.02)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            self._capture.release()

    def latest(self, after_sequence=0):
        """새 프레임만 반환하고 입력 중단 시 오래된 결과를 재사용하지 않습니다."""
        with self._lock:
            if self._error:
                raise RuntimeError(f'실시간 영상 입력 오류: {self._error}')
            if time.monotonic() - self._frame.captured_at > self._timeout:
                raise RuntimeError('카메라의 새 영상이 도착하지 않습니다. 연결 상태를 확인하세요.')
            if self._frame.sequence <= after_sequence:
                return None
            return self._frame

    def release(self):
        self._stop.set()
        # read가 멈춰도 UI 종료를 오래 막지 않습니다. 실제 release는
        # read가 반환된 뒤 캡처 스레드에서 수행합니다.
        self._thread.join(timeout=1.0)


class VideoPlayback:
    """미디어 시간에 맞춰 프레임을 고르고 대기 중에는 UI로 제어를 돌려줍니다.

    open_video_capture에서 첫 프레임을 이미 읽었으므로 다음 인덱스는 1입니다.
    첫 화면을 표시한 뒤 start()를 호출해 모델 준비 시간을 재생 시간에서 뺍니다.
    """

    def __init__(self, capture, fps, clock=None):
        self.capture = capture
        self.fps = float(fps)
        if not 0 < self.fps < float('inf'):
            raise ValueError('영상 FPS는 0보다 큰 유한한 값이어야 합니다.')
        self.clock = clock or time.monotonic
        self.started_at = None
        self.paused_at = None
        self.next_index = 1
        self.max_grabs = max(1, round(self.fps / 3))

    def start(self):
        if self.started_at is None:
            self.started_at = self.clock()

    def pause(self):
        if self.started_at is not None and self.paused_at is None:
            self.paused_at = self.clock()

    def resume(self):
        if self.paused_at is not None:
            self.started_at += self.clock() - self.paused_at
            self.paused_at = None

    def read(self):
        """(정상, 프레임)을 반환합니다. 프레임 None은 이벤트 처리 후 재호출할 대기 상태입니다."""
        self.start()
        if self.paused_at is not None:
            return True, None
        target = int(max(0., self.clock() - self.started_at) * self.fps)
        if target < self.next_index:
            return True, None
        skip = min(target - self.next_index, self.max_grabs)
        for _ in range(skip):
            if not self.capture.grab():
                return False, None
            self.next_index += 1
        if self.next_index < target:
            return True, None
        ok, frame = self.capture.read()
        if not ok or frame is None or not frame.size:
            return False, None
        self.next_index += 1
        return True, frame
