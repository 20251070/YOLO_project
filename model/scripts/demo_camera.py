"""저장 영상, USB 웹캠, RTSP/IP 스트림을 이용한 우회전 보행자 경고 시연입니다.
연구·시연 전용입니다. 보행자 신호는 수동 입력하며 법적·교통 상황을 자동 판단하지 않습니다.
단축키: a 자동 재탐색, z 수동 다각형 지정, g 녹색, r 적색, u 알 수 없음, p 일시 정지, q 종료.
관심 영역 좌표는 정규화하여 카메라·시점별로 --roi-file에 저장합니다.
"""
import argparse
import json
import time
import os
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

if __package__:
    from .video_source import resolve_video_path, open_video_capture, LatestFrameCapture, VideoPlayback
    from .crosswalk_roi import detect_crosswalk_roi
    from .roi_polygon import select_polygon, normalized_points, pixel_polygon, contains_point
else:
    from video_source import resolve_video_path, open_video_capture, LatestFrameCapture, VideoPlayback
    from crosswalk_roi import detect_crosswalk_roi
    from roi_polygon import select_polygon, normalized_points, pixel_polygon, contains_point

ROOT = Path(__file__).resolve().parents[1]
TITLE = '우회전 보조 - 카메라 시연'


def fit(img, max_w=1280, max_h=800):
    h, w = img.shape[:2]
    k = min(1.0, max_w/w, max_h/h)
    if k >= 1:
        return img, 1.0
    return cv2.resize(img, (round(w*k), round(h*k)), interpolation=cv2.INTER_AREA), k


def select_roi(frame):
    image, _ = fit(frame, 900, 700)
    print('경계를 따라 점을 찍고 Enter로 확정하세요. 오른쪽 클릭=점 취소, C/Esc=취소.', flush=True)
    return select_polygon(image, text_fit)


def auto_detect_crosswalk_roi(frame):
    return detect_crosswalk_roi(frame)


def prepare_frame(frame, max_size):
    """분석·그리기 크기를 제한합니다. 0이면 원본 해상도를 유지합니다."""
    return fit(frame, max_size, max_size)[0] if max_size else frame


def save_roi(roi_path, roi):
    """사용자가 지정한 동일 파일에 영역을 갱신합니다."""
    normalized_points(roi)  # 기존 사각형과 새 다각형 모두 검사합니다.
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text(json.dumps({'roi': roi}, indent=2), encoding='utf-8')


def refresh_auto_roi(frame, roi_path):
    """현재 프레임에서 한 번 재탐색하고 성공한 경우 같은 파일에 저장합니다."""
    print('횡단보도 자동 탐색 중(색상 기반 추정)...', flush=True)
    roi = auto_detect_crosswalk_roi(frame)
    if roi is None:
        print('자동 탐색 실패. a=자동 재탐색, z=수동 영역 지정', flush=True)
        return None
    save_roi(roi_path, roi)
    print(f'횡단보도 자동 탐색 완료. 동일 파일 갱신: {roi_path}', flush=True)
    return roi


def pixel_roi(norm, width, height):
    """기존 호출을 위한 경계 사각형입니다. 실제 내부 판정은 다각형으로 수행합니다."""
    polygon = pixel_polygon(norm, width, height)
    left, top = np.floor(polygon.min(axis=0)).astype(int)
    right, bottom = np.ceil(polygon.max(axis=0)).astype(int)
    return int(left), int(top), int(right), int(bottom)


@lru_cache(maxsize=64)
def korean_font(size):
    """운영체제에 설치된 한글 글꼴을 크기별로 불러와 재사용합니다."""
    candidates = (
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/malgun.ttf',
        Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf'),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise SystemExit('한글 글꼴이 없습니다. 맑은 고딕, 나눔고딕 또는 Noto Sans CJK를 설치하세요.')


def text_fit(img, msg, x, y, font_scale, color, thick=2):
    # OpenCV 기본 글꼴은 한글을 지원하지 않아 Pillow로 필요한 영역만 그립니다.
    size = max(9, round(32 * font_scale))
    stroke = max(0, thick - 1)
    while True:
        font = korean_font(size)
        left, top, right, bottom = font.getbbox(msg, anchor='ls', stroke_width=stroke)
        if right - left <= img.shape[1] - x - 12 or size <= 9:
            break
        size = max(9, round(size * .86))
    x1, y1 = max(0, x + left), max(0, y + top)
    x2, y2 = min(img.shape[1], x + right), min(img.shape[0], y + bottom)
    if x1 >= x2 or y1 >= y2:
        return
    crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(crop).text((x - x1, y - y1), msg, font=font, anchor='ls',
                             fill=tuple(color[::-1]), stroke_width=stroke)
    img[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)


def draw_panel(height, width, signal, crossing, nearby, alert_hold, red_hold, last_fps,
               live=False, frame_age_ms=0.0, roi_defined=True):
    signal_label = {'GREEN': '녹색', 'RED': '적색', 'UNKNOWN': '알 수 없음'}[signal]
    panel = np.full((height, width, 3), (30, 37, 44), dtype=np.uint8)
    if red_hold:
        label, detail, color = '경고', '적색 신호 중 새 진입 감지(미확인)', (45, 65, 225)
    elif crossing or alert_hold:
        label, detail, color = '경고', '횡단보도 안에 보행자가 있습니다', (45, 65, 225)
    elif nearby:
        label, detail, color = '주의', '주변에서 보행자가 감지되었습니다', (0, 175, 255)
    else:
        label, detail, color = '확인 필요', '감지된 보행자가 없습니다', (125, 145, 150)
    if not roi_defined and not nearby:
        detail = '전체 화면에서 보행자를 분석 중입니다'
    base = min(width/620, height/720, 1.15)
    bar_top = max(60, round(height*.10))
    bar_bottom = min(height-1, bar_top+max(48, round(height*.11)))
    cv2.rectangle(panel, (12, bar_top), (width-12, bar_bottom), color, -1)
    text_fit(panel, '카메라 실시간 분석' if live else '우회전 보조 / 시연', 16, max(25,bar_top-18), .75*base, (245,245,245))
    text_fit(panel, label, 23, bar_top+max(34,round((bar_bottom-bar_top)*.7)), 1.4*base, (255,255,255), 2)
    y = min(height-80, bar_bottom+44)
    text_fit(panel, detail, 16, y, .7*base, (245,245,245))
    counts = (f'횡단 중: {crossing}명 | 주변 감지: {nearby}명' if roi_defined
              else f'전체 화면 보행자: {nearby}명 | a=자동 z=수동')
    text_fit(panel, counts, 16, y+35, .62*base, (235,235,235))
    text_fit(panel, f'보행자 신호(수동): {signal_label}', 16, y+70, .6*base, (155,220,255))
    performance = f'분석: {last_fps:.1f} FPS'
    if live:
        performance += f' | 수신 후 {frame_age_ms:.0f} ms'
    text_fit(panel, performance, 16, y+105, .56*base, (180,210,220))
    text_fit(panel, '운전자가 직접 횡단보도 상황을 확인해야 합니다.', 16, max(y+145, height-56), .55*base, (210,210,210), 1)
    text_fit(panel, 'g=녹색 r=적색 u=모름 a=자동 z=수동 p=정지 q=종료', 16, height-18, .45*base, (190,190,190), 1)
    return panel


def main():
    p = argparse.ArgumentParser(description='카메라 실시간 보행자 감지 및 추적')
    p.add_argument('--source', default='0', help='웹캠 0, USB 카메라 1 또는 RTSP/HTTP 주소, 영상 파일 경로')
    p.add_argument('--model', default=str(ROOT / 'yolo11n.pt'))
    p.add_argument('--conf', type=float, default=.3)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--max-frame-size', type=int, default=1280,
                   help='분석·그리기에 사용할 최대 변 길이. 0이면 원본 크기')
    p.add_argument('--video-mode', choices=('auto', 'realtime', 'all'), default='auto',
                   help='기본은 원본 시간에 맞춰 프레임 선택. --output 저장 시에는 전체 프레임 처리')
    p.add_argument('--device', default='', help='비우면 자동 선택. CPU는 cpu, NVIDIA GPU는 0')
    p.add_argument('--roi-file', default=None, help='영역을 갱신할 동일 파일. 상대 경로는 model 폴더 기준입니다.')
    p.add_argument('--output', default='', help='분석 화면을 MP4로 저장할 경로')
    p.add_argument('--max-width', type=int, default=1280)
    p.add_argument('--max-height', type=int, default=800)
    p.add_argument('--live-timeout', type=float, default=5.0, help='새 카메라 프레임 수신 제한 시간(초)')
    p.add_argument('--camera-backend', choices=('auto', 'dshow', 'msmf'), default='auto')
    p.add_argument('--check-camera', action='store_true', help='영상 입력 연결만 확인한 뒤 종료')
    a = p.parse_args()
    if a.max_width < 1 or a.max_height < 1 or a.imgsz < 1:
        p.error('화면 크기와 분석 이미지 크기는 0보다 커야 합니다.')
    if a.max_frame_size < 0:
        p.error('--max-frame-size는 0 이상이어야 합니다.')
    if not np.isfinite(a.live_timeout) or a.live_timeout <= 0:
        p.error('--live-timeout은 0보다 큰 유한한 값이어야 합니다.')
    if a.source.isdecimal():
        source = int(a.source)
    elif '://' in a.source:
        source = a.source
    else:
        try:
            source = str(resolve_video_path(a.source))
        except FileNotFoundError as exc:
            p.error(str(exc))
    is_live = isinstance(source, int) or '://' in str(source)
    # 모델 준비를 끝낸 뒤 카메라를 엽니다.
    if not a.check_camera:
        model_path = Path(a.model).expanduser()
        if not model_path.is_absolute() and not model_path.is_file() and (ROOT / model_path).is_file():
            model_path = ROOT / model_path
        model = YOLO(str(model_path))
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        people = [int(i) for i, v in names.items() if str(v).strip().lower() in ('person', 'pedestrian', 'people', 'human', 'humans')]
        if not people:
            p.error(f'모델에 사람 클래스가 없습니다: {names}')
    try:
        cap, frame = open_video_capture(source, a.camera_backend)
    except RuntimeError as exc:
        p.error(str(exc))

    live_capture = None
    playback = None
    writer = None
    window_created = False
    try:
        if a.check_camera:
            height, width = frame.shape[:2]
            print(f'영상 입력 확인 완료: {width} x {height}')
            return
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 20.0
        if is_live:
            live_capture = LatestFrameCapture(cap, frame, timeout=a.live_timeout)
        frame = prepare_frame(frame, a.max_frame_size)
        realtime_file = not is_live and (a.video_mode == 'realtime'
                                        or (a.video_mode == 'auto' and not a.output))
        if realtime_file:
            playback = VideoPlayback(cap, source_fps)
            print(f'영상 실시간 재생: 원본 {source_fps:.2f} FPS, 처리 지연 시 중간 프레임을 건너뜁니다.', flush=True)
        if a.roi_file:
            roi_path = Path(a.roi_file)
        elif isinstance(source, int):
            roi_path = ROOT / f'config/camera_{source}_roi.json'
        elif is_live:
            roi_path = ROOT / 'config/ip_roi.json'
        else:
            roi_path = ROOT / 'config/crosswalk_roi.json'
        roi_path = roi_path.expanduser()
        if not roi_path.is_absolute():
            roi_path = ROOT / roi_path
        roi_path = roi_path.resolve()
        # 저장 파일의 존재와 무관하게 실행할 때마다 현재 화면에서 새로 탐색합니다.
        # 실패하면 전체 화면의 보행자 감지를 계속하며 a/z로 다시 지정할 수 있습니다.
        roi = refresh_auto_roi(frame, roi_path)
        output_size = (frame.shape[1]*2, frame.shape[0])
        if a.output:
            out = Path(a.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), source_fps, output_size)
            if not writer.isOpened():
                raise RuntimeError(f'영상 파일을 저장할 수 없습니다: {out}')
        if is_live:
            print('실시간 분석 시작: q=종료, p=일시 정지, a=자동 재탐색, z=수동 지정', flush=True)
            if roi is None:
                print('전체 화면의 보행자를 감지합니다. a=자동 재탐색, z=수동 영역 지정', flush=True)
        signal = 'UNKNOWN'
        previous_inside = {}
        crossing_until = red_until = 0.0
        paused = False
        last_fps = 0.0
        last_analysis_at = None
        last_sequence = 0
        preview = None
        first_file_frame = True
        device_options = {'device': a.device} if a.device else {}
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        window_created = True
        while True:
            # 일시 정지 중에도 입력 중단을 확인하고 캡처는 계속 진행합니다.
            packet = live_capture.latest(last_sequence) if live_capture else None
            if not paused:
                ready = True
                captured_at = None
                if live_capture:
                    ready = packet is not None
                    if ready:
                        last_sequence = packet.sequence
                        frame = prepare_frame(packet.image, a.max_frame_size)
                        captured_at = packet.captured_at
                elif first_file_frame:
                    first_file_frame = False
                else:
                    ok, next_frame = playback.read() if playback else cap.read()
                    if not ok:
                        break
                    ready = next_frame is not None and bool(next_frame.size)
                    if ready:
                        frame = prepare_frame(next_frame, a.max_frame_size)
                    elif playback is None:
                        break
                if ready:
                    height, width = frame.shape[:2]
                    if writer and (width*2, height) != output_size:
                        raise RuntimeError('입력 해상도가 변경되어 녹화를 종료합니다. 다시 실행하세요.')
                    if roi is not None:
                        roi_shape = pixel_polygon(roi, width, height)
                        roi_left, roi_top = np.rint(roi_shape.min(axis=0)).astype(int)
                    result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                         classes=people, imgsz=a.imgsz, conf=a.conf,
                                         verbose=False, **device_options)[0]
                    view = frame.copy()
                    if roi is not None:
                        cv2.polylines(view, [np.rint(roi_shape).astype(np.int32)], True, (255,185,0), 2, cv2.LINE_AA)
                        text_fit(view, '횡단보도 영역', int(roi_left), max(20,int(roi_top)-8), .6, (255,185,0))
                    boxes = result.boxes
                    n_crossing = n_nearby = 0
                    next_inside = {}
                    entered_red = False
                    if boxes is not None:
                        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(boxes)
                        for box, ident in zip(boxes, ids):
                            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                            foot = (max(0,min(width-1,(x1+x2)//2)), max(0,min(height-1,y2)))
                            inside = roi is not None and contains_point(roi_shape, foot)
                            n_crossing += inside
                            n_nearby += not inside
                            if ident is not None:
                                if inside and previous_inside.get(ident) is False and signal == 'RED':
                                    entered_red = True
                                next_inside[ident] = inside
                            color = (30,70,240) if inside else (0,200,240)
                            cv2.rectangle(view,(x1,y1),(x2,y2),color,2)
                            cv2.circle(view,foot,4,color,-1)
                            text_fit(view, f'{"영역 안" if inside else "보행자"} #{ident if ident is not None else "?"}',
                                     max(2,x1),max(20,y1-7),.5,color)
                    previous_inside = next_inside
                    now = time.monotonic()
                    if n_crossing:
                        crossing_until = now + 1.0
                    if entered_red:
                        red_until = now + 1.0
                    if last_analysis_at is not None:
                        fps = 1 / max(.0001, now - last_analysis_at)
                        last_fps = .8*last_fps+.2*fps if last_fps else fps
                    last_analysis_at = now
                    frame_age_ms = (now - captured_at)*1000 if captured_at is not None else 0.0
                    panel = draw_panel(height,width,signal,n_crossing,n_nearby,
                                       now<crossing_until,now<red_until,last_fps,
                                       live=is_live, frame_age_ms=frame_age_ms, roi_defined=roi is not None)
                    display = np.hstack((view,panel))
                    if writer:
                        writer.write(display)
                    preview, _ = fit(display,a.max_width,a.max_height)
                    cv2.imshow(TITLE, preview)
                    if playback:
                        playback.start()
            key = cv2.waitKey(10 if paused else 1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('p'):
                paused = not paused
                if playback:
                    playback.pause() if paused else playback.resume()
                last_analysis_at = None
                last_fps = 0.0
                if paused and preview is not None:
                    paused_preview = preview.copy()
                    text_fit(paused_preview, '일시 정지 | p=실시간 분석 재개', 16, 35, .8, (0,220,255))
                    cv2.imshow(TITLE, paused_preview)
            elif key == ord('g'):
                signal = 'GREEN'
            elif key == ord('r'):
                signal = 'RED'
            elif key == ord('u'):
                signal = 'UNKNOWN'
            elif key == ord('a'):
                # 모델이나 입력을 재시작하지 않고 최신 화면의 영역만 다시 찾습니다.
                latest = live_capture.latest() if live_capture else None
                auto_frame = prepare_frame(latest.image, a.max_frame_size) if latest is not None else frame
                new_roi = refresh_auto_roi(auto_frame, roi_path)
                if new_roi is not None:
                    roi = new_roi
                    previous_inside.clear()
                    crossing_until = red_until = 0
                    paused = False
                    if playback:
                        playback.resume()
                    last_analysis_at = None
                    last_fps = 0.0
                elif roi is not None:
                    print('현재 사용 중인 횡단보도 영역을 유지합니다.', flush=True)
            elif key == ord('z'):
                # 영역 선택 때만 의도적으로 한 프레임을 고정합니다.
                if playback:
                    playback.pause()
                latest = live_capture.latest() if live_capture else None
                manual_frame = prepare_frame(latest.image, a.max_frame_size) if latest is not None else frame
                new_roi = select_roi(manual_frame)
                if new_roi is not None:
                    roi = new_roi
                    save_roi(roi_path, roi)
                    previous_inside.clear()
                    crossing_until = red_until = 0
                    paused = False
                    last_analysis_at = None
                    last_fps = 0.0
                if playback and not paused:
                    playback.resume()
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        p.exit(1, f'{exc}\n')
    finally:
        if live_capture:
            live_capture.release()
        else:
            cap.release()
        if writer:
            writer.release()
        if window_created:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
