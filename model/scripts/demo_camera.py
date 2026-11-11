"""저장 영상, USB 웹캠, RTSP/IP 스트림을 이용한 우회전 보행자 경고 시연입니다.
연구·시연 전용입니다. 보행자 신호는 수동 입력하며 법적·교통 상황을 자동 판단하지 않습니다.
단축키: z 횡단보도 영역 지정, g 녹색, r 적색, u 알 수 없음, p 일시 정지, q 종료.
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
    from .video_source import resolve_video_path, open_video_capture
else:
    from video_source import resolve_video_path, open_video_capture

ROOT = Path(__file__).resolve().parents[1]
TITLE = '우회전 보조 - 카메라 시연'


def fit(img, max_w=1280, max_h=800):
    h, w = img.shape[:2]
    k = min(1.0, max_w/w, max_h/h)
    if k >= 1:
        return img, 1.0
    return cv2.resize(img, (round(w*k), round(h*k)), interpolation=cv2.INTER_AREA), k


def select_roi(frame):
    img, ratio = fit(frame, 900, 700)
    print('횡단보도 영역을 마우스로 드래그한 뒤 Enter 키로 확정하세요. C 키로 취소합니다.')
    rect = cv2.selectROI('횡단보도 관심 영역 지정', img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow('횡단보도 관심 영역 지정')
    x, y, w, h = rect
    if not w or not h:
        return None
    fh, fw = frame.shape[:2]
    return [max(0., min(1., x / ratio / fw)),
            max(0., min(1., y / ratio / fh)),
            max(0., min(1., w / ratio / fw)),
            max(0., min(1., h / ratio / fh))]


def pixel_roi(norm, width, height):
    x, y, w, h = norm
    left = max(0, min(width-1, round(x*width)))
    top = max(0, min(height-1, round(y*height)))
    right = max(left+1, min(width, round((x+w)*width)))
    bottom = max(top+1, min(height, round((y+h)*height)))
    return left, top, right, bottom


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


def draw_panel(height, width, signal, crossing, nearby, alert_hold, red_hold, last_fps):
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
    base = min(width/620, height/720, 1.15)
    bar_top = max(60, round(height*.10))
    bar_bottom = min(height-1, bar_top+max(48, round(height*.11)))
    cv2.rectangle(panel, (12, bar_top), (width-12, bar_bottom), color, -1)
    text_fit(panel, '우회전 보조 / 시연', 16, max(25,bar_top-18), .75*base, (245,245,245))
    text_fit(panel, label, 23, bar_top+max(34,round((bar_bottom-bar_top)*.7)), 1.4*base, (255,255,255), 2)
    y = min(height-80, bar_bottom+44)
    text_fit(panel, detail, 16, y, .7*base, (245,245,245))
    text_fit(panel, f'횡단 중: {crossing}명 | 주변 감지: {nearby}명', 16, y+35, .62*base, (235,235,235))
    text_fit(panel, f'보행자 신호(수동): {signal_label}', 16, y+70, .6*base, (155,220,255))
    text_fit(panel, f'미리보기 초당 프레임 수: {last_fps:.1f}', 16, y+105, .56*base, (180,210,220))
    text_fit(panel, '운전자가 직접 횡단보도 상황을 확인해야 합니다.', 16, max(y+145, height-56), .55*base, (210,210,210), 1)
    text_fit(panel, 'g=녹색 r=적색 u=모름 z=영역 p=정지 q=종료', 16, height-18, .45*base, (190,190,190), 1)
    return panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='0', help='웹캠은 0, 그 외에는 파일 경로나 RTSP/HTTP 영상 주소를 지정합니다.')
    p.add_argument('--model', default=str(ROOT / 'yolo11n.pt'), help='추가 학습한 best.pt 또는 기본 yolo11n.pt를 지정합니다.')
    p.add_argument('--conf', type=float, default=.3)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--roi-file', default=str(ROOT / 'config/crosswalk_roi.json'))
    p.add_argument('--output', default='', help='영상 저장 시 사용할 MP4 파일 경로입니다(예: results.mp4).')
    p.add_argument('--max-width', type=int, default=1280)
    p.add_argument('--max-height', type=int, default=800)
    p.add_argument('--camera-backend', choices=('auto', 'dshow', 'msmf'), default='auto',
                   help='카메라 연결 방식입니다. auto는 Windows 연결 방식을 순서대로 시도합니다.')
    p.add_argument('--check-camera', action='store_true',
                   help='영상 입력 연결만 확인하고 모델 실행 없이 종료합니다.')
    a = p.parse_args()
    if a.max_width < 1 or a.max_height < 1:
        p.error('미리보기 너비와 높이는 0보다 커야 합니다.')
    if a.source.isdecimal():
        source = int(a.source)
    elif '://' in a.source:
        source = a.source
    else:
        try:
            source = str(resolve_video_path(a.source))
        except FileNotFoundError as exc:
            p.error(str(exc))
    try:
        cap, frame = open_video_capture(source, a.camera_backend)
    except RuntimeError as exc:
        p.error(str(exc))
    ok = True
    if a.check_camera:
        height, width = frame.shape[:2]
        print(f'영상 입력 확인 완료: {width} x {height}')
        cap.release()
        return
    model_path = Path(a.model).expanduser()
    if not model_path.is_absolute() and not model_path.is_file() and (ROOT / model_path).is_file():
        model_path = ROOT / model_path
    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    people = [int(i) for i, v in names.items() if str(v).lower() in ('person', 'pedestrian', 'people')]
    if not people:
        raise SystemExit(f'모델에 사람 클래스가 없습니다: {names}')
    roi_path = Path(a.roi_file)
    try:
        roi = json.loads(roi_path.read_text(encoding='utf-8'))['roi'] if roi_path.exists() else None
        if not (isinstance(roi, list) and len(roi) == 4 and all(0 <= float(v) <= 1 for v in roi)):
            roi = None
    except (ValueError, KeyError, TypeError):
        roi = None
    if roi is None:
        roi = select_roi(frame)
        if roi is None:
            cap.release()
            raise SystemExit('횡단보도 관심 영역이 필요합니다. 다시 실행하여 횡단보도를 지정하세요.')
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        roi_path.write_text(json.dumps({'roi': roi}, indent=2), encoding='utf-8')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 20.0
    writer = None
    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), source_fps,
                                 (frame.shape[1]*2, frame.shape[0]))
        if not writer.isOpened():
            cap.release()
            raise SystemExit(f'영상 파일을 저장할 수 없습니다: {out}')
    signal = 'UNKNOWN'
    previous_inside = {}
    crossing_until = red_until = 0.0
    paused = False
    last_fps = 0.0
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    try:
        while ok:
            if not paused:
                tick = time.monotonic()
                height, width = frame.shape[:2]
                x1r, y1r, x2r, y2r = pixel_roi(roi, width, height)
                result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                     classes=people, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
                view = frame.copy()
                cv2.rectangle(view, (x1r,y1r), (x2r,y2r), (255,185,0), 2)
                text_fit(view, '횡단보도 영역', x1r, max(20,y1r-8), .6, (255,185,0))
                boxes = result.boxes
                n_crossing = n_nearby = 0
                next_inside = {}
                entered_red = False
                if boxes is not None:
                    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(boxes)
                    for box, ident in zip(boxes, ids):
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        foot = ((x1+x2)//2, y2)
                        inside = x1r <= foot[0] <= x2r and y1r <= foot[1] <= y2r
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
                elapsed = max(.0001, time.monotonic()-tick)
                last_fps = .8*last_fps+.2*(1/elapsed) if last_fps else 1/elapsed
                panel = draw_panel(height,width,signal,n_crossing,n_nearby,
                                   now<crossing_until,now<red_until,last_fps)
                display = np.hstack((view,panel))
                if writer:
                    writer.write(display)
                preview, _ = fit(display,a.max_width,a.max_height)
                cv2.imshow(TITLE, preview)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('g'):
                signal = 'GREEN'
            elif key == ord('r'):
                signal = 'RED'
            elif key == ord('u'):
                signal = 'UNKNOWN'
            elif key == ord('z'):
                new_roi = select_roi(frame)
                if new_roi is not None:
                    roi = new_roi
                    roi_path.parent.mkdir(parents=True, exist_ok=True)
                    roi_path.write_text(json.dumps({'roi': roi},indent=2),encoding='utf-8')
                    previous_inside.clear()
                    crossing_until = red_until = 0
                    paused = False
            if not paused:
                ok, frame = cap.read()
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if a.output:
            print(f'저장 완료: {a.output}')


if __name__ == '__main__':
    main()
