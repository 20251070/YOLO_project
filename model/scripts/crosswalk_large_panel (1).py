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
    from .video_source import resolve_video_path, open_video_capture, LatestFrameCapture
else:
    from video_source import resolve_video_path, open_video_capture, LatestFrameCapture

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
def auto_detect_crosswalk_roi(frame):
    #프레임 하단부에서 흰색 횡단보도 줄무늬(얼룩말 패턴) 영역을 자동으로 찾아 정규화 좌표를 반환
    fh, fw = frame.shape[:2]
    # 하늘이나 상단 배경 오탐지를 막기 위해 화면의 아래쪽 65% 영역만 탐색
    y_offset = int(fh * 0.35)
    roi_search = frame[y_offset:, :]

    # 1. 밝은 흰색 영역 추출 -> HSV 명도 V가 높고 채도 S가 낮은 영역
    hsv = cv2.cvtColor(roi_search, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2. 노이즈 제거 및 떨어진 흰색 줄무늬들을 하나의 횡단보도 덩어리로 묶기(Morphology)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
    
    # 가로 방향으로 넓게 팽창시켜 여러 개의 횡단보도 막대를 하나의 영역으로 연결
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, fw // 15), 15))
    merged = cv2.dilate(mask, kernel_merge, iterations=2)

    # 3. 가장 큰 덩어리(횡단보도 후보) 찾기
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    max_area = (fw * fh) * 0.03  # 전체 화면의 최소 3% 이상 크기만 인정

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # 가로가 세로보다 긴 형태(횡단보도 특성) 우선 선택
        if area > max_area and w > h * 1.2:
            max_area = area
            best_rect = (x, y + y_offset, w, h)

    if best_rect is None:
        return None

    x, y, w, h = best_rect
    return [
        max(0.0, min(1.0, x / fw)),
        max(0.0, min(1.0, y / fh)),
        max(0.0, min(1.0, w / fw)),
        max(0.0, min(1.0, h / fh)),
    ]

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


@lru_cache(maxsize=64)
def korean_bold_font(size):
    candidates = (
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/malgunbd.ttf',
        Path('/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return korean_font(size)


def text_fit(img, msg, x, y, font_scale, color, thick=2, bold=False):
    # OpenCV 기본 글꼴은 한글을 지원하지 않아 Pillow로 필요한 영역만 그립니다.
    size = max(9, round(32 * font_scale))
    stroke = max(0, thick - 1)
    while True:
        font = korean_bold_font(size) if bold else korean_font(size)
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


def draw_footer_title(img, message, footer_top, base):
    """하단 배너의 중앙에 굵은 한글 제목을 선명하게 그립니다."""
    height, width = img.shape[:2]
    footer_height = height - footer_top
    size = min(max(18, round(40 * base)), max(14, round(footer_height * .60)))
    stroke = max(2, round(2.2 * base))

    while True:
        font = korean_bold_font(size)
        bbox = font.getbbox(message, stroke_width=stroke)
        text_width = bbox[2] - bbox[0]
        if text_width <= width - 32 or size <= 14:
            break
        size -= 2

    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2 - bbox[0]
    y = (footer_height - text_height) // 2 - bbox[1]

    crop = Image.fromarray(cv2.cvtColor(img[footer_top:height], cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(crop)
    draw.text((x, y), message, font=font, fill=(25, 25, 25),
              stroke_width=stroke, stroke_fill=(25, 25, 25))
    img[footer_top:height] = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)


def draw_crosswalk_warning_box(img, top, base):
    """횡단보도 안에서 보행자가 감지되면 빨간 굵은 글씨를 테두리 박스에 표시합니다."""
    height, width = img.shape[:2]
    margin = max(12, round(width * 0.035))
    box_left = margin
    box_right = width - margin
    box_height = min(round(height * 0.24), max(96, round(154 * base)))
    box_top = max(0, min(top, height - box_height - 12))
    box_bottom = box_top + box_height

    red = (0, 0, 255)
    cv2.rectangle(img, (box_left, box_top), (box_right, box_bottom), red, thickness=max(2, round(3 * base)))

    lines = ('횡단보도 내 보행자 확인', '서행 혹은 정지하세요')
    max_text_width = box_right - box_left - max(24, round(28 * base))
    size = max(14, round(37 * base))
    stroke = max(1, round(1.2 * base))
    while size > 14:
        font = korean_bold_font(size)
        widths = [font.getbbox(line, stroke_width=stroke)[2] - font.getbbox(line, stroke_width=stroke)[0]
                  for line in lines]
        if max(widths) <= max_text_width:
            break
        size -= 2

    font = korean_bold_font(size)
    line_gap = max(5, round(8 * base))
    line_height = max(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines)
    total_height = line_height * 2 + line_gap
    center_x = width // 2
    first_y = box_top + (box_height - total_height) // 2 + line_height // 2

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for index, line in enumerate(lines):
        y = first_y + index * (line_height + line_gap)
        draw.text((center_x, y), line, font=font, fill=(255, 0, 0), anchor='mm',
                  stroke_width=stroke, stroke_fill=(255, 0, 0))
    img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return box_bottom


def draw_panel(height, width, signal, crossing, nearby, alert_hold, red_hold, last_fps,
               live=False, frame_age_ms=0.0, roi_defined=True):
    signal_label = {'GREEN': '녹색', 'RED': '적색', 'UNKNOWN': '알 수 없음'}[signal]
    panel = np.full((height, width, 3), (30, 37, 44), dtype=np.uint8)
    crosswalk_warning = bool(crossing or alert_hold)

    if crosswalk_warning:
        label, detail, color = '경고', '', (45, 65, 225)
    elif red_hold:
        label, detail, color = '경고', '적색 신호 중 새 진입 감지(미확인)', (45, 65, 225)
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
    text_fit(panel, '카메라 실시간 분석' if live else '우회전 보조 / 시연',
             16, max(25, bar_top-18), .75*base, (245,245,245))
    text_fit(panel, label, 23, bar_top+max(34,round((bar_bottom-bar_top)*.7)),
             1.4*base, (255,255,255), 2)

    content_top = min(height-80, bar_bottom+44)
    if crosswalk_warning:
        box_bottom = draw_crosswalk_warning_box(panel, content_top, base)
        info_y = box_bottom + max(18, round(22 * base))
    else:
        text_fit(panel, detail, 16, content_top, .7*base, (245,245,245))
        info_y = content_top + 35

    counts = (f'횡단 중: {crossing}명 | 주변 감지: {nearby}명' if roi_defined
              else f'전체 화면 보행자: {nearby}명 | z=횡단보도 지정')
    text_fit(panel, counts, 16, info_y, .62*base, (235,235,235))
    text_fit(panel, f'보행자 신호(수동): {signal_label}', 16, info_y+35,
             .6*base, (155,220,255))
    performance = f'분석: {last_fps:.1f} FPS'
    if live:
        performance += f' | 수신 후 {frame_age_ms:.0f} ms'
    text_fit(panel, performance, 16, info_y+70, .56*base, (180,210,220))

    footer_height = min(height, max(72, round(height * .15)))
    footer_top = height - footer_height
    disclaimer_y = min(max(info_y+110, height-105), max(20, footer_top-34))
    text_fit(panel, '운전자가 직접 횡단보도 상황을 확인해야 합니다.', 16,
             disclaimer_y, .55*base, (210,210,210), 1)
    text_fit(panel, 'g=녹색 r=적색 u=모름 z=영역 p=정지 q=종료', 16,
             max(18, footer_top-9), .45*base, (190,190,190), 1)

    # 시스템 이름은 패널의 가장 아래쪽 전용 배너 중앙에 굵은 고딕체로 표시합니다.
    cv2.rectangle(panel, (0, footer_top), (width-1, height-1), (0, 215, 255), -1)
    draw_footer_title(panel, '스마트 보행 안전 시스템', footer_top, base)
    return panel


def main():
    p = argparse.ArgumentParser(description='카메라 실시간 보행자 감지 및 추적')
    p.add_argument('--source', default='0', help='웹캠 0, USB 카메라 1 또는 RTSP/HTTP 주소, 영상 파일 경로')
    p.add_argument('--model', default=str(ROOT / 'yolo11n.pt'))
    p.add_argument('--conf', type=float, default=.3)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='', help='비우면 자동 선택. CPU는 cpu, NVIDIA GPU는 0')
    p.add_argument('--roi-file', default=None, help='횡단보도 영역 파일. 카메라와 영상은 기본 파일을 따로 사용합니다.')
    p.add_argument('--output', default='', help='분석 화면을 MP4로 저장할 경로')
    p.add_argument('--max-width', type=int, default=1280)
    p.add_argument('--max-height', type=int, default=800)
    p.add_argument('--live-timeout', type=float, default=5.0, help='새 카메라 프레임 수신 제한 시간(초)')
    p.add_argument('--camera-backend', choices=('auto', 'dshow', 'msmf'), default='auto')
    p.add_argument('--check-camera', action='store_true', help='영상 입력 연결만 확인한 뒤 종료')
    a = p.parse_args()
    if a.max_width < 1 or a.max_height < 1 or a.imgsz < 1:
        p.error('화면 크기와 분석 이미지 크기는 0보다 커야 합니다.')
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
        if a.roi_file:
            roi_path = Path(a.roi_file)
        elif isinstance(source, int):
            roi_path = ROOT / f'config/camera_{source}_roi.json'
        elif is_live:
            roi_path = ROOT / 'config/ip_roi.json'
        else:
            roi_path = ROOT / 'config/crosswalk_roi.json'
        try:
            roi = json.loads(roi_path.read_text(encoding='utf-8'))['roi'] if roi_path.exists() else None
            if not (isinstance(roi, list) and len(roi) == 4
                    and all(np.isfinite(float(v)) and 0 <= float(v) <= 1 for v in roi)):
                roi = None
            elif float(roi[2]) <= 0 or float(roi[3]) <= 0:
                roi = None
            else:
                roi = list(map(float, roi))
        except (ValueError, KeyError, TypeError):
            roi = None
        if roi is None:
            print('횡단보도 영역을 자동으로 탐색합니다(색상 기반 추정).', flush=True)
            roi = auto_detect_crosswalk_roi(frame)
            if roi is not None:
                print('횡단보도 영역 추정 완료. z 키로 수정할 수 있습니다.', flush=True)
                roi_path.parent.mkdir(parents=True, exist_ok=True)
                roi_path.write_text(json.dumps({'roi': roi}, indent=2), encoding='utf-8')
        if roi is None and not is_live:
            roi = select_roi(frame)
            if roi is None:
                raise RuntimeError('횡단보도 관심 영역을 지정해 주세요.')
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_path.write_text(json.dumps({'roi': roi}, indent=2), encoding='utf-8')
        output_size = (frame.shape[1]*2, frame.shape[0])
        if a.output:
            out = Path(a.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), source_fps, output_size)
            if not writer.isOpened():
                raise RuntimeError(f'영상 파일을 저장할 수 없습니다: {out}')
        if is_live:
            print('실시간 분석 시작: q=종료, p=일시 정지, z=횡단보도 지정', flush=True)
            if roi is None:
                print('전체 화면의 보행자를 감지합니다. 횡단 여부를 분석하려면 z 키로 영역을 지정하세요.', flush=True)
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
                        frame, captured_at = packet.image, packet.captured_at
                elif first_file_frame:
                    first_file_frame = False
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None or not frame.size:
                        break
                if ready:
                    height, width = frame.shape[:2]
                    if writer and (width*2, height) != output_size:
                        raise RuntimeError('입력 해상도가 변경되어 녹화를 종료합니다. 다시 실행하세요.')
                    if roi is not None:
                        x1r, y1r, x2r, y2r = pixel_roi(roi, width, height)
                    result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                         classes=people, imgsz=a.imgsz, conf=a.conf,
                                         verbose=False, **device_options)[0]
                    view = frame.copy()
                    if roi is not None:
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
                            inside = roi is not None and x1r <= foot[0] <= x2r and y1r <= foot[1] <= y2r
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
            key = cv2.waitKey(10 if paused else 1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('p'):
                paused = not paused
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
            elif key == ord('z'):
                # 영역 선택 때만 의도적으로 한 프레임을 고정합니다.
                latest = live_capture.latest() if live_capture else None
                new_roi = select_roi(latest.image if latest is not None else frame)
                if new_roi is not None:
                    roi = new_roi
                    roi_path.parent.mkdir(parents=True, exist_ok=True)
                    roi_path.write_text(json.dumps({'roi': roi},indent=2),encoding='utf-8')
                    previous_inside.clear()
                    crossing_until = red_until = 0
                    paused = False
                    last_analysis_at = None
                    last_fps = 0.0
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
