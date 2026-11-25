
"""사각형/다각형 ROI 호환, 다각형 선택 및 실제 영역 내부 판정을 지원합니다."""
import cv2
import numpy as np


def normalized_points(roi):
    points = np.asarray(roi, dtype=np.float64)
    if points.ndim == 1 and points.size == 4:
        x, y, w, h = points
        if w <= 0 or h <= 0:
            raise ValueError('영역의 너비와 높이는 0보다 커야 합니다.')
        points = np.array([[x, y], [min(1., x+w), y],
                           [min(1., x+w), min(1., y+h)], [x, min(1., y+h)]])
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('경계에 점을 세 개 이상 찍으세요.')
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or len(np.unique(points, axis=0)) != len(points):
        raise ValueError('같은 점이 중복되었습니다. 마지막 점을 취소해 주세요.')
    if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError('좌표는 화면 안에 있어야 합니다.')
    if abs(cv2.contourArea(points.astype(np.float32))) <= 1e-10:
        raise ValueError('영역 넓이가 0입니다. 점의 위치를 바꿔 주세요.')

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def on_segment(a, b, p):
        return (abs(cross(a,b,p)) <= 1e-10
                and min(a[0],b[0])-1e-10 <= p[0] <= max(a[0],b[0])+1e-10
                and min(a[1],b[1])-1e-10 <= p[1] <= max(a[1],b[1])+1e-10)

    def intersects(a, b, c, d):
        ab_c, ab_d, cd_a, cd_b = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
        return ((ab_c*ab_d < 0 and cd_a*cd_b < 0)
                or on_segment(a,b,c) or on_segment(a,b,d)
                or on_segment(c,d,a) or on_segment(c,d,b))

    count = len(points)
    for i in range(count):
        for j in range(i+1, count):
            if j == i+1 or (i == 0 and j == count-1):
                continue
            if intersects(points[i], points[(i+1)%count], points[j], points[(j+1)%count]):
                raise ValueError('경계선이 교차합니다. 점을 경계 순서대로 찍어 주세요.')
    return points


def pixel_polygon(roi, width, height):
    points = normalized_points(roi)
    return (points * [max(0,width-1), max(0,height-1)]).astype(np.float32)


def contains_point(polygon, point):
    return cv2.pointPolygonTest(polygon, (float(point[0]),float(point[1])), False) >= 0


def select_polygon(image, draw_text):
    """경계 순서대로 점을 찍어 오목한 다각형도 지정합니다."""
    title = 'Crosswalk polygon'
    height, width = image.shape[:2]
    points = []
    hover = [None]
    message = ['경계를 따라 점을 세 개 이상 찍으세요.']

    def mouse(event, x, y, flags, param):
        inside = 0 <= x < width and 0 <= y < height
        hover[0] = (x,y) if inside else None
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            points.append((x,y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            preview = image.copy()
            if len(points) >= 3:
                overlay = preview.copy()
                cv2.fillPoly(overlay, [np.array(points,np.int32)], (255,185,0))
                preview = cv2.addWeighted(overlay,.22,preview,.78,0)
            if len(points) >= 2:
                cv2.polylines(preview,[np.array(points,np.int32)],len(points)>=3,(255,185,0),2,cv2.LINE_AA)
            if points and hover[0] is not None:
                cv2.line(preview,points[-1],hover[0],(180,220,255),1,cv2.LINE_AA)
            for index,point in enumerate(points,1):
                cv2.circle(preview,point,4,(0,255,255),-1)
                cv2.putText(preview,str(index),(point[0]+5,point[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
            canvas = cv2.copyMakeBorder(preview,0,108,0,0,cv2.BORDER_CONSTANT,value=(30,37,44))
            for row,text in enumerate((
                    '왼쪽 클릭: 점 추가 | Enter/Space: 확정',
                    '오른쪽 클릭/Backspace: 마지막 점 취소',
                    'X: 다시 그리기 | C/Esc: 취소',
                    message[0])):
                draw_text(canvas,text,10,height+22+row*23,.48,(240,240,240),1)
            cv2.imshow(title,canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27,ord('c'),ord('C'),ord('q')):
                return None
            try:
                if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE) < 1:
                    return None
            except cv2.error:
                return None
            if key in (8,127):
                if points: points.pop()
            elif key in (ord('x'),ord('X')):
                points.clear()
            elif key in (10,13,32):
                try:
                    normalized = np.asarray(points,dtype=np.float64) / [max(1,width-1),max(1,height-1)]
                    return normalized_points(normalized).tolist()
                except ValueError as exc:
                    message[0] = str(exc)
    finally:
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass
