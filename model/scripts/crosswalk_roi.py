"""명암 대비와 반복되는 평행 도색 띠로 횡단보도 후보를 추정합니다."""
import math

import cv2
import numpy as np



def detect_crosswalk_roi(frame):
    """반복되는 평행 도색 띠를 명암과 회전 사각형으로 묶어 영역을 추정합니다."""
    height, width = frame.shape[:2]
    scale = min(1.0, 960.0 / max(width, height))
    small = cv2.resize(frame, (round(width*scale), round(height*scale))) if scale < 1 else frame
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    size = max(15, (round(max(sw,sh)*.055) // 2)*2+1)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((size,size),np.uint8))
    level, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(contrast, max(12,level), 255, cv2.THRESH_BINARY)
    mask[hsv[:, :, 1] > 120] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripes = []
    edge = max(sw, sh)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(10, sw*sh*.000025) or area > sw*sh*.12:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side < 2 or long_side < edge*.025 or long_side > edge*.45 or not 1.8 <= long_side/short_side <= 25:
            continue
        if area / max(1, rw*rh) < .5:
            continue
        box = cv2.boxPoints(rect)
        edges = np.roll(box, -1, axis=0) - box
        vector = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        direction = vector / max(1e-6, np.linalg.norm(vector))
        angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
        stripes.append({'center': np.array(rect[0]), 'long': long_side, 'short': short_side,
                        'angle': angle, 'direction': direction, 'area': area, 'contour': contour})
    stripes = sorted(stripes, key=lambda stripe: stripe['area'], reverse=True)[:200]
    parents = list(range(len(stripes)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for i, one in enumerate(stripes):
        for j in range(i):
            two = stripes[j]
            angle_delta = abs(one['angle']-two['angle'])
            angle_delta = min(angle_delta, math.pi-angle_delta)
            if angle_delta > math.radians(20):
                continue
            if min(one['long'], two['long'])/max(one['long'], two['long']) < .45:
                continue
            delta = one['center']-two['center']
            longest = max(one['long'], two['long'])
            if np.linalg.norm(delta) > longest*.75:
                continue
            if abs(float(np.dot(delta, one['direction']))) > longest*.3:
                continue
            parents[find(i)] = find(j)
    groups = {}
    for index, stripe in enumerate(stripes):
        groups.setdefault(find(index), []).append(stripe)
    best, best_score = None, 0
    for group in groups.values():
        if len(group) < 4:
            continue
        points = np.concatenate([s['contour'] for s in group])
        x, y, w, h = cv2.boundingRect(points)
        if w*h < sw*sh*.008:
            continue
        coverage = sum(s['area'] for s in group)/(w*h)
        if not .08 <= coverage <= .85:
            continue
        score = sum(s['area'] for s in group)*math.sqrt(len(group))
        if score > best_score:
            padding = max(2, round(np.median([s['short'] for s in group])*.5))
            left, top = max(0,x-padding), max(0,y-padding)
            right, bottom = min(sw,x+w+padding), min(sh,y+h+padding)
            best = [left/sw, top/sh, (right-left)/sw, (bottom-top)/sh]
            best_score = score
    return best
