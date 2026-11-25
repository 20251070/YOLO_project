
"""사선·오목한 영역 판정, 기존 사각형 호환 및 선택창 입력을 검증합니다."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from scripts.roi_polygon import normalized_points, pixel_polygon, contains_point, select_polygon


class PolygonGeometryTests(unittest.TestCase):
    def test_diagonal_triangle_does_not_include_bounding_box_corner(self):
        contour=pixel_polygon([[.1,.1],[.9,.1],[.1,.9]],101,101)
        self.assertTrue(contains_point(contour,(20,20)))
        self.assertFalse(contains_point(contour,(80,80)))
        self.assertTrue(contains_point(contour,(10,10)))

    def test_concave_boundary_preserves_notch(self):
        contour=pixel_polygon([[.1,.1],[.9,.1],[.9,.3],[.3,.3],[.3,.9],[.1,.9]],101,101)
        self.assertTrue(contains_point(contour,(20,70)))
        self.assertTrue(contains_point(contour,(70,20)))
        self.assertFalse(contains_point(contour,(70,70)))

    def test_legacy_rectangle_still_works(self):
        contour=pixel_polygon([.1,.2,.5,.4],101,101)
        self.assertTrue(contains_point(contour,(30,40)))
        self.assertFalse(contains_point(contour,(90,90)))

    def test_normalized_polygon_scales_with_resolution(self):
        roi=[[.1,.1],[.9,.1],[.5,.8]]
        for width,height in ((101,101),(1281,721),(3841,2161)):
            contour=pixel_polygon(roi,width,height)
            self.assertTrue(contains_point(contour,((width-1)*.5,(height-1)*.3)))
            self.assertFalse(contains_point(contour,((width-1)*.95,(height-1)*.95)))

    def test_crossing_edges_and_degenerate_regions_are_rejected(self):
        for roi in (
            [[.1,.1],[.9,.9],[.1,.9],[.9,.1]],
            [[.1,.1],[.2,.2],[.3,.3]],
            [[.1,.1],[.1,.1],[.5,.5]],
        ):
            with self.assertRaises(ValueError):
                normalized_points(roi)

    def test_explicit_closing_point_is_allowed(self):
        result=normalized_points([[.1,.1],[.9,.1],[.1,.9],[.1,.1]])
        self.assertEqual(len(result),3)


class PolygonSelectorTests(unittest.TestCase):
    def select(self, actions):
        callback=[]
        def register(title,func):
            callback.append(func)
        def key(delay):
            events,result=next(actions)
            for event,x,y in events:
                callback[0](event,x,y,0,None)
            return result
        image=np.zeros((160,240,3),dtype=np.uint8)
        with patch.object(cv2,'namedWindow'),patch.object(cv2,'setMouseCallback',side_effect=register), \
             patch.object(cv2,'imshow') as shown,patch.object(cv2,'waitKey',side_effect=key), \
             patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'destroyWindow'):
            result=select_polygon(image,lambda *args:None)
        self.assertEqual(shown.call_args.args[1].shape,(268,240,3))
        return result

    def test_click_points_undo_and_confirm(self):
        left,right=cv2.EVENT_LBUTTONDOWN,cv2.EVENT_RBUTTONDOWN
        actions=iter([([(left,20,20),(left,200,20),(left,200,130),(left,50,50),(right,0,0)],13)])
        result=self.select(actions)
        expected=np.array([[20,20],[200,20],[200,130]],dtype=float)/[239,159]
        np.testing.assert_allclose(result,expected)

    def test_escape_keeps_previous_region(self):
        self.assertIsNone(self.select(iter([([],27)])))

    def test_invalid_polygon_stays_open_until_corrected(self):
        left=cv2.EVENT_LBUTTONDOWN
        actions=iter([
            ([(left,20,20),(left,100,100),(left,20,100),(left,100,20)],13),
            ([],ord('x')),
            ([(left,20,20),(left,200,20),(left,100,130)],13),
        ])
        self.assertEqual(len(self.select(actions)),3)


if __name__=='__main__':
    unittest.main()
