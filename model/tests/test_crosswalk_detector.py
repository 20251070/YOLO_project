"""어두운 도색과 기울어진 줄무늬 및 반복 패턴이 없는 배경을 검사합니다."""
import unittest
import cv2
import numpy as np
from scripts.crosswalk_roi import detect_crosswalk_roi


class CrosswalkTests(unittest.TestCase):
    def stripes(self):
        image=np.full((480,720,3),80,dtype=np.uint8)
        for x in range(150,560,48):
            cv2.rectangle(image,(x,40),(x+16,170),(150,150,150),-1)
        return image

    def test_gray_crosswalk_in_upper_image_is_detected(self):
        roi=detect_crosswalk_roi(self.stripes())
        self.assertIsNotNone(roi)
        self.assertLess(roi[1],.35)
        self.assertGreater(roi[2],.4)

    def test_rotated_crosswalk_is_detected(self):
        matrix=cv2.getRotationMatrix2D((360,240),20,1.)
        frame=cv2.warpAffine(self.stripes(),matrix,(720,480),borderValue=(80,80,80))
        self.assertIsNotNone(detect_crosswalk_roi(frame))

    def test_uniform_bright_or_dark_background_is_not_crosswalk(self):
        for value in (20,150,255):
            self.assertIsNone(detect_crosswalk_roi(np.full((480,720,3),value,dtype=np.uint8)))

    def test_single_white_rectangle_is_not_repeated_stripes(self):
        image=np.full((480,720,3),80,dtype=np.uint8)
        cv2.rectangle(image,(200,100),(240,300),(220,220,220),-1)
        self.assertIsNone(detect_crosswalk_roi(image))


if __name__=='__main__':
    unittest.main()
