"""하드웨어 영상 디코딩과 미지원 시 기본 디코더 복귀를 검증합니다."""
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from scripts.video_source import open_video_capture


class CaptureBackendTests(unittest.TestCase):
    def capture(self):
        cap=Mock()
        cap.isOpened.return_value=True
        cap.get.return_value=0
        cap.read.return_value=(True,np.zeros((10,10,3),dtype=np.uint8))
        return cap

    def test_file_prefers_available_hardware_decoder(self):
        cap=self.capture()
        with patch.object(cv2,'VideoCapture',return_value=cap) as opened:
            actual,frame=open_video_capture('test.mp4')
        self.assertIs(actual,cap)
        self.assertEqual(opened.call_args.args[1],cv2.CAP_FFMPEG)
        self.assertEqual(opened.call_args.args[2],[cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_ANY])

    def test_unavailable_hardware_falls_back_and_releases_failed_capture(self):
        failed=self.capture()
        failed.isOpened.return_value=False
        fallback=self.capture()
        with patch.object(cv2,'VideoCapture',side_effect=[failed,fallback]) as opened:
            actual,frame=open_video_capture('test.mp4')
        self.assertIs(actual,fallback)
        self.assertEqual(opened.call_count,2)
        failed.release.assert_called_once()
        self.assertEqual(opened.call_args.args,('test.mp4',))

    def test_unsupported_hardware_parameter_falls_back(self):
        fallback=self.capture()
        with patch.object(cv2,'VideoCapture',side_effect=[cv2.error('unsupported'),fallback]):
            actual,frame=open_video_capture('test.mp4')
        self.assertIs(actual,fallback)


if __name__=='__main__':
    unittest.main()
