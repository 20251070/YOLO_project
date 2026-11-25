"""영상 시간 동기화, 건너뛰기, 일시 정지 및 EOF 처리를 확인합니다."""
import unittest
import numpy as np
from scripts.video_source import VideoPlayback
from scripts.demo_camera import prepare_frame


class FakeCapture:
    def __init__(self, count=100):
        self.index=1
        self.count=count
        self.reads=0
        self.grabs=0

    def grab(self):
        self.grabs+=1
        if self.index>=self.count:return False
        self.index+=1
        return True

    def read(self):
        self.reads+=1
        if self.index>=self.count:return False,None
        frame=np.full((2,2,3),self.index,dtype=np.uint8)
        self.index+=1
        return True,frame


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.now=0.
        self.cap=FakeCapture()
        self.playback=VideoPlayback(self.cap,10,clock=lambda:self.now)
        self.playback.start()

    def test_wait_returns_control_without_reading_a_duplicate_frame(self):
        self.now=.05
        self.assertEqual(self.playback.read(),(True,None))
        self.assertEqual(self.cap.reads,0)

    def test_late_processing_skips_to_media_time(self):
        self.now=.31
        ok,frame=self.playback.read()
        self.assertTrue(ok)
        self.assertEqual(int(frame[0,0,0]),3)
        self.assertEqual(self.cap.grabs,2)

    def test_pause_does_not_fast_forward_after_resume(self):
        self.now=.1
        self.playback.pause()
        self.now=10.1
        self.assertEqual(self.playback.read(),(True,None))
        self.playback.resume()
        self.now=10.21
        ok,frame=self.playback.read()
        self.assertTrue(ok)
        self.assertEqual(int(frame[0,0,0]),2)

    def test_large_backlog_yields_for_ui_events(self):
        self.now=5.
        self.assertEqual(self.playback.read(),(True,None))
        self.assertEqual(self.cap.grabs,self.playback.max_grabs)
        self.assertEqual(self.cap.reads,0)

    def test_eof_while_skipping_stops_playback(self):
        self.playback=VideoPlayback(FakeCapture(count=3),10,clock=lambda:self.now)
        self.playback.start()
        self.now=2.
        self.assertEqual(self.playback.read(),(False,None))

    def test_processing_size_is_limited_and_can_be_disabled(self):
        frame=np.zeros((2160,3840,3),dtype=np.uint8)
        self.assertEqual(prepare_frame(frame,1280).shape,(720,1280,3))
        self.assertIs(prepare_frame(frame,0),frame)


if __name__=='__main__':
    unittest.main()
