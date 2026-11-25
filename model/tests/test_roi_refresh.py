"""매 실행 자동 탐색, 동일 파일 갱신 및 자동/수동 재지정을 검증합니다."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scripts import demo_camera


class FakeLive:
    def __init__(self):
        self.calls = 0
        self.release = Mock()

    def latest(self, after_sequence=0):
        self.calls += 1
        if self.calls > 20:
            raise AssertionError('UI 이벤트 처리 없는 반복문')
        return SimpleNamespace(sequence=self.calls,
            image=np.full((480, 640, 3), self.calls, dtype=np.uint8),
            captured_at=time.monotonic())


class RoiRefreshTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.roi_file = self.base / 'config/fixed_roi.json'

    def seed(self, roi):
        self.roi_file.parent.mkdir(parents=True, exist_ok=True)
        self.roi_file.write_text(json.dumps({'roi': roi}), encoding='utf-8')

    def saved(self):
        return json.loads(self.roi_file.read_text(encoding='utf-8'))['roi']

    def run_demo(self, autos, keys, manual=None, source='0', relative=False):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = Mock()
        cap.get.return_value = 30
        cap.read.return_value = (True, image)
        model = Mock()
        model.names = {0: 'person'}
        model.track.return_value = [SimpleNamespace(boxes=None)]
        live = FakeLive()
        roi_arg = 'config/fixed_roi.json' if relative else str(self.roi_file)
        args = ['demo_camera.py', '--source', source, '--roi-file', roi_arg]
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv', args))
            stack.enter_context(patch.object(demo_camera, 'ROOT', self.base))
            load_model = stack.enter_context(patch.object(demo_camera, 'YOLO', return_value=model))
            opened = stack.enter_context(patch.object(demo_camera, 'open_video_capture', return_value=(cap, image)))
            stack.enter_context(patch.object(demo_camera, 'resolve_video_path', return_value=self.base/'video.mp4'))
            stack.enter_context(patch.object(demo_camera, 'LatestFrameCapture', return_value=live))
            auto = stack.enter_context(patch.object(demo_camera, 'auto_detect_crosswalk_roi', side_effect=autos))
            manual_call = stack.enter_context(patch.object(demo_camera, 'select_roi', return_value=manual))
            panel = stack.enter_context(patch.object(demo_camera, 'draw_panel', return_value=image.copy()))
            stack.enter_context(patch.object(cv2, 'namedWindow'))
            stack.enter_context(patch.object(cv2, 'imshow'))
            poll = stack.enter_context(patch.object(cv2, 'waitKey', side_effect=keys))
            stack.enter_context(patch.object(cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(cv2, 'destroyAllWindows'))
            demo_camera.main()
        load_model.assert_called_once()
        opened.assert_called_once()
        self.assertEqual(poll.call_count, len(keys))
        if source.isdecimal():
            live.release.assert_called_once()
            cap.read.assert_not_called()
        else:
            cap.release.assert_called_once()
        return auto, manual_call, model, panel

    def test_each_launch_overwrites_one_existing_file(self):
        self.seed([.01, .01, .1, .1])
        first, second = [.1, .2, .5, .4], [.2, .3, .6, .5]
        for roi in (first, second):
            auto, manual, _, _ = self.run_demo([roi], [ord('q')], relative=True)
            auto.assert_called_once()
            manual.assert_not_called()
            self.assertEqual(self.saved(), roi)
            self.assertEqual(list(self.roi_file.parent.iterdir()), [self.roi_file])

    def test_a_uses_latest_frame_without_reloading_model(self):
        first, second = [.1, .2, .5, .4], [.2, .3, .6, .5]
        auto, manual, model, _ = self.run_demo([first, second], [ord('a'), -1, ord('q')])
        self.assertEqual(auto.call_count, 2)
        self.assertEqual(int(auto.call_args_list[0].args[0][0,0,0]), 0)
        self.assertEqual(int(auto.call_args_list[1].args[0][0,0,0]), 2)
        self.assertEqual(model.track.call_count, 3)
        manual.assert_not_called()
        self.assertEqual(self.saved(), second)
        self.assertEqual(len(list(self.roi_file.parent.iterdir())), 1)

    def test_failed_retry_preserves_active_roi_and_file(self):
        roi = [.1, .2, .5, .4]
        _, _, _, panel = self.run_demo([roi, None], [ord('a'), ord('q')])
        self.assertEqual(self.saved(), roi)
        self.assertTrue(panel.call_args.kwargs['roi_defined'])

    def test_startup_failure_does_not_reuse_stale_roi_or_open_manual_dialog(self):
        old = [.1, .1, .2, .2]
        for source in ('0', 'video.mp4'):
            self.seed(old)
            auto, manual, _, panel = self.run_demo([None], [ord('q')], source=source)
            auto.assert_called_once()
            manual.assert_not_called()
            self.assertFalse(panel.call_args.kwargs['roi_defined'])
            self.assertEqual(self.saved(), old)

    def test_z_keeps_manual_selection_and_same_file(self):
        roi = [.2, .3, .4, .5]
        auto, manual, _, _ = self.run_demo([None], [ord('z'), ord('q')], manual=roi)
        auto.assert_called_once()
        manual.assert_called_once()
        self.assertEqual(self.saved(), roi)
        self.assertEqual(len(list(self.roi_file.parent.iterdir())), 1)

    def test_failed_auto_after_manual_selection_preserves_manual_roi(self):
        roi = [.2, .3, .4, .5]
        auto, manual, _, panel = self.run_demo(
            [None, None], [ord('z'), ord('a'), ord('q')], manual=roi)
        self.assertEqual(auto.call_count, 2)
        manual.assert_called_once()
        self.assertEqual(self.saved(), roi)
        self.assertTrue(panel.call_args.kwargs['roi_defined'])

    def test_cancel_manual_selection_preserves_auto_roi(self):
        roi = [.1, .2, .5, .4]
        _, manual, _, _ = self.run_demo([roi], [ord('z'), ord('q')], manual=None)
        manual.assert_called_once()
        self.assertEqual(self.saved(), roi)


if __name__ == '__main__':
    unittest.main()
