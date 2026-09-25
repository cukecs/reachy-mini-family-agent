from __future__ import annotations

import unittest

import cv2
import numpy as np

from reachy_ai.robot_agent import (
    _jpeg_sharpness,
    _prepare_vision_frame,
    _visual_user_content,
    requires_vision,
)


class VisionFrameTests(unittest.TestCase):
    def test_large_camera_frame_is_resized_for_model(self) -> None:
        source = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)

        prepared = _prepare_vision_frame(encoded.tobytes())
        decoded = cv2.imdecode(np.frombuffer(prepared, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (864, 1536))

    def test_sharpness_score_prefers_focused_edges(self) -> None:
        sharp = np.zeros((240, 320, 3), dtype=np.uint8)
        for x in range(0, 320, 16):
            cv2.line(sharp, (x, 0), (x, 239), (255, 255, 255), 3)
        blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
        ok_sharp, sharp_jpeg = cv2.imencode(".jpg", sharp)
        ok_blurred, blurred_jpeg = cv2.imencode(".jpg", blurred)
        self.assertTrue(ok_sharp and ok_blurred)

        self.assertGreater(
            _jpeg_sharpness(sharp_jpeg.tobytes()),
            _jpeg_sharpness(blurred_jpeg.tobytes()),
        )

    def test_temporal_keyframes_are_sent_in_order(self) -> None:
        content = _visual_user_content("你看见我做什么？", [b"one", b"two", b"three"])
        image_items = [item for item in content if item["type"] == "image_url"]
        labels = [item["text"] for item in content if item["type"] == "text"]
        self.assertEqual(len(image_items), 3)
        self.assertIn("按时间顺序", labels[0])
        self.assertEqual(labels[1:], ["关键帧 1/3：", "关键帧 2/3：", "关键帧 3/3："])

    def test_camera_is_only_sent_for_visual_requests(self) -> None:
        self.assertTrue(requires_vision("你看这是什么？"))
        self.assertTrue(requires_vision("猜猜这是哪个主题"))
        self.assertFalse(requires_vision("我到底叫什么名字？"))
        self.assertFalse(requires_vision("刚才和你说话的小女孩叫什么？"))


if __name__ == "__main__":
    unittest.main()
