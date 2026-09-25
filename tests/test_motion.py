from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\ReachyAI\app")))

from reachy_ai.motion import (
    look_toward_sound,
    perform_actions,
    requested_actions,
    requested_emotions,
    sound_turn_components,
    strip_action_tags,
    strip_emotion_tags,
)


class FakeReachy:
    def __init__(self):
        self.look_calls = []
        self.goto_calls = []

    def look_at_world(self, x, y, z, duration, perform_movement=True):
        self.look_calls.append((x, y, z, duration, perform_movement))
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        return pose

    def goto_target(
        self, head=None, antennas=None, duration=0.5, method=None, body_yaw=0.0
    ):
        self.goto_calls.append(
            {
                "head": None if head is None else np.array(head, copy=True),
                "antennas": None if antennas is None else list(antennas),
                "duration": duration,
                "body_yaw": body_yaw,
            }
        )


class MotionTests(unittest.TestCase):
    def test_explicit_chinese_actions(self):
        self.assertEqual(requested_actions("请点点头，然后向右看"), ["nod", "look_right"])

    def test_negated_action_is_not_executed(self):
        self.assertEqual(requested_actions("不要摇头"), [])

    def test_model_tag_is_hidden_from_speech(self):
        text, actions = strip_action_tags("当然可以。[[ACTION:nod]]")
        self.assertEqual(text, "当然可以。")
        self.assertEqual(actions, ["nod"])

    def test_unknown_model_action_is_hidden_but_not_executed(self):
        text, actions = strip_action_tags("我正在看着你。[[ACTION:look_at_user]]")
        self.assertEqual(text, "我正在看着你。")
        self.assertEqual(actions, [])

    def test_nod_uses_bounded_fixed_sequence(self):
        reachy = FakeReachy()
        perform_actions(reachy, ["nod"])
        self.assertEqual(len(reachy.look_calls), 3)
        self.assertEqual(reachy.look_calls[-1][:3], (1.0, 0.0, 0.0))
        self.assertTrue(all(call[4] is False for call in reachy.look_calls))
        self.assertTrue(all(call["body_yaw"] is None for call in reachy.goto_calls))

    def test_explicit_body_and_antenna_actions(self):
        self.assertEqual(
            requested_actions("身体向左转一下，再摇摇天线"),
            ["body_left", "antenna_wiggle"],
        )
        self.assertEqual(requested_actions("转动一下你的天线"), ["antenna_wiggle"])
        self.assertEqual(requested_actions("摇一下你的天线"), ["antenna_wiggle"])
        self.assertEqual(requested_actions("那你动一下你的身体"), ["body_wiggle"])
        self.assertEqual(requested_actions("身体大幅度转动一下"), ["body_wiggle_large"])

    def test_body_turns_are_conservative_and_centerable(self):
        reachy = FakeReachy()
        perform_actions(reachy, ["body_left", "body_right", "body_center"])
        yaws = [call["body_yaw"] for call in reachy.goto_calls]
        self.assertGreater(yaws[0], 0.0)
        self.assertLess(yaws[1], 0.0)
        self.assertEqual(yaws[2], 0.0)
        self.assertTrue(all(abs(yaw) <= np.deg2rad(60.0) for yaw in yaws))

    def test_body_wiggle_is_visible_and_returns_to_center(self):
        reachy = FakeReachy()
        perform_actions(reachy, ["body_wiggle", "body_wiggle_large"])
        yaws = [call["body_yaw"] for call in reachy.goto_calls]
        self.assertTrue(any(abs(yaw) >= np.deg2rad(45.0) for yaw in yaws))
        self.assertTrue(any(abs(yaw) >= np.deg2rad(85.0) for yaw in yaws))
        self.assertEqual(yaws[-1], 0.0)

    def test_antenna_wiggle_preserves_body_and_returns_to_neutral(self):
        reachy = FakeReachy()
        perform_actions(reachy, ["antenna_wiggle"])
        self.assertEqual(len(reachy.goto_calls), 3)
        self.assertTrue(all(call["body_yaw"] is None for call in reachy.goto_calls))
        self.assertTrue(
            np.allclose(reachy.goto_calls[-1]["antennas"], [-0.1745, 0.1745])
        )

    def test_official_emotion_request(self):
        self.assertEqual(requested_emotions("做一个开心的表情"), ["laughing2"])
        text, emotions = strip_emotion_tags("好。[[EMOTION:curious1]]")
        self.assertEqual(text, "好。")
        self.assertEqual(emotions, ["curious1"])

    def test_sound_direction_maps_left_and_right(self):
        reachy = FakeReachy()
        look_toward_sound(reachy, 0.0)
        look_toward_sound(reachy, np.pi)
        self.assertGreater(reachy.look_calls[0][1], 0)
        self.assertLess(reachy.look_calls[1][1], 0)
        self.assertGreater(reachy.goto_calls[0]["body_yaw"], 0)
        self.assertIsNone(reachy.goto_calls[1]["body_yaw"])
        self.assertLess(reachy.goto_calls[2]["body_yaw"], 0)
        self.assertIsNone(reachy.goto_calls[3]["body_yaw"])
        self.assertGreater(
            abs(reachy.goto_calls[0]["body_yaw"]), np.deg2rad(50.0)
        )
        self.assertGreater(
            abs(reachy.goto_calls[2]["body_yaw"]), np.deg2rad(50.0)
        )

    def test_sound_direction_body_turn_is_proportional(self):
        local_offset, body_yaw = sound_turn_components(np.pi / 4.0)
        self.assertIsNotNone(body_yaw)
        self.assertGreater(body_yaw, np.deg2rad(25.0))
        self.assertLess(body_yaw, np.deg2rad(35.0))
        self.assertAlmostEqual(local_offset, np.deg2rad(45.0), places=5)

    def test_near_front_sound_keeps_body_still(self):
        local_offset, body_yaw = sound_turn_components(np.pi / 2.0)
        self.assertIsNone(body_yaw)
        self.assertEqual(local_offset, 0.0)

    def test_combined_right_turn_is_staged_body_then_head(self):
        reachy = FakeReachy()
        perform_actions(reachy, ["look_right", "body_right"])
        self.assertEqual(len(reachy.goto_calls), 2)
        self.assertIsNone(reachy.goto_calls[0]["head"])
        self.assertLess(reachy.goto_calls[0]["body_yaw"], 0.0)
        self.assertIsNotNone(reachy.goto_calls[1]["head"])
        self.assertIsNone(reachy.goto_calls[1]["body_yaw"])
        self.assertLess(reachy.look_calls[0][1], 0.0)

    def test_combined_left_turn_is_staged_body_then_head(self):
        reachy = FakeReachy()
        perform_actions(reachy, ["body_left", "look_left"])
        self.assertEqual(len(reachy.goto_calls), 2)
        self.assertIsNone(reachy.goto_calls[0]["head"])
        self.assertGreater(reachy.goto_calls[0]["body_yaw"], 0.0)
        self.assertIsNotNone(reachy.goto_calls[1]["head"])
        self.assertIsNone(reachy.goto_calls[1]["body_yaw"])
        self.assertGreater(reachy.look_calls[0][1], 0.0)


if __name__ == "__main__":
    unittest.main()
