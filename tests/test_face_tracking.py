from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\ReachyAI\app")))

from reachy_ai.face_tracking import (
    ContinuousFaceTracker,
    find_locked_face,
    select_tracking_target,
    should_turn_to_sound,
    sound_direction_changed,
)


@dataclass
class Observation:
    embedding: np.ndarray
    center: tuple[int, int]
    center_distance: float


class FakeFaceIdentifier:
    def __init__(self, observations):
        self.observations = observations

    def observe_all(self, jpeg):
        if isinstance(self.observations, Exception):
            raise self.observations
        return list(self.observations)


class FakeCamera:
    resolution = (640, 480)


class FakeMedia:
    def __init__(self):
        self.camera = FakeCamera()

    def get_frame_jpeg(self):
        return b"jpeg"


class FakeReachy:
    def __init__(self):
        self.media = FakeMedia()
        self.look_calls = []
        self.goto_calls = []

    def look_at_image(self, u, v, duration=1.0, perform_movement=True):
        self.look_calls.append((u, v, duration, perform_movement))
        return np.eye(4, dtype=np.float64)

    def goto_target(self, head=None, duration=0.5, body_yaw=0.0, **kwargs):
        self.goto_calls.append(
            {"head": head, "duration": duration, "body_yaw": body_yaw}
        )


class FaceTrackingTests(unittest.TestCase):
    def test_previous_face_wins_over_new_center_face(self):
        previous = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        same = Observation(
            np.array([0.98, 0.05, 0.0], dtype=np.float32),
            (520, 240),
            0.30,
        )
        centered_other = Observation(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            (321, 240),
            0.01,
        )
        self.assertIs(
            select_tracking_target([centered_other, same], previous),
            same,
        )

    def test_face_lock_monitor_never_moves_motors(self):
        observation = Observation(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (570, 240),
            0.30,
        )
        reachy = FakeReachy()
        tracker = ContinuousFaceTracker(
            reachy,
            FakeFaceIdentifier([observation]),
        )
        tracker.enable(True)
        tracker.lock_face(observation.embedding, now=9.0)
        self.assertTrue(tracker.track_once(now=10.0))
        self.assertTrue(tracker.locked_face_visible)
        self.assertEqual(reachy.look_calls, [])
        self.assertEqual(reachy.goto_calls, [])

    def test_locked_face_is_not_replaced_by_another_visible_face(self):
        locked = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        other = Observation(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            (320, 240),
            0.0,
        )
        self.assertIsNone(find_locked_face([other], locked))
        reachy = FakeReachy()
        tracker = ContinuousFaceTracker(
            reachy,
            FakeFaceIdentifier([other]),
            lost_timeout_seconds=1.0,
        )
        tracker.enable(True)
        tracker.lock_face(locked, now=8.0)
        self.assertFalse(tracker.track_once(now=10.0))
        self.assertTrue(tracker.locked_face_lost)
        self.assertEqual(reachy.goto_calls, [])

    def test_no_face_marks_lock_lost_without_movement(self):
        reachy = FakeReachy()
        tracker = ContinuousFaceTracker(
            reachy,
            FakeFaceIdentifier(LookupError("no face")),
            lost_timeout_seconds=1.0,
        )
        tracker.enable(True)
        tracker.lock_face(np.array([1.0, 0.0, 0.0], dtype=np.float32), now=8.0)
        self.assertFalse(tracker.track_once(now=10.0))
        self.assertTrue(tracker.locked_face_lost)
        self.assertEqual(reachy.look_calls, [])
        self.assertEqual(reachy.goto_calls, [])

    def test_sound_direction_jitter_is_ignored(self):
        self.assertFalse(sound_direction_changed(0.0, np.deg2rad(10.0)))
        self.assertTrue(sound_direction_changed(0.0, np.deg2rad(30.0)))

    def test_turn_state_machine_reacquires_on_speech_after_face_loss(self):
        # First meaningful source turn is allowed.
        self.assertTrue(
            should_turn_to_sound(
                has_lock=False,
                locked_face_lost=False,
                previous_radians=None,
                current_radians=0.0,
            )
        )
        # Once a face is locked and visible, even a very different source does
        # not make the motors chase it.
        self.assertFalse(
            should_turn_to_sound(
                has_lock=True,
                locked_face_lost=False,
                previous_radians=0.0,
                current_radians=np.deg2rad(90.0),
            )
        )
        # Once the face is gone, the next valid speech must reacquire even
        # when it comes from nearly the same direction.
        self.assertTrue(
            should_turn_to_sound(
                has_lock=True,
                locked_face_lost=True,
                previous_radians=0.0,
                current_radians=np.deg2rad(8.0),
            )
        )
        # A clearly different direction also turns.
        self.assertTrue(
            should_turn_to_sound(
                has_lock=True,
                locked_face_lost=True,
                previous_radians=0.0,
                current_radians=np.deg2rad(45.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
