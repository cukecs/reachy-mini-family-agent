from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\ReachyAI\app")))

from reachy_ai.speech import stable_doa_direction


class SpeechDoATests(unittest.TestCase):
    def test_stable_front_axis_samples_are_accepted(self):
        result = stable_doa_direction(
            [
                np.deg2rad(87.0),
                np.deg2rad(90.0),
                np.deg2rad(92.0),
                np.deg2rad(89.0),
            ]
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(np.degrees(result), 89.5, places=1)

    def test_wildly_scattered_samples_are_rejected(self):
        result = stable_doa_direction(
            [
                np.deg2rad(5.0),
                np.deg2rad(90.0),
                np.deg2rad(175.0),
            ]
        )
        self.assertIsNone(result)

    def test_out_of_documented_range_is_ignored(self):
        result = stable_doa_direction(
            [
                np.deg2rad(350.0),
                np.deg2rad(89.0),
                np.deg2rad(91.0),
            ]
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(np.degrees(result), 90.0, places=1)

    def test_single_sample_is_not_enough_to_move_robot(self):
        self.assertIsNone(stable_doa_direction([np.deg2rad(30.0)]))


if __name__ == "__main__":
    unittest.main()
