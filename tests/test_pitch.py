from __future__ import annotations

import unittest

import numpy as np

from reachy_ai.pitch import analyze_pitch


class PitchAnalysisTests(unittest.TestCase):
    @staticmethod
    def _melody(frequencies: list[float], sample_rate: int = 16000) -> np.ndarray:
        pieces = [np.zeros(int(sample_rate * 0.18), dtype=np.float32)]
        for frequency in frequencies:
            time = np.arange(int(sample_rate * 0.55), dtype=np.float32) / sample_rate
            tone = 0.25 * np.sin(2.0 * np.pi * frequency * time)
            pieces.extend([tone.astype(np.float32), np.zeros(int(sample_rate * 0.10), dtype=np.float32)])
        return np.concatenate(pieces)

    def test_three_note_melody(self) -> None:
        analysis = analyze_pitch(self._melody([261.626, 329.628, 391.995]), 16000)
        self.assertTrue(analysis.reliable)
        names = [note.name for note in analysis.notes]
        self.assertEqual(names[:3], ["C4", "E4", "G4"])
        self.assertEqual(analysis.lowest_note, "C4")
        self.assertEqual(analysis.highest_note, "G4")

    def test_silence_is_not_a_pitch(self) -> None:
        analysis = analyze_pitch(np.zeros(16000, dtype=np.float32), 16000)
        self.assertFalse(analysis.reliable)
        self.assertEqual(analysis.notes, ())


if __name__ == "__main__":
    unittest.main()
