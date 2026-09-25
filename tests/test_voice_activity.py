from __future__ import annotations

import unittest
import wave
from pathlib import Path

import numpy as np

from reachy_ai.speech import SpeechEngine


class VoiceActivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SpeechEngine()

    def test_mouse_click_like_impulse_is_rejected(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        audio[4000:4020] = 0.95
        result = self.engine.analyze_voice_activity(audio, 16000)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "non_speech_or_impulse")

    def test_keyboard_like_impulses_are_rejected(self) -> None:
        audio = np.zeros(32000, dtype=np.float32)
        for start in (2500, 6100, 9800, 14100, 18800, 24000):
            audio[start : start + 35] = np.linspace(0.7, 0.0, 35, dtype=np.float32)
        result = self.engine.analyze_voice_activity(audio, 16000)
        self.assertFalse(result.accepted)

    def test_real_speech_is_accepted(self) -> None:
        source = Path(
            r"D:\ReachyAI\models\speech\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17\test_wavs\zh.wav"
        )
        with wave.open(str(source), "rb") as wav:
            sample_rate = wav.getframerate()
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
        result = self.engine.analyze_voice_activity(audio / 32768.0, sample_rate)
        self.assertTrue(result.accepted)
        self.assertGreater(result.longest_speech_seconds, 0.30)


if __name__ == "__main__":
    unittest.main()
