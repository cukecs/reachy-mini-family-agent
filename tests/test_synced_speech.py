from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reachy_ai.runtime_state import RuntimeState
from reachy_ai.speech import (
    SpeechEngine,
    build_progressive_slices,
    split_speech_segments,
)


class _FakeMedia:
    def __init__(self, sample_rate: int = 16000, channels: int = 2) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self.events: list[str] = []
        self.pushed_frames = 0

    def get_output_audio_samplerate(self) -> int:
        return self._sample_rate

    def get_output_channels(self) -> int:
        return self._channels

    def start_playing(self) -> None:
        self.events.append("start_playing")

    def push_audio_sample(self, block: np.ndarray) -> None:
        self.events.append("push")
        self.pushed_frames += len(block)

    def stop_playing(self) -> None:
        self.events.append("stop_playing")


class SyncedSpeechTests(unittest.TestCase):
    def test_runtime_state_begin_and_update_event_avoids_empty_bubble(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = RuntimeState(path=path, max_messages=10)
            event_id = state.begin_event("assistant", route="deepseek")

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["messages"], [])

            state.update_event(event_id, "第一句。")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["messages"]), 1)
            self.assertEqual(saved["messages"][0]["text"], "第一句。")
            self.assertEqual(saved["messages"][0]["route"], "deepseek")

            state.update_event(event_id, "第一句。第二句。")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["messages"]), 1)
            self.assertEqual(saved["messages"][0]["text"], "第一句。第二句。")

    def test_split_segments_and_progressive_slices(self) -> None:
        text = "你好，今天天气不错！我们一起学英语吧。Think about it carefully, and speak slowly."
        segments = split_speech_segments(text)
        self.assertGreaterEqual(len(segments), 3)
        slices = build_progressive_slices(text, segments)
        self.assertEqual(len(slices), len(segments))
        self.assertEqual(slices[-1], text)
        for earlier, later in zip(slices, slices[1:]):
            self.assertTrue(later.startswith(earlier))

    def test_speak_emits_progress_only_after_audio_starts(self) -> None:
        engine = SpeechEngine()
        media = _FakeMedia(sample_rate=16000, channels=2)
        observed: list[tuple[str, bool]] = []

        def fake_synth(segment: str, speed: float = 1.0):
            return np.zeros(1600, dtype=np.float32), 16000

        def on_progress(partial: str) -> None:
            observed.append((partial, "start_playing" in media.events))

        with patch.object(engine, "synthesize", side_effect=fake_synth):
            engine.speak(
                media,
                "第一句话。第二句话。",
                display_text="第一句话。第二句话。",
                on_text_progress=on_progress,
            )

        self.assertTrue(observed)
        self.assertTrue(all(started for _, started in observed))
        self.assertEqual(observed[-1][0], "第一句话。第二句话。")
        self.assertIn("stop_playing", media.events)

    def test_speak_interrupt_before_playback_emits_no_text(self) -> None:
        engine = SpeechEngine()
        media = _FakeMedia()
        observed: list[str] = []

        with patch.object(engine, "_consume_interrupt", side_effect=[False, True]):
            with patch.object(
                engine,
                "synthesize",
                return_value=(np.zeros(1600, dtype=np.float32), 16000),
            ):
                engine.speak(
                    media,
                    "这句话不应该出现。",
                    on_text_progress=observed.append,
                )

        self.assertEqual(observed, [])
        self.assertNotIn("start_playing", media.events)


if __name__ == "__main__":
    unittest.main()
