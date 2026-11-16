from __future__ import annotations

import unittest

from reachy_ai.conversation_gate import ConversationGate, ConversationGateConfig


class ConversationGateTests(unittest.TestCase):
    def test_youtube_or_ambient_speech_ignored_while_standby(self) -> None:
        gate = ConversationGate(
            ConversationGateConfig(
                enabled=True,
                mode="wake_word",
                wake_words=("天线宝宝", "Reachy"),
                active_window_seconds=45.0,
            )
        )
        res = gate.evaluate("欢迎来到我的频道，今天我们来看一个新视频", now=100.0)
        self.assertFalse(res.accepted)
        self.assertEqual(res.reason, "standby_ignored")

    def test_wake_word_opens_window_and_strips_prefix(self) -> None:
        gate = ConversationGate(
            ConversationGateConfig(
                enabled=True,
                mode="wake_word",
                wake_words=("天线宝宝", "Reachy"),
                active_window_seconds=45.0,
            )
        )
        wake_only = gate.evaluate("天线宝宝", now=100.0)
        self.assertTrue(wake_only.accepted)
        self.assertTrue(wake_only.wake_only)
        self.assertTrue(gate.is_awake(120.0))

        follow_up = gate.evaluate("今天上海天气怎么样？", now=120.0)
        self.assertTrue(follow_up.accepted)
        self.assertEqual(follow_up.reason, "active_window")
        self.assertEqual(follow_up.cleaned_text, "今天上海天气怎么样？")

        # Expired after 45s without speech
        expired = gate.evaluate("YouTube video talking again", now=170.0)
        self.assertFalse(expired.accepted)

        # Wake with inline command and fuzzy ASR match
        inline = gate.evaluate("添线抱抱，帮我查一下时间", now=200.0)
        self.assertTrue(inline.accepted)
        self.assertEqual(inline.cleaned_text, "帮我查一下时间")

    def test_sleep_command_returns_to_standby(self) -> None:
        gate = ConversationGate(
            ConversationGateConfig(
                enabled=True,
                mode="wake_word",
                wake_words=("天线宝宝",),
                active_window_seconds=45.0,
            )
        )
        gate.evaluate("天线宝宝", now=10.0)
        self.assertTrue(gate.is_awake(20.0))

        slept = gate.evaluate("我先看视频了", now=25.0)
        self.assertTrue(slept.slept_now)
        self.assertFalse(gate.is_awake(26.0))

        after = gate.evaluate("视频里的旁白", now=30.0)
        self.assertFalse(after.accepted)


if __name__ == "__main__":
    unittest.main()
