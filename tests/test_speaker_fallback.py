from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from reachy_ai.identity import IdentityStabilizer, Match, resolve_speaker_identity
from reachy_ai.memory import MemoryStore
from reachy_ai.robot_agent import RobotBrain


class SpeakerFallbackTests(unittest.TestCase):
    def setUp(self):
        self.stabilizer = IdentityStabilizer()
        self.no_voice = Match(None, 0.0, "voice")
        self.face = Match("owner", 0.85, "face")

    def resolve(self, voice=None, faces=None):
        return resolve_speaker_identity(
            voice if voice is not None else self.no_voice,
            faces if faces is not None else [self.face],
            stabilizer=self.stabilizer,
        )

    def test_unmatched_voice_uses_current_face(self):
        result = self.resolve()
        self.assertEqual(result.user_id, "owner")
        self.assertEqual(result.reason, "face-fallback")
        self.assertEqual(result.confidence, 0.85)
        self.assertFalse(result.private_memory_allowed)

    def test_weak_voice_with_other_candidate_does_not_block_face(self):
        result = self.resolve(Match(None, 0.43, "voice", "guest"))
        self.assertEqual(result.user_id, "owner")
        self.assertEqual(result.reason, "face-fallback")

    def test_pending_same_voice_can_use_face_then_confirm_voice(self):
        voice = Match("owner", 0.57, "voice")
        self.assertEqual(self.resolve(voice).reason, "face-fallback")
        self.assertEqual(self.resolve(voice).reason, "voice-confirmed")

    def test_pending_voice_for_other_person_blocks_face_swap(self):
        result = self.resolve(Match("guest", 0.57, "voice"))
        self.assertIsNone(result.user_id)
        self.assertEqual(result.reason, "face/voice conflict")

    def test_confirmed_voice_wins_over_other_visible_person(self):
        result = self.resolve(Match("guest", 0.81, "voice"))
        self.assertEqual(result.user_id, "guest")
        self.assertEqual(result.reason, "voice-confirmed")

    def test_confirmed_voice_works_without_camera(self):
        result = self.resolve(Match("owner", 0.81, "voice"), [])
        self.assertEqual(result.user_id, "owner")

    def test_multiple_known_faces_do_not_choose_centered_person(self):
        result = self.resolve(faces=[self.face, Match("guest", 0.92, "face")])
        self.assertIsNone(result.user_id)
        self.assertEqual(result.reason, "face-multiple")

    def test_one_known_face_and_one_unknown_is_still_multiple(self):
        result = self.resolve(faces=[self.face, Match(None, 0.1, "face")])
        self.assertIsNone(result.user_id)
        self.assertEqual(result.reason, "face-multiple")

    def test_unmatched_face_candidate_is_not_promoted(self):
        result = self.resolve(faces=[Match(None, 0.86, "face", "owner")])
        self.assertIsNone(result.user_id)
        self.assertEqual(result.reason, "face-unmatched")

    def test_low_quality_or_invalid_face_not_promoted(self):
        for score in [0.67, float("nan"), float("inf")]:
            with self.subTest(score=score):
                self.assertIsNone(self.resolve(faces=[Match("owner", score, "face")]).user_id)

    def test_fallback_threshold_is_inclusive(self):
        self.assertEqual(self.resolve(faces=[Match("owner", 0.68, "face")]).user_id, "owner")

    def test_fallback_never_carries_previous_person_to_missing_frame(self):
        self.assertEqual(self.resolve().user_id, "owner")
        result = self.resolve(faces=[])
        self.assertIsNone(result.user_id)
        self.assertEqual(result.reason, "unknown")

    def test_fallback_can_change_to_current_face_without_stale_identity(self):
        self.assertEqual(self.resolve().user_id, "owner")
        self.assertEqual(self.resolve(faces=[Match("guest", 0.84, "face")]).user_id, "guest")

    def test_fallback_identity_reaches_model_and_family_memory(self):
        store = MemoryStore(":memory:", official_memory_root=None)
        self.addCleanup(store.close)
        store.ensure_user("owner", "Test Person")
        store.upsert_fact("owner", "favorite_drink", "green tea")
        decision = self.resolve()
        brain = RobotBrain(memory=store)
        brain.set_identity(
            decision.user_id, private_memory_allowed=True,
            source=decision.reason, confidence=decision.confidence,
        )
        with patch.object(brain, "_route", return_value={"route": "local"}), \
             patch.object(brain, "_chat", return_value=("我在听。", {})) as chat, \
             patch("reachy_ai.robot_agent.maybe_use_network_tool", return_value=None):
            brain.answer("跟我聊两句", visible_people=["Test Person"])
        messages = chat.call_args.args[0]
        system = messages[0]["content"]
        self.assertIn("当前说话者：Test Person", system)
        self.assertIn("回退到当前唯一可靠匹配的人脸特征", system)
        self.assertIn("green tea", system)
        self.assertNotIn("仅由声纹模块给出", system)
        self.assertIn("[当前说话者：Test Person]", messages[-1]["content"])

    def test_strong_fallback_allows_explicit_profile_update(self):
        store = MemoryStore(":memory:", official_memory_root=None)
        self.addCleanup(store.close)
        store.ensure_user("owner", "Test Person")
        brain = RobotBrain(memory=store)
        brain.set_identity("owner", private_memory_allowed=True, source="face-fallback", confidence=0.85)
        brain._remember_turn("以后叫我小李", "记住了。", [("preferred_name", "小李")])
        self.assertEqual(store.user_name("owner"), "小李")

    def test_fallback_history_preserves_strong_match_and_hides_weak_one(self):
        history = RobotBrain._model_history([
            {"role": "user", "content": "第一句", "speaker_name": "Test Person",
             "speaker_source": "face-fallback", "speaker_confidence": "0.85"},
            {"role": "user", "content": "第二句", "speaker_name": "Test Person",
             "speaker_source": "face-fallback", "speaker_confidence": "0.70"},
        ])
        self.assertIn("[说话者：Test Person]", history[0]["content"])
        self.assertEqual(history[1]["content"], "第二句")


if __name__ == "__main__":
    unittest.main()
