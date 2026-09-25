from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from reachy_ai.identity import (
    FaceIdentifier,
    FaceObservation,
    IdentityDecision,
    IdentityStabilizer,
    Match,
    _match_templates,
    fuse_identity,
    select_centered_face,
    speaker_identity_from_voice,
)
from reachy_ai.memory import MemoryStore
from reachy_ai.robot_agent import (
    BIOMETRIC_CAPABILITY_QUESTION,
    FACE_ENROLL_REQUEST,
    IDENTITY_LINK_QUESTION,
    MODEL_BACKEND_QUESTION,
    RobotBrain,
    _is_meaningful_utterance,
    _text_similarity,
    conversational_user_id,
    extract_robot_name,
    extract_self_declared_surname,
    extract_self_introduced_name,
    extract_third_person_name,
    strip_memory_tags,
)


class MemoryIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:", official_memory_root=None)
        self.store.ensure_user("owner", "Owner")

    def tearDown(self) -> None:
        self.store.close()

    def test_fact_round_trip(self) -> None:
        self.store.upsert_fact("owner", "english_level", "middle_school")
        self.assertIn("middle_school", self.store.memory_context("owner"))

    def test_biometric_dpapi_round_trip(self) -> None:
        vector = np.arange(16, dtype=np.float32)
        self.store.add_biometric("owner", "face", vector)
        loaded = self.store.biometric_templates("face")[0][1]
        np.testing.assert_array_equal(vector, loaded)

    def test_matching_face_and_voice_unlock_private_memory(self) -> None:
        decision = fuse_identity(
            Match("owner", 0.90, "face"), Match("owner", 0.84, "voice")
        )
        self.assertEqual(decision.user_id, "owner")
        self.assertTrue(decision.private_memory_allowed)

    def test_conflict_locks_private_memory(self) -> None:
        decision = fuse_identity(
            Match("owner", 0.90, "face"), Match("guest", 0.90, "voice")
        )
        self.assertIsNone(decision.user_id)
        self.assertFalse(decision.private_memory_allowed)

    def test_weak_single_modality_does_not_select_a_user(self) -> None:
        decision = fuse_identity(
            Match(None, 0.0, "face"), Match("owner", 0.629, "voice")
        )
        self.assertIsNone(decision.user_id)
        self.assertFalse(decision.private_memory_allowed)

    def test_runtime_speaker_identity_is_voice_only(self) -> None:
        decision = speaker_identity_from_voice(Match("owner", 0.81, "voice"))
        self.assertEqual(decision.user_id, "owner")
        self.assertEqual(decision.reason, "voice-public")

    def test_runtime_weak_voice_is_not_promoted_by_visible_face(self) -> None:
        decision = speaker_identity_from_voice(Match("owner", 0.49, "voice"))
        self.assertIsNone(decision.user_id)
        self.assertEqual(decision.reason, "unknown")

    def test_runtime_moderate_voice_is_usable_without_face_or_direction(self) -> None:
        decision = speaker_identity_from_voice(Match("owner", 0.57, "voice"))
        self.assertEqual(decision.user_id, "owner")
        self.assertEqual(decision.reason, "voice-public")

    def test_moderate_voice_match_sets_public_speaker_identity(self) -> None:
        decision = fuse_identity(
            Match(None, 0.0, "face"), Match("owner", 0.678, "voice")
        )
        self.assertEqual(decision.user_id, "owner")
        self.assertEqual(decision.reason, "voice-public")
        self.assertFalse(decision.private_memory_allowed)

    def test_borderline_voice_requires_two_consecutive_matches(self) -> None:
        stabilizer = IdentityStabilizer()
        raw = IdentityDecision("owner", 0.650, False, "voice-public")
        first = stabilizer.update(raw)
        second = stabilizer.update(raw)
        self.assertIsNone(first.user_id)
        self.assertEqual(first.reason, "voice-pending")
        self.assertEqual(second.user_id, "owner")
        self.assertEqual(second.reason, "voice-confirmed")

    def test_borderline_voice_does_not_carry_across_people(self) -> None:
        stabilizer = IdentityStabilizer()
        owner = IdentityDecision("owner", 0.65, False, "voice-public")
        daughter = IdentityDecision("daughter", 0.65, False, "voice-public")
        self.assertIsNone(stabilizer.update(owner).user_id)
        self.assertIsNone(stabilizer.update(daughter).user_id)

    def test_strong_voice_can_confirm_immediately(self) -> None:
        stabilizer = IdentityStabilizer()
        decision = stabilizer.update(
            IdentityDecision("owner", 0.81, False, "voice-public")
        )
        self.assertEqual(decision.user_id, "owner")
        self.assertEqual(decision.reason, "voice-confirmed")

    def test_clear_voice_match_confirms_at_observed_robot_score(self) -> None:
        stabilizer = IdentityStabilizer()
        decision = stabilizer.update(
            IdentityDecision("owner", 0.671, False, "voice-public")
        )
        self.assertEqual(decision.user_id, "owner")

    def test_centered_face_is_selected_as_speaker_candidate(self) -> None:
        side = FaceObservation(
            np.array([1.0], dtype=np.float32), 0.9, (50, 50), center_distance=0.28
        )
        center = FaceObservation(
            np.array([1.0], dtype=np.float32), 0.9, (160, 120), center_distance=0.02
        )
        self.assertIs(select_centered_face([side, center]), center)

    def test_multiple_faces_are_assigned_to_different_registered_people(self) -> None:
        self.store.ensure_user("daughter", "Daughter")
        for _ in range(3):
            self.store.add_biometric("owner", "face", [1.0, 0.0])
            self.store.add_biometric("daughter", "face", [0.0, 1.0])
        identifier = FaceIdentifier.__new__(FaceIdentifier)
        identifier.store = self.store
        observations = [
            FaceObservation(
                np.array([0.80, 0.60], dtype=np.float32),
                1.0,
                (160, 120),
            ),
            FaceObservation(
                np.array([0.70, 0.714], dtype=np.float32),
                1.0,
                (260, 120),
            ),
        ]
        matches = identifier.match_visible_faces(observations)
        self.assertEqual([item.user_id for item in matches], ["owner", "daughter"])

    def test_single_contaminated_template_does_not_win(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        templates = [
            ("owner", np.array([0.91, 0.41], dtype=np.float32), 1.0),
            ("owner", np.array([0.89, 0.46], dtype=np.float32), 1.0),
            ("owner", np.array([0.88, 0.48], dtype=np.float32), 1.0),
            ("guest", np.array([1.0, 0.0], dtype=np.float32), 1.0),
            ("guest", np.array([0.35, 0.94], dtype=np.float32), 1.0),
            ("guest", np.array([0.30, 0.95], dtype=np.float32), 1.0),
        ]
        user_id, _ = _match_templates(
            query, templates, threshold=0.50, ambiguity_margin=0.045
        )
        self.assertEqual(user_id, "owner")

    def test_explicit_chinese_self_introduction(self) -> None:
        self.assertEqual(extract_self_introduced_name("你好，我叫李明。"), "李明")
        self.assertEqual(extract_self_introduced_name("你叫我老王就行了"), "老王")
        self.assertEqual(
            extract_self_introduced_name("我叫张明远，麻烦你更新特征库。"),
            "张明远",
        )
        self.assertIsNone(extract_self_introduced_name("我是来问天气的"))
        self.assertIsNone(
            extract_self_introduced_name("那为什么你刚才叫我李小雨呢？")
        )
        self.assertIsNone(
            extract_self_introduced_name("你为什么后来又叫我张先生？")
        )
        self.assertIsNone(extract_self_introduced_name("我到底叫什么名字？"))
        self.assertIsNone(extract_self_introduced_name("我叫什么名字？"))

    def test_explicit_english_self_introduction(self) -> None:
        self.assertEqual(extract_self_introduced_name("My name is Alice."), "Alice")

    def test_explicit_third_person_introduction(self) -> None:
        self.assertEqual(
            extract_third_person_name("这是我女儿，她叫李小雨。"), "李小雨"
        )
        self.assertEqual(extract_third_person_name("我旁边的人叫李明"), "李明")
        self.assertIsNone(extract_third_person_name("她叫什么名字？"))
        self.assertIsNone(extract_third_person_name("她叫李小雨吗？"))

    def test_explicit_surname_declaration(self) -> None:
        self.assertEqual(extract_self_declared_surname("我姓张。"), "张")
        self.assertEqual(extract_self_declared_surname("我姓张，不姓王。"), "张")
        self.assertIsNone(extract_self_declared_surname("我姓什么？"))

    def test_conversational_identity_does_not_depend_on_preseeded_owner(self) -> None:
        user_id = conversational_user_id(self.store, "李明", b"face")
        self.assertTrue(user_id.startswith("person_"))
        self.assertNotEqual(user_id, "owner")

    def test_hidden_conversation_fact_is_extracted(self) -> None:
        text, facts = strip_memory_tags(
            "我会记住。[[REMEMBER:favorite_drink=茉莉花茶]]"
        )
        self.assertEqual(text, "我会记住。")
        self.assertEqual(facts, [("favorite_drink", "茉莉花茶")])

    def test_robot_name_is_persistent(self) -> None:
        self.store.set_setting("robot_name", "Atlas")
        self.assertEqual(self.store.get_setting("robot_name"), "Atlas")

    def test_stale_official_names_are_removed_by_authoritative_profile(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = MemoryStore(root / "memory.sqlite3", root / "users")
            store.ensure_user("owner", "张先生")
            store.set_setting("robot_name", "小助手")
            instance = store.official_instance_path("owner")
            assert instance is not None
            memory_file = instance / "memory.v1.json"
            memory_file.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "facts": [
                            {"id": "1", "text": "name: 张先生", "createdAt": 1},
                            {"id": "2", "text": "The user's preferred name is 老王.", "createdAt": 2},
                            {"id": "3", "text": "The robot's chosen name is Atlas.", "createdAt": 3},
                            {"id": "4", "text": "The robot's chosen name is 小助手.", "createdAt": 4},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            store.reconcile_official_memory()
            texts = [
                item["text"]
                for item in json.loads(memory_file.read_text(encoding="utf-8"))["facts"]
            ]
            self.assertEqual(
                texts, ["name: 张先生", "The robot's chosen name is 小助手."]
            )
            store.close()

    def test_explicit_robot_rename(self) -> None:
        self.assertEqual(extract_robot_name("以后你就叫阿特拉斯"), "阿特拉斯")
        self.assertEqual(extract_robot_name("Can I call you Atlas?"), "Atlas")
        self.assertEqual(extract_robot_name("你可以叫阿特拉斯 Atlas"), "Atlas")
        self.assertEqual(extract_robot_name("你的名字叫做小助手"), "小助手")
        self.assertEqual(extract_robot_name("你叫小助手"), "小助手")
        self.assertIsNone(extract_robot_name("我可以改你的名字吗？"))
        self.assertIsNone(extract_robot_name("你叫什么名字？"))

    def test_asr_punctuation_noise_is_ignored(self) -> None:
        self.assertFalse(_is_meaningful_utterance("..."))
        self.assertTrue(_is_meaningful_utterance("你好"))

    def test_identity_link_question_is_detected(self) -> None:
        self.assertIsNotNone(IDENTITY_LINK_QUESTION.search("特征库无法和名字关联起来吗？"))
        self.assertIsNotNone(IDENTITY_LINK_QUESTION.search("名字怎么和人脸关联起来？"))
        self.assertIsNotNone(
            BIOMETRIC_CAPABILITY_QUESTION.search("你有权限录入新的声纹和图像特征吗？")
        )
        self.assertIsNotNone(MODEL_BACKEND_QUESTION.search("你是什么模型？"))
        self.assertIsNotNone(FACE_ENROLL_REQUEST.search("记住我的脸"))
        self.assertIsNotNone(FACE_ENROLL_REQUEST.search("把我的人脸录入进去"))

    def test_similar_replies_are_detected(self) -> None:
        similarity = _text_similarity(
            "您好，请问有什么可以帮助您的吗？",
            "您好！请问有什么可以帮助您的吗？",
        )
        self.assertGreaterEqual(similarity, 0.80)

    def test_identity_uncertainty_does_not_clear_short_term_history(self) -> None:
        brain = RobotBrain()
        brain.record_external_turn("第一句", "我记得第一句。")
        brain.set_identity("owner", private_memory_allowed=True)
        brain.set_identity(None, private_memory_allowed=False)
        self.assertEqual(
            brain.history,
            [
                {"role": "user", "content": "第一句"},
                {"role": "assistant", "content": "我记得第一句。"},
            ],
        )

    def test_unknown_speaker_is_not_addressed_by_stale_family_name(self) -> None:
        self.store.ensure_user("daughter", "李小雨")
        brain = RobotBrain(memory=self.store)
        guarded = brain._guard_unverified_identity_reply(
            "我在听", "嗯，李小雨，我在听。"
        )
        self.assertEqual(guarded, "嗯，我在听。")

    def test_short_term_history_survives_agent_restart(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "conversation-history.json"
            first = RobotBrain(history_path=path)
            first.record_external_turn("我刚才问了什么？", "你刚才问的是天气。")
            restored = RobotBrain(history_path=path)
            self.assertEqual(restored.history, first.history)

    def test_speaker_identity_is_persisted_and_sent_with_each_user_turn(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "conversation-history.json"
            brain = RobotBrain(history_path=path)
            brain.record_user("这是我说的话")
            brain.label_last_user(
                "owner", "张先生", source="voice-public", confidence=0.678
            )
            restored = RobotBrain(history_path=path)
            self.assertEqual(restored.history[-1]["speaker_id"], "owner")
            self.assertEqual(restored.history[-1]["speaker_name"], "张先生")
            messages = restored._model_history(restored.history)
            self.assertEqual(messages[-1]["content"], "这是我说的话")

    def test_only_strong_visual_history_label_is_sent_to_model(self) -> None:
        brain = RobotBrain()
        messages = brain._model_history(
            [
                {
                    "role": "user",
                    "content": "这是我说的话",
                    "speaker_name": "张先生",
                    "speaker_source": "face-public",
                    "speaker_confidence": "0.850",
                }
            ]
        )
        self.assertEqual(messages[-1]["content"], "[说话者：张先生]\n这是我说的话")

    def test_history_window_uses_token_budget_and_keeps_newest_context(self) -> None:
        brain = RobotBrain(max_history_messages=200, max_history_tokens=40)
        brain.history = [
            {"role": "user", "content": "很早以前的无关内容" * 10},
            {"role": "assistant", "content": "中间回答"},
            {"role": "user", "content": "刚刚的问题"},
        ]
        window = brain._history_window()
        self.assertEqual(window[-1]["content"], "刚刚的问题")
        self.assertNotIn("很早以前的无关内容" * 10, [item["content"] for item in window])

    def test_route_specific_history_budget_can_exceed_local_window(self) -> None:
        brain = RobotBrain(
            max_history_messages=5000,
            max_history_tokens=40,
            cloud_history_tokens=900000,
        )
        brain.history = [
            {"role": "user", "content": "较早的有效云端上下文" * 10},
            {"role": "assistant", "content": "最近回答"},
        ]
        local_window = brain._history_window(token_budget=brain.max_history_tokens)
        cloud_window = brain._history_window(token_budget=brain.cloud_history_tokens)
        self.assertEqual(len(local_window), 1)
        self.assertEqual(len(cloud_window), 2)

    def test_generic_turn_is_not_saved_as_long_term_memory(self) -> None:
        brain = RobotBrain(
            memory=self.store,
            user_id="owner",
            private_memory_allowed=True,
        )
        brain._remember_turn("今天天气不错", "是的，天气很好。", [])
        self.assertEqual(self.store.recent_episodes("owner"), [])

    def test_model_memory_tool_updates_identity_label(self) -> None:
        brain = RobotBrain(
            memory=self.store,
            user_id="owner",
            private_memory_allowed=True,
        )
        brain.set_identity(
            "owner", private_memory_allowed=True, source="face+voice", confidence=0.9
        )
        brain._remember_turn(
            "请叫我张先生", "记住了。", [("preferred_name", "张先生")]
        )
        self.assertEqual(self.store.user_name("owner"), "张先生")
        fact = next(item for item in self.store.facts("owner") if item["key"] == "preferred_name")
        self.assertEqual(fact["visibility"], "public")

    def test_model_memory_tool_saves_surname_as_public_fact(self) -> None:
        brain = RobotBrain(
            memory=self.store,
            user_id="owner",
            private_memory_allowed=True,
        )
        brain.set_identity(
            "owner", private_memory_allowed=True, source="face+voice", confidence=0.9
        )
        brain._remember_turn("我姓张", "记住了。", [("surname", "张")])
        fact = next(item for item in self.store.facts("owner") if item["key"] == "surname")
        self.assertEqual(fact["value"], "张")
        self.assertEqual(fact["visibility"], "public")

    def test_unconfirmed_voice_cannot_rename_previous_speaker(self) -> None:
        brain = RobotBrain(
            memory=self.store,
            user_id="owner",
            private_memory_allowed=True,
        )
        brain.set_identity(
            "owner",
            private_memory_allowed=True,
            source="voice-public",
            confidence=0.65,
        )
        brain._remember_turn(
            "我叫另一个名字", "记住了。", [("preferred_name", "错误名字")]
        )
        self.assertEqual(self.store.user_name("owner"), "Owner")


if __name__ == "__main__":
    unittest.main()
