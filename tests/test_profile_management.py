from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reachy_ai.memory import MemoryStore
from reachy_ai.robot_agent import RobotBrain


class ProfileManagementTests(unittest.TestCase):
    def test_rename_user_updates_display_name_and_survives_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "memory.sqlite3"
            store = MemoryStore(db_path, official_memory_root=None)
            store.ensure_user("member_a", "旧名字")
            store.upsert_fact("member_a", "name", "旧名字", source="conversation")
            store.add_biometric("member_a", "face", [0.1, 0.2, 0.3], quality=0.9)
            store.add_biometric("member_a", "voice", [0.4, 0.5, 0.6], quality=1.0)

            store.rename_user("member_a", "新名字")
            self.assertEqual(store.user_name("member_a"), "新名字")
            store.reconcile_identity_metadata()
            self.assertEqual(store.user_name("member_a"), "新名字")

            profiles = store.list_profiles()
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0]["display_name"], "新名字")
            self.assertEqual(profiles[0]["face_count"], 1)
            self.assertEqual(profiles[0]["voice_count"], 1)
            store.close()

    def test_clear_biometrics_delete_fact_and_delete_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "memory.sqlite3"
            store = MemoryStore(db_path, official_memory_root=None)
            store.ensure_user("member_b", "测试成员")
            store.add_biometric("member_b", "face", [0.1, 0.2], quality=0.9)
            store.add_biometric("member_b", "voice", [0.3, 0.4], quality=1.0)
            store.upsert_fact("member_b", "hobby", "音乐")

            removed_faces = store.clear_biometrics("member_b", "face")
            self.assertEqual(removed_faces, 1)
            self.assertEqual(store.biometric_count("member_b", "face"), 0)
            self.assertEqual(store.biometric_count("member_b", "voice"), 1)

            self.assertTrue(store.delete_fact("member_b", "hobby"))
            self.assertEqual(store.facts("member_b"), [])

            self.assertTrue(store.delete_user("member_b"))
            self.assertEqual(store.list_profiles(), [])
            self.assertEqual(store.biometric_count("member_b", "voice"), 0)
            store.close()

    def test_robot_brain_executes_rename_forget_and_biometric_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "memory.sqlite3"
            store = MemoryStore(db_path, official_memory_root=None)
            store.ensure_user("owner", "旧称呼")
            store.upsert_fact("owner", "job", "工程师")

            brain = RobotBrain(memory=store, user_id="owner", private_memory_allowed=True)
            brain.set_identity(
                "owner",
                private_memory_allowed=True,
                source="voice-confirmed",
                confidence=0.80,
            )

            with patch.object(brain, "_route", return_value={"route": "local"}):
                with patch.object(
                    brain,
                    "_chat",
                    return_value=(
                        "好的，已经帮你改名并更新特征。[[RENAME_USER:新称呼]][[FORGET:job]]"
                        "[[UPDATE_BIOMETRIC:voice:replace]][[CLEAR_BIOMETRIC:face]]",
                        {"route": "local"},
                    ),
                ):
                    reply, route = brain.answer("帮我改名叫新称呼，忘掉职业，重录声纹并清空人脸")

            self.assertEqual(reply, "好的，已经帮你改名并更新特征。")
            self.assertEqual(store.user_name("owner"), "新称呼")
            self.assertEqual([f["key"] for f in store.facts("owner") if f["key"] == "job"], [])
            self.assertEqual(route["renamed_users"], ["owner->新称呼"])
            self.assertEqual(route["forgotten"], ["job"])
            self.assertEqual(route["update_biometrics"], [("voice", True)])
            self.assertEqual(route["clear_biometrics"], ["face"])

            # Unverified identity must NOT rename anyone.
            brain.set_identity("owner", private_memory_allowed=False, source="voice-public", confidence=0.55)
            with patch.object(brain, "_route", return_value={"route": "local"}):
                with patch.object(
                    brain,
                    "_chat",
                    return_value=("好的。[[RENAME_USER:黑客]]", {"route": "local"}),
                ):
                    reply2, route2 = brain.answer("把我改名叫黑客")
            self.assertEqual(store.user_name("owner"), "新称呼")
            self.assertEqual(route2.get("profile_update_blocked"), "unverified_identity")
            self.assertIn("确认你的身份", reply2)
            store.close()

    def test_robot_brain_native_tool_calling_and_stabilized_voice_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "memory.sqlite3"
            store = MemoryStore(db_path, official_memory_root=None)
            store.ensure_user("owner", "旧名字")

            brain = RobotBrain(memory=store, user_id="owner", private_memory_allowed=True)
            # Matches real IdentityStabilizer 2-turn confirmed score (0.613 < 0.66).
            brain.set_identity(
                "owner",
                private_memory_allowed=True,
                source="voice-confirmed",
                confidence=0.613,
            )

            responses = [
                {
                    "reachy_router": {"route": "cloud", "model_id": "deepseek"},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "rename_family_member",
                                            "arguments": '{"new_name": "新名字"}',
                                        },
                                    },
                                    {
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "update_member_biometric",
                                            "arguments": '{"modality": "voice", "replace": true}',
                                        },
                                    },
                                ],
                            }
                        }
                    ],
                },
                {
                    "reachy_router": {"route": "cloud", "model_id": "deepseek"},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "好的，已经把名字改成新名字，并安排更新声纹。",
                            }
                        }
                    ],
                },
            ]

            with (
                patch.object(brain, "_route", return_value={"route": "cloud"}),
                patch("reachy_ai.robot_agent._post", side_effect=responses),
            ):
                reply, route = brain.answer("帮我把名字改成新名字并替换声纹")

            self.assertEqual(reply, "好的，已经把名字改成新名字，并安排更新声纹。")
            self.assertEqual(store.user_name("owner"), "新名字")
            self.assertEqual(route["update_biometrics"], [("voice", True)])
            store.close()


if __name__ == "__main__":
    unittest.main()
