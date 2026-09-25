from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\ReachyAI\app")))

from reachy_ai.config import RouterConfig
from reachy_ai.openai_client import ModelError
from reachy_ai.router import SmartRouter


class OfflineClient:
    def chat(self, *args, **kwargs):
        raise ModelError("offline")


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.router = SmartRouter(OfflineClient(), RouterConfig(long_input_chars=100))

    def test_greeting_uses_local_9b(self):
        result = self.router.decide([{"role": "user", "content": "你好"}])
        self.assertEqual(result.route, "local")
        self.assertEqual(result.source, "hard-rule")

    def test_explicit_deep_analysis_escalates(self):
        result = self.router.decide([{"role": "user", "content": "请深入分析这个问题"}])
        self.assertEqual(result.route, "cloud")

    def test_high_stakes_escalates(self):
        result = self.router.decide([{"role": "user", "content": "请给我具体药物剂量"}])
        self.assertEqual(result.route, "cloud")

    def test_long_input_escalates(self):
        result = self.router.decide([{"role": "user", "content": "x" * 101}])
        self.assertEqual(result.route, "cloud")

    def test_short_request_skips_classifier(self):
        result = self.router.decide([{"role": "user", "content": "把头向左转一点"}])
        self.assertEqual(result.route, "local")
        self.assertEqual(result.source, "hard-rule")

    def test_system_memory_does_not_make_short_request_complex(self):
        result = self.router.decide(
            [
                {"role": "system", "content": "long memory " * 100},
                {"role": "assistant", "content": "previous answer " * 100},
                {"role": "user", "content": "你应该怎么称呼我？"},
            ]
        )
        self.assertEqual(result.route, "local")
        self.assertEqual(result.source, "hard-rule")

    def test_explicit_deepseek_uses_cloud(self):
        result = self.router.decide(
            [{"role": "user", "content": "请用 DeepSeek 回答这个问题"}]
        )
        self.assertEqual(result.route, "cloud")

    def test_deepseek_alias_can_be_forced(self):
        result = self.router.decide(
            [{"role": "user", "content": "你好"}], force_route="deepseek"
        )
        self.assertEqual(result.route, "cloud")

    def test_ordinary_dialogue_uses_selected_default_model(self):
        router = SmartRouter(OfflineClient(), RouterConfig())
        result = router.decide([{"role": "user", "content": "我叫什么名字？"}])
        self.assertEqual(result.route, "local")

    def test_explicit_local_uses_default_role(self):
        router = SmartRouter(OfflineClient(), RouterConfig())
        result = router.decide(
            [{"role": "user", "content": "请用本地9B模型回答"}]
        )
        self.assertEqual(result.route, "local")

    def test_legacy_medium_override_maps_to_second_model(self):
        result = self.router.decide(
            [{"role": "user", "content": "你好"}], force_route="medium"
        )
        self.assertEqual(result.route, "cloud")


if __name__ == "__main__":
    unittest.main()
