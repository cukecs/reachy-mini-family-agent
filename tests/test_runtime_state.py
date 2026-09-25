from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reachy_ai.runtime_state import RuntimeState


class RuntimeStateTests(unittest.TestCase):
    def test_status_and_captions_are_published_as_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = RuntimeState(path=path, max_messages=2)
            state.status("ready", "connected")
            state.event("user", "你好")
            state.event("assistant", "你好！", route="local")
            state.event("assistant", "第三条", route="cloud")

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "ready")
            self.assertEqual(len(saved["messages"]), 2)
            self.assertEqual(saved["messages"][-1]["text"], "第三条")
            self.assertEqual(saved["messages"][-1]["route"], "cloud")

    def test_captions_survive_agent_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            first = RuntimeState(path=path, max_messages=4)
            first.event("user", "上一句话")
            first.event("assistant", "上一句回答")
            second = RuntimeState(path=path, max_messages=4)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(second._data["messages"], saved["messages"])
            self.assertEqual(saved["messages"][-1]["text"], "上一句回答")


if __name__ == "__main__":
    unittest.main()
