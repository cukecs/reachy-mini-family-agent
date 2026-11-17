from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from reachy_ai.dev_server import DevDebugHandler
from reachy_ai.dev_tools import (
    DevInterfaceConfig,
    dev_edit_file,
    dev_list_files,
    dev_read_file,
    dev_search_code,
    resolve_workspace_path,
)
from reachy_ai.memory import MemoryStore
from reachy_ai.robot_agent import RobotBrain


class DevInterfaceTests(unittest.TestCase):
    def test_workspace_path_guard_blocks_secrets_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app").mkdir()
            (root / "app" / "sample.py").write_text("print('hi')\n", encoding="utf-8")
            (root / "config").mkdir()
            (root / "config" / "settings.json").write_text("{}", encoding="utf-8")
            (root / "config" / "settings.example.json").write_text("{}", encoding="utf-8")
            (root / "data").mkdir()
            (root / "data" / "memory.sqlite3").write_text("secret", encoding="utf-8")

            allowed = resolve_workspace_path("app/sample.py", root=root)
            self.assertEqual(allowed.name, "sample.py")
            ex = resolve_workspace_path("config/settings.example.json", root=root)
            self.assertEqual(ex.name, "settings.example.json")

            for forbidden in (
                "../outside.txt",
                "config/settings.json",
                "data/memory.sqlite3",
                ".env",
            ):
                with self.assertRaises(PermissionError):
                    resolve_workspace_path(forbidden, root=root)

            listed = dev_list_files("app", "*.py", root=root)
            self.assertEqual(listed["count"], 1)
            read_res = dev_read_file("app/sample.py", 1, 10, root=root)
            self.assertIn("1: print('hi')", read_res["content"])
            search_res = dev_search_code("print", "app", root=root)
            self.assertEqual(len(search_res["matches"]), 1)

            with self.assertRaises(PermissionError):
                dev_edit_file("app/sample.py", "hi", "bye", allow_code_edit=False, root=root)
            edit_res = dev_edit_file("app/sample.py", "hi", "bye", allow_code_edit=True, root=root)
            self.assertEqual(edit_res["replacements"], 1)
            self.assertIn("bye", (root / "app" / "sample.py").read_text(encoding="utf-8"))

    def test_dev_server_rest_and_mcp_jsonrpc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app").mkdir()
            (root / "app" / "demo.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
            (root / "data").mkdir()
            store = MemoryStore(root / "data" / "memory.sqlite3", official_memory_root=None)
            store.ensure_user("owner", "测试用户")
            store.close()

            cfg = DevInterfaceConfig(enabled=True, listen_host="127.0.0.1", listen_port=0)
            handler_cls = type(
                "_TestDevHandler",
                (DevDebugHandler,),
                {"dev_config": cfg, "dev_token": "secret-tok", "workspace_root": root},
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as resp:
                    html = resp.read().decode("utf-8")
                    self.assertIn("Reachy AI 远程开发与调试控制台", html)

                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/mcp",
                    data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    tools_payload = json.loads(resp.read().decode("utf-8"))
                tool_names = [t["name"] for t in tools_payload["result"]["tools"]]
                self.assertIn("reachy_read_file", tool_names)
                self.assertIn("reachy_diagnostics", tool_names)
                self.assertNotIn("reachy_edit_file", tool_names)

                call_req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/mcp",
                    data=json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "reachy_read_file",
                                "arguments": {"path": "app/demo.py", "start_line": 1, "end_line": 5},
                            },
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(call_req, timeout=3) as resp:
                    call_payload = json.loads(resp.read().decode("utf-8"))
                self.assertFalse(call_payload["result"]["isError"])
                self.assertIn("def hello", call_payload["result"]["content"][0]["text"])
            finally:
                server.shutdown()
                server.server_close()

    def test_robot_brain_can_invoke_ide_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "memory.sqlite3"
            store = MemoryStore(db_path, official_memory_root=None)
            store.ensure_user("owner", "开发者")
            brain = RobotBrain(memory=store, user_id="owner", private_memory_allowed=True)
            brain.set_identity("owner", private_memory_allowed=True, source="voice-confirmed", confidence=0.80)

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
                                        "id": "tc_dev_1",
                                        "type": "function",
                                        "function": {
                                            "name": "dev_search_code",
                                            "arguments": '{"query": "ConversationGate", "subdir": "app"}',
                                        },
                                    }
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
                                "content": "我在 app/reachy_ai/conversation_gate.py 里找到了 ConversationGate 类。",
                            }
                        }
                    ],
                },
            ]
            with (
                patch.object(brain, "_route", return_value={"route": "cloud"}),
                patch("reachy_ai.robot_agent._post", side_effect=responses),
            ):
                reply, _ = brain.answer("帮我在源码里搜一下 ConversationGate 在哪")

            self.assertIn("conversation_gate.py", reply)
            store.close()


if __name__ == "__main__":
    unittest.main()
