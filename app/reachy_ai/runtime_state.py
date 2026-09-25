from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


STATE_PATH = Path(r"D:\ReachyAI\data\runtime-state.json")


class RuntimeState:
    """Atomically publishes robot status and captions for the desktop console."""

    def __init__(self, path: Path = STATE_PATH, max_messages: int = 120) -> None:
        self.path = path
        self.max_messages = max_messages
        self._lock = threading.Lock()
        previous_messages: list[dict[str, Any]] = []
        try:
            previous = json.loads(self.path.read_text(encoding="utf-8"))
            raw_messages = previous.get("messages", [])
            if isinstance(raw_messages, list):
                previous_messages = [
                    item for item in raw_messages if isinstance(item, dict)
                ][-self.max_messages :]
        except (OSError, ValueError, TypeError):
            pass
        self._data: dict[str, Any] = {
            "status": "starting",
            "detail": "",
            # Captions are a local conversation transcript.  Keep them across
            # Agent restarts so both the UI and RobotBrain can recover context.
            "messages": previous_messages,
            "updated_at": self._now(),
        }
        self._write()

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def status(self, value: str, detail: str = "") -> None:
        with self._lock:
            self._data["status"] = value
            self._data["detail"] = detail
            self._data["updated_at"] = self._now()
            self._write()

    def event(self, role: str, text: str, **metadata: Any) -> None:
        cleaned = str(text).strip()
        if not cleaned:
            return
        with self._lock:
            message = {"role": role, "text": cleaned, "time": self._now()}
            message.update({k: v for k, v in metadata.items() if v is not None})
            self._data["messages"].append(message)
            self._data["messages"] = self._data["messages"][-self.max_messages :]
            self._data["updated_at"] = self._now()
            self._write()
