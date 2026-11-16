from __future__ import annotations

import json
import os
import threading
import time
import uuid
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
        self._pending_events: dict[str, tuple[str, dict[str, Any]]] = {}
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
        for attempt in range(5):
            try:
                os.replace(temporary, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01)

    def status(self, value: str, detail: str = "") -> None:
        with self._lock:
            self._data["status"] = value
            self._data["detail"] = detail
            self._data["updated_at"] = self._now()
            self._write()

    def begin_event(self, role: str, **metadata: Any) -> str:
        """Reserve an event ID without rendering an empty bubble before audio starts."""
        event_id = uuid.uuid4().hex[:12]
        clean_meta = {k: v for k, v in metadata.items() if v is not None}
        with self._lock:
            self._pending_events[event_id] = (role, clean_meta)
        return event_id

    def update_event(self, event_id: str, text: str, **metadata: Any) -> None:
        """Create or progressively update a synced transcript message in place."""
        cleaned = str(text).strip()
        if not cleaned:
            return
        with self._lock:
            for item in reversed(self._data["messages"]):
                if item.get("id") == event_id:
                    if item.get("text") == cleaned:
                        return
                    item["text"] = cleaned
                    for key, value in metadata.items():
                        if value is not None:
                            item[key] = value
                    self._data["updated_at"] = self._now()
                    self._write()
                    return

            role, base_meta = self._pending_events.pop(event_id, ("assistant", {}))
            message: dict[str, Any] = {
                "id": event_id,
                "role": role,
                "text": cleaned,
                "time": self._now(),
            }
            message.update(base_meta)
            message.update({k: v for k, v in metadata.items() if v is not None})
            self._data["messages"].append(message)
            self._data["messages"] = self._data["messages"][-self.max_messages :]
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
