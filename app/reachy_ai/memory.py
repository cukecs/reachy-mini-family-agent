from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import win32crypt

from reachy_mini_conversation_app.memory import (
    add_memory_fact,
    forget_memory_fact,
    format_memory_for_prompt,
)


DEFAULT_DB = Path(r"D:\ReachyAI\data\memory.sqlite3")
DEFAULT_OFFICIAL_MEMORY_ROOT = Path(r"D:\ReachyAI\data\conversation_users")
DPAPI_FLAGS = 0x1  # CRYPTPROTECT_UI_FORBIDDEN


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _protect(data: bytes) -> bytes:
    return win32crypt.CryptProtectData(
        data, "ReachyAI biometric template", None, None, None, DPAPI_FLAGS
    )


def _unprotect(data: bytes) -> bytes:
    return win32crypt.CryptUnprotectData(data, None, None, None, DPAPI_FLAGS)[1]


@dataclass(frozen=True)
class DailyEnglish:
    user_id: str
    day: str
    phrase: str
    translation: str
    status: str


class MemoryStore:
    """Persistent user memory. Biometric templates are encrypted with DPAPI."""

    def __init__(
        self,
        path: Path | str = DEFAULT_DB,
        official_memory_root: Path | str | None = DEFAULT_OFFICIAL_MEMORY_ROOT,
    ):
        self.path = Path(path)
        self.official_memory_root = (
            Path(official_memory_root) if official_memory_root is not None else None
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self.reconcile_identity_metadata()
        self.reconcile_official_memory()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'user',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                summary TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_user_time
                ON episodes(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS task_state (
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                task_key TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, task_key)
            );
            CREATE TABLE IF NOT EXISTS daily_english (
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                day TEXT NOT NULL,
                phrase TEXT NOT NULL,
                translation TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'offered',
                attempt_text TEXT,
                score REAL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, day)
            );
            CREATE TABLE IF NOT EXISTS biometric_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                modality TEXT NOT NULL CHECK(modality IN ('face', 'voice')),
                encrypted_embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                quality REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_biometrics_user_modality
                ON biometric_templates(user_id, modality);
            """
        )
        self.connection.commit()

    def ensure_user(self, user_id: str, display_name: str | None = None) -> None:
        now = _now()
        self.connection.execute(
            """
            INSERT INTO users(user_id, display_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                display_name=COALESCE(?, users.display_name), updated_at=excluded.updated_at
            """,
            (user_id, display_name or user_id, now, now, display_name),
        )
        self.connection.commit()

    def user_name(self, user_id: str) -> str:
        row = self.connection.execute(
            "SELECT display_name FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        return str(row[0]) if row else user_id

    def user_ids(self) -> list[str]:
        return [str(row[0]) for row in self.connection.execute("SELECT user_id FROM users")]

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return all users with template and memory counts for management UI/CLI."""
        rows = self.connection.execute(
            """
            SELECT
                u.user_id,
                u.display_name,
                u.created_at,
                u.updated_at,
                (SELECT COUNT(*) FROM biometric_templates b WHERE b.user_id=u.user_id AND b.modality='face') AS face_count,
                (SELECT COUNT(*) FROM biometric_templates b WHERE b.user_id=u.user_id AND b.modality='voice') AS voice_count,
                (SELECT COUNT(*) FROM facts f WHERE f.user_id=u.user_id) AS fact_count,
                (SELECT COUNT(*) FROM episodes e WHERE e.user_id=u.user_id) AS episode_count
            FROM users u
            ORDER BY u.updated_at DESC, u.user_id ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def rename_user(self, user_id: str, new_name: str) -> None:
        """Update display name and canonical preferred_name fact together."""
        cleaned = str(new_name).strip()
        if not cleaned:
            raise ValueError("新名字不能为空")
        now = _now()
        self.ensure_user(user_id, cleaned)
        with self.connection:
            self.connection.execute(
                "UPDATE users SET display_name=?, updated_at=? WHERE user_id=?",
                (cleaned, now, user_id),
            )
            self.connection.execute(
                "DELETE FROM facts WHERE user_id=? AND key='name'",
                (user_id,),
            )
        self.upsert_fact(
            user_id,
            "preferred_name",
            cleaned,
            visibility="public",
            confidence=1.0,
            source="admin",
        )
        self.remember_official_fact(user_id, f"The user prefers to be called {cleaned}.")

    def delete_user(self, user_id: str) -> bool:
        """Delete a user profile and cascade-remove its templates and memory."""
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM users WHERE user_id=?", (user_id,)
            )
        return cursor.rowcount > 0

    def reconcile_identity_metadata(self) -> None:
        """Apply model-written canonical name facts to identity display labels."""
        rows = self.connection.execute(
            """
            SELECT user_id,key,value FROM facts
            WHERE key IN ('preferred_name','name')
            ORDER BY CASE key WHEN 'preferred_name' THEN 0 ELSE 1 END, updated_at DESC
            """
        ).fetchall()
        selected: dict[str, str] = {}
        for row in rows:
            user_id, value = str(row[0]), str(row[2]).strip()
            if user_id not in selected and value:
                selected[user_id] = value
        now = _now()
        with self.connection:
            for user_id, display_name in selected.items():
                self.connection.execute(
                    "UPDATE users SET display_name=?,updated_at=? WHERE user_id=?",
                    (display_name, now, user_id),
                )

    def reconcile_official_memory(self) -> None:
        """Remove stale name aliases superseded by authoritative user settings."""
        if self.official_memory_root is None or not self.official_memory_root.exists():
            return
        robot_name = self.get_setting("robot_name", "Reachy Mini") or "Reachy Mini"
        for user_id in self.user_ids():
            path = self.official_memory_root / user_id / "memory.v1.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                facts = payload.get("facts", [])
                if not isinstance(facts, list):
                    continue
                canonical_name = self.user_name(user_id)
                retained = []
                for fact in facts:
                    text = str(fact.get("text", "")) if isinstance(fact, dict) else ""
                    lower = text.casefold()
                    stale_user_name = (
                        ("preferred name" in lower or re.match(r"^name\s*:", lower))
                        and canonical_name not in text
                    )
                    stale_robot_name = (
                        "robot's chosen name" in lower and robot_name not in text
                    )
                    if not stale_user_name and not stale_robot_name:
                        retained.append(fact)
                if len(retained) == len(facts):
                    continue
                payload["facts"] = retained
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                temporary.replace(path)
            except (OSError, ValueError, TypeError):
                continue

    def set_setting(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO settings(key,value,updated_at) VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,updated_at=excluded.updated_at
            """,
            (key, value, _now()),
        )
        self.connection.commit()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return str(row[0]) if row else default

    def official_instance_path(self, user_id: str) -> Path | None:
        if self.official_memory_root is None:
            return None
        safe_user_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in user_id
        ).strip("_")
        if not safe_user_id:
            raise ValueError("user_id does not contain a usable path component")
        path = self.official_memory_root / safe_user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def remember_official_fact(self, user_id: str, text: str) -> None:
        instance_path = self.official_instance_path(user_id)
        if instance_path is not None:
            add_memory_fact(instance_path, text)

    def forget_official_fact(self, user_id: str, query: str) -> bool:
        instance_path = self.official_instance_path(user_id)
        if instance_path is None:
            return False
        return forget_memory_fact(instance_path, query=query).removed is not None

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        visibility: str = "private",
        confidence: float = 1.0,
        source: str = "user",
    ) -> None:
        self.ensure_user(user_id)
        self.connection.execute(
            """
            INSERT INTO facts(user_id,key,value,visibility,confidence,source,updated_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(user_id,key) DO UPDATE SET
                value=excluded.value, visibility=excluded.visibility,
                confidence=excluded.confidence, source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (user_id, key, value, visibility, confidence, source, _now()),
        )
        self.connection.commit()

    def facts(self, user_id: str, *, allow_private: bool = True, limit: int = 24) -> list[dict[str, Any]]:
        sql = "SELECT key,value,visibility,confidence FROM facts WHERE user_id=?"
        params: list[Any] = [user_id]
        if not allow_private:
            sql += " AND visibility='public'"
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.connection.execute(sql, params)]

    def delete_fact(self, user_id: str, key: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM facts WHERE user_id=? AND key=?", (user_id, key)
            )
        return cursor.rowcount > 0

    def add_episode(self, user_id: str, summary: str, importance: float = 0.5) -> None:
        self.ensure_user(user_id)
        self.connection.execute(
            "INSERT INTO episodes(user_id,summary,importance,created_at) VALUES(?,?,?,?)",
            (user_id, summary[:1200], max(0.0, min(1.0, importance)), _now()),
        )
        self.connection.commit()

    def recent_episodes(self, user_id: str, limit: int = 6) -> list[str]:
        rows = self.connection.execute(
            "SELECT summary FROM episodes WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        return [str(row[0]) for row in rows]

    def set_task(self, user_id: str, key: str, state: dict[str, Any]) -> None:
        self.ensure_user(user_id)
        self.connection.execute(
            """
            INSERT INTO task_state(user_id,task_key,state_json,updated_at) VALUES(?,?,?,?)
            ON CONFLICT(user_id,task_key) DO UPDATE SET
                state_json=excluded.state_json,updated_at=excluded.updated_at
            """,
            (user_id, key, json.dumps(state, ensure_ascii=False), _now()),
        )
        self.connection.commit()

    def get_task(self, user_id: str, key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT state_json FROM task_state WHERE user_id=? AND task_key=?",
            (user_id, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_daily_english(self, user_id: str, day: str | None = None) -> DailyEnglish | None:
        day = day or date.today().isoformat()
        row = self.connection.execute(
            "SELECT user_id,day,phrase,translation,status FROM daily_english WHERE user_id=? AND day=?",
            (user_id, day),
        ).fetchone()
        return DailyEnglish(**dict(row)) if row else None

    def save_daily_english(self, user_id: str, phrase: str, translation: str, day: str | None = None) -> DailyEnglish:
        self.ensure_user(user_id)
        day = day or date.today().isoformat()
        self.connection.execute(
            """
            INSERT INTO daily_english(user_id,day,phrase,translation,status,updated_at)
            VALUES(?,?,?,?, 'offered', ?)
            ON CONFLICT(user_id,day) DO NOTHING
            """,
            (user_id, day, phrase, translation, _now()),
        )
        self.connection.commit()
        result = self.get_daily_english(user_id, day)
        assert result is not None
        return result

    def complete_daily_english(self, user_id: str, attempt: str, score: float | None = None) -> None:
        self.connection.execute(
            """
            UPDATE daily_english SET status='completed',attempt_text=?,score=?,updated_at=?
            WHERE user_id=? AND day=?
            """,
            (attempt, score, _now(), user_id, date.today().isoformat()),
        )
        self.connection.commit()

    def memory_context(self, user_id: str, *, allow_private: bool = True) -> str:
        facts = self.facts(user_id, allow_private=allow_private)
        # A face-only or voice-only match may expose the public identity label,
        # but conversational episodes and task state remain private until both
        # modalities agree.
        episodes = self.recent_episodes(user_id) if allow_private else []
        canonical_name = self.user_name(user_id)
        episodes = [
            episode
            for episode in episodes
            if not (
                re.search(r"(?:自我介绍|我叫|名字|称呼)", episode, re.IGNORECASE)
                and canonical_name not in episode
            )
        ]
        daily = self.get_daily_english(user_id) if allow_private else None
        lines = [
            f"Authoritative current identity label: {self.user_name(user_id)} (id={user_id}).",
            "If an older memory line contains a different name, ignore the older name.",
        ]
        if facts:
            lines.append("Stable facts:")
            lines.extend(f"- {f['key']}: {f['value']}" for f in facts)
        if episodes:
            lines.append("Recent relevant events:")
            lines.extend(f"- {item}" for item in episodes)
        if daily:
            lines.append(
                f"Daily English {daily.day}: {daily.phrase} / {daily.translation}; status={daily.status}"
            )
        if allow_private:
            instance_path = self.official_instance_path(user_id)
            if instance_path is not None:
                official_memory = format_memory_for_prompt(instance_path)
                if official_memory:
                    lines.append(official_memory)
        return "\n".join(lines)

    def add_biometric(
        self,
        user_id: str,
        modality: str,
        embedding: Iterable[float],
        quality: float = 1.0,
        *,
        max_per_user: int = 12,
    ) -> None:
        import numpy as np

        self.ensure_user(user_id)
        vector = np.asarray(list(embedding), dtype=np.float32).reshape(-1)
        encrypted = _protect(vector.tobytes())
        self.connection.execute(
            """
            INSERT INTO biometric_templates(user_id,modality,encrypted_embedding,dimension,quality,created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (user_id, modality, encrypted, len(vector), quality, _now()),
        )
        self.connection.execute(
            """
            DELETE FROM biometric_templates
            WHERE id IN (
                SELECT id FROM biometric_templates
                WHERE user_id=? AND modality=?
                ORDER BY quality DESC, id DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (user_id, modality, max_per_user),
        )
        self.connection.commit()

    def biometric_count(self, user_id: str, modality: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM biometric_templates WHERE user_id=? AND modality=?",
            (user_id, modality),
        ).fetchone()
        return int(row[0]) if row else 0

    def clear_biometrics(self, user_id: str, modality: str | None = None) -> int:
        """Delete stored face/voice templates for user_id (or all modalities if None)."""
        with self.connection:
            if modality:
                cursor = self.connection.execute(
                    "DELETE FROM biometric_templates WHERE user_id=? AND modality=?",
                    (user_id, modality),
                )
            else:
                cursor = self.connection.execute(
                    "DELETE FROM biometric_templates WHERE user_id=?",
                    (user_id,),
                )
        return cursor.rowcount

    def biometric_templates(self, modality: str) -> list[tuple[str, Any, float]]:
        import numpy as np

        rows = self.connection.execute(
            "SELECT user_id,encrypted_embedding,dimension,quality FROM biometric_templates WHERE modality=?",
            (modality,),
        )
        result = []
        for row in rows:
            raw = _unprotect(bytes(row[1]))
            vector = np.frombuffer(raw, dtype=np.float32, count=int(row[2])).copy()
            result.append((str(row[0]), vector, float(row[3])))
        return result
