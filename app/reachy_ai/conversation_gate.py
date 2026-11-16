from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

SETTINGS_PATH = Path(r"D:\ReachyAI\config\settings.json")

_DEFAULT_WAKE_WORDS = ("天线宝宝", "Reachy")
_TIANXIAN_FUZZY_RE = re.compile(
    r"(?:天线|添线|天仙|天先)\s*(?:宝宝|抱抱|饱饱|包包)"
)
_LEADING_GREETING_RE = re.compile(
    r"^(?:(?:你好|您好|哈喽|嗨|喂|hey|hi|hello)[，,、\s]*)+",
    re.IGNORECASE,
)
_WAKE_ONLY_FILLER_RE = re.compile(
    r"^(?:在吗|在不在|在嘛|你好|您好|嗯|啊|呀|哦|吗|嘛|呢|hello|hi|hey)?[，,。.!！？?\s]*$",
    re.IGNORECASE,
)
_SLEEP_COMMAND_RE = re.compile(
    r"^(?:(?:好的|那|行|嗯)[，,\s]*)?"
    r"(?:(?:我)?(?:先)?看(?:会|一会|一下)?(?:视频|油管|youtube|电视|电影)|"
    r"(?:你)?(?:先)?(?:歇会儿|歇一会|休息一下|休息吧|安静|别插话|别搭茬|闭嘴|待机|休眠|退下)|"
    r"(?:进入|开启)(?:待机|休眠)(?:模式)?|"
    r"go to sleep|standby mode)(?:了|吧|啦)?[。！？.!?\s]*$",
    re.IGNORECASE,
)
_OPEN_MODE_RE = re.compile(
    r"(?:关闭|取消|退出)(?:唤醒|待机)(?:模式|门控)|(?:开启|进入)自由对话(?:模式)?",
    re.IGNORECASE,
)
_WAKE_MODE_RE = re.compile(
    r"(?:开启|打开|进入)(?:唤醒|免打扰|防误触)(?:模式|门控)?|"
    r"(?:只听|喊)(?:天线宝宝|唤醒词)(?:才)?(?:回答|说话|唤醒)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationGateConfig:
    enabled: bool = True
    mode: str = "wake_word"  # "wake_word" | "family_voice_or_wake" | "open"
    wake_words: tuple[str, ...] = _DEFAULT_WAKE_WORDS
    active_window_seconds: float = 45.0

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "ConversationGateConfig":
        defaults = cls()
        try:
            values = json.loads(path.read_text(encoding="utf-8")).get(
                "conversation_gate", {}
            )
        except (OSError, ValueError, TypeError):
            return defaults
        if not isinstance(values, dict):
            return defaults
        raw_mode = str(values.get("mode", defaults.mode)).strip().lower()
        mode = (
            raw_mode
            if raw_mode in {"wake_word", "family_voice_or_wake", "open"}
            else defaults.mode
        )
        raw_words = values.get("wake_words", defaults.wake_words)
        if isinstance(raw_words, str):
            parsed_words = tuple(
                part.strip()
                for part in re.split(r"[,，;；\s]+", raw_words)
                if part.strip()
            )
        elif isinstance(raw_words, list):
            parsed_words = tuple(
                str(item).strip() for item in raw_words if str(item).strip()
            )
        else:
            parsed_words = defaults.wake_words
        if not parsed_words:
            parsed_words = defaults.wake_words
        window = float(
            values.get("active_window_seconds", defaults.active_window_seconds)
        )
        return cls(
            enabled=bool(values.get("enabled", defaults.enabled)),
            mode=mode,
            wake_words=parsed_words,
            active_window_seconds=max(5.0, min(600.0, window)),
        )


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    cleaned_text: str
    reason: str
    woke_now: bool = False
    slept_now: bool = False
    wake_only: bool = False
    mode_changed: str | None = None


class ConversationGate:
    """Filter background media/ambient speech via wake word and active window."""

    def __init__(self, config: ConversationGateConfig | None = None) -> None:
        self.config = config or ConversationGateConfig.load()
        self.active_until: float = 0.0
        self.mode_override: str | None = None

    @property
    def current_mode(self) -> str:
        return self.mode_override or self.config.mode

    def is_awake(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return current < self.active_until

    def wake(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        self.active_until = current + self.config.active_window_seconds

    def sleep(self) -> None:
        self.active_until = 0.0

    def match_wake_word(
        self, text: str, extra_names: tuple[str, ...] = ()
    ) -> tuple[bool, str]:
        """Return (matched, text_without_leading_wake_phrase)."""
        raw = str(text or "").strip()
        if not raw:
            return False, ""

        candidates = list(dict.fromkeys([*self.config.wake_words, *extra_names]))
        cleaned = raw
        matched = False

        # Try stripping optional greeting prefix before matching wake word at start.
        without_greeting = _LEADING_GREETING_RE.sub("", raw).strip()
        fuzzy_match = _TIANXIAN_FUZZY_RE.search(raw)
        if fuzzy_match:
            matched = True
            if fuzzy_match.start() <= 6:
                cleaned = (
                    raw[: fuzzy_match.start()] + raw[fuzzy_match.end() :]
                ).strip(" ，,。.!！？?:：、")
                cleaned = _LEADING_GREETING_RE.sub("", cleaned).strip(
                    " ，,。.!！？?:：、"
                )

        if not matched:
            for candidate in candidates:
                word = str(candidate).strip()
                if not word:
                    continue
                if word.isascii():
                    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
                    m = pattern.search(raw)
                else:
                    m = re.search(re.escape(word), raw, re.IGNORECASE)
                if m:
                    matched = True
                    if m.start() <= len(raw) - len(without_greeting) + 4:
                        cleaned = (raw[: m.start()] + raw[m.end() :]).strip(
                            " ，,。.!！？?:：、"
                        )
                        cleaned = _LEADING_GREETING_RE.sub("", cleaned).strip(
                            " ，,。.!！？?:：、"
                        )
                    break

        if not matched:
            return False, raw
        if _WAKE_ONLY_FILLER_RE.match(cleaned):
            return True, ""
        return True, cleaned

    def evaluate(
        self,
        text: str,
        *,
        voice_user_id: str | None = None,
        now: float | None = None,
        extra_names: tuple[str, ...] = (),
    ) -> GateDecision:
        current = time.monotonic() if now is None else now
        raw = str(text or "").strip()
        if not raw:
            return GateDecision(False, "", "empty")

        matched_wake, stripped = self.match_wake_word(raw, extra_names=extra_names)
        if _WAKE_MODE_RE.search(raw):
            self.mode_override = "wake_word"
            self.sleep()
            return GateDecision(
                True,
                raw,
                "mode_wake_word",
                mode_changed="wake_word",
            )
        if _OPEN_MODE_RE.search(raw):
            self.mode_override = "open"
            self.wake(current)
            return GateDecision(
                True,
                raw,
                "mode_open",
                mode_changed="open",
            )

        mode = self.current_mode
        if not self.config.enabled or mode == "open":
            return GateDecision(True, stripped if matched_wake else raw, "open")

        awake = self.is_awake(current)
        sleep_target = stripped if matched_wake else raw
        if (awake or matched_wake) and _SLEEP_COMMAND_RE.match(sleep_target):
            self.sleep()
            return GateDecision(True, sleep_target, "sleep_command", slept_now=True)

        if matched_wake:
            self.wake(current)
            return GateDecision(
                True,
                stripped,
                "wake_word",
                woke_now=True,
                wake_only=(not stripped),
            )

        if awake:
            self.wake(current)
            return GateDecision(True, raw, "active_window")

        if mode == "family_voice_or_wake" and voice_user_id:
            self.wake(current)
            return GateDecision(True, raw, "family_voice")

        return GateDecision(False, "", "standby_ignored")
