from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def _normalized_embedding(value: np.ndarray) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
    return embedding


def select_tracking_target(
    observations: list[Any],
    previous_embedding: np.ndarray | None = None,
    *,
    continuity_threshold: float = 0.45,
) -> Any:
    """Prefer the same face across frames, otherwise the face nearest image center."""
    if not observations:
        raise LookupError("No face detected")
    if previous_embedding is not None:
        previous = _normalized_embedding(previous_embedding)
        scored: list[tuple[float, Any]] = []
        for observation in observations:
            current = _normalized_embedding(observation.embedding)
            scored.append((float(np.dot(previous, current)), observation))
        score, candidate = max(scored, key=lambda item: item[0])
        if score >= continuity_threshold:
            return candidate
    return min(observations, key=lambda item: item.center_distance)


def find_locked_face(
    observations: list[Any],
    locked_embedding: np.ndarray,
    *,
    continuity_threshold: float = 0.45,
) -> Any | None:
    """Return only the previously locked face; never substitute another face."""
    if not observations:
        return None
    locked = _normalized_embedding(locked_embedding)
    scored: list[tuple[float, Any]] = []
    for observation in observations:
        current = _normalized_embedding(observation.embedding)
        scored.append((float(np.dot(locked, current)), observation))
    score, candidate = max(scored, key=lambda item: item[0])
    return candidate if score >= continuity_threshold else None


def sound_direction_changed(
    previous_radians: float | None,
    current_radians: float | None,
    *,
    threshold_radians: float = np.deg2rad(20.0),
) -> bool:
    """Compare circular sound directions and ignore small microphone jitter."""
    if current_radians is None:
        return False
    if previous_radians is None:
        return True
    delta = abs(
        float(
            np.arctan2(
                np.sin(float(current_radians) - float(previous_radians)),
                np.cos(float(current_radians) - float(previous_radians)),
            )
        )
    )
    return delta >= max(0.0, float(threshold_radians))


def should_turn_to_sound(
    *,
    has_lock: bool,
    locked_face_lost: bool,
    previous_radians: float | None,
    current_radians: float | None,
    threshold_radians: float = np.deg2rad(20.0),
) -> bool:
    """Actively reacquire on speech whenever no visible face is locked."""
    if current_radians is None:
        return False
    if has_lock and not locked_face_lost:
        return False
    return True


class ContinuousFaceTracker:
    """Monitor one locked face without issuing any motor commands.

    The class name is retained for compatibility with the existing agent.
    Movement is sound-triggered in robot_agent.py. This background thread only
    checks whether the exact face locked after the last sound-source turn is
    still visible.
    """

    def __init__(
        self,
        reachy: Any,
        face_identifier: Any,
        *,
        interval_seconds: float = 0.35,
        lost_timeout_seconds: float = 1.5,
        continuity_threshold: float = 0.45,
        **_: Any,
    ) -> None:
        self.reachy = reachy
        self.face_identifier = face_identifier
        self.interval_seconds = max(0.12, float(interval_seconds))
        self.lost_timeout_seconds = max(0.2, float(lost_timeout_seconds))
        self.continuity_threshold = min(
            1.0, max(0.05, float(continuity_threshold))
        )
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._paused_until = 0.0
        self._locked_embedding: np.ndarray | None = None
        self._last_seen_at = 0.0
        self._lost = False
        self._last_warning_at = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="reachy-face-lock-monitor",
            daemon=True,
        )
        self._thread.start()

    def enable(self, enabled: bool = True) -> None:
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()

    def pause_for(self, seconds: float) -> None:
        with self._state_lock:
            self._paused_until = max(
                self._paused_until,
                time.monotonic() + max(0.0, float(seconds)),
            )

    def lock_face(self, embedding: np.ndarray, *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        with self._state_lock:
            self._locked_embedding = _normalized_embedding(embedding)
            self._last_seen_at = timestamp
            self._lost = False

    def clear_lock(self) -> None:
        with self._state_lock:
            self._locked_embedding = None
            self._last_seen_at = 0.0
            self._lost = False

    @property
    def has_lock(self) -> bool:
        with self._state_lock:
            return self._locked_embedding is not None

    @property
    def locked_face_visible(self) -> bool:
        with self._state_lock:
            return self._locked_embedding is not None and not self._lost

    @property
    def locked_face_lost(self) -> bool:
        with self._state_lock:
            return self._locked_embedding is not None and self._lost

    def stop(self) -> None:
        self._stop.set()
        self._enabled.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            if not self._thread.is_alive():
                self._thread = None

    def _is_paused(self, now: float) -> bool:
        with self._state_lock:
            return now < self._paused_until

    def track_once(self, *, now: float | None = None) -> bool:
        """Observe one cached frame; never move the head or body."""
        now = time.monotonic() if now is None else float(now)
        if not self._enabled.is_set() or self._is_paused(now):
            return False
        with self._state_lock:
            locked = (
                None
                if self._locked_embedding is None
                else self._locked_embedding.copy()
            )
            last_seen = self._last_seen_at
        if locked is None:
            return False

        jpeg = self.reachy.media.get_frame_jpeg()
        observations: list[Any] = []
        if jpeg:
            try:
                observations = self.face_identifier.observe_all(jpeg)
            except LookupError:
                observations = []

        target = find_locked_face(
            observations,
            locked,
            continuity_threshold=self.continuity_threshold,
        )
        with self._state_lock:
            if target is not None:
                self._locked_embedding = _normalized_embedding(target.embedding)
                self._last_seen_at = now
                self._lost = False
                return True
            if last_seen and now - last_seen >= self.lost_timeout_seconds:
                self._lost = True
        return False

    def _run(self) -> None:
        self.enable(True)
        while not self._stop.is_set():
            try:
                self.track_once()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_warning_at >= 5.0:
                    logger.warning("Face lock monitor skipped: %s", exc)
                    self._last_warning_at = now
            self._stop.wait(self.interval_seconds)
