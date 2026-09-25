from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from reachy_mini.media.audio_gstreamer import GStreamerAudio
from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs
from reachy_mini.media.camera_utils import scale_intrinsics
from reachy_mini.media.device_detection import get_video_device

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class DirectReachyCamera:
    """Single-owner Windows camera source for the autonomous local agent."""

    def __init__(self) -> None:
        Gst.init(None)
        device_name, detected_specs = get_video_device()
        if not device_name:
            raise RuntimeError("Reachy Mini Camera was not detected")
        if "Reachy Mini" not in device_name:
            raise RuntimeError(f"Unexpected camera selected: {device_name}")
        self.camera_specs = detected_specs or ReachyMiniLiteCamSpecs()
        self._resolution = self.camera_specs.default_resolution
        width, height, fps, crop_scale = self._resolution.value
        self.resolution = (width, height)
        self.K = scale_intrinsics(
            self.camera_specs.K,
            (3840, 2592),
            self.resolution,
            crop_scale,
        )
        self.D = self.camera_specs.D
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._recent_jpegs: deque[tuple[float, bytes]] = deque(maxlen=24)
        self._next_history_at = 0.0
        self._stop_capture = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._live_path = Path(r"D:\ReachyAI\data\live-camera.jpg")

        self.pipeline = Gst.Pipeline.new("reachy_direct_camera")
        source = Gst.ElementFactory.make("mfvideosrc")
        source.set_property("device-name", device_name)
        input_caps = Gst.ElementFactory.make("capsfilter")
        input_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"image/jpeg,width={width},height={height},framerate={fps}/1"
            ),
        )
        jpegdec = Gst.ElementFactory.make("jpegdec")
        convert = Gst.ElementFactory.make("videoconvert")
        output_caps = Gst.ElementFactory.make("capsfilter")
        output_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={width},height={height}"
            ),
        )
        self.appsink = Gst.ElementFactory.make("appsink")
        self.appsink.set_property("drop", True)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("sync", False)
        elements = (source, input_caps, jpegdec, convert, output_caps, self.appsink)
        if not all(elements):
            raise RuntimeError("Failed to create direct camera pipeline")
        for element in elements:
            self.pipeline.add(element)
        for upstream, downstream in zip(elements, elements[1:]):
            if not upstream.link(downstream):
                raise RuntimeError("Failed to link direct camera pipeline")

    def open(self) -> None:
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Reachy camera refused PLAYING state")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame = self._pull_frame()
            if frame is not None:
                self._publish_frame(frame, persist=True)
                self._capture_thread = threading.Thread(
                    target=self._capture_loop,
                    name="reachy-camera-capture",
                    daemon=True,
                )
                self._capture_thread.start()
                return
        raise RuntimeError("Reachy camera opened but returned no frames")

    def _pull_frame(self) -> np.ndarray | None:
        sample = self.appsink.emit("try-pull-sample", 250_000_000)
        if sample is None:
            return None
        buffer = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        data = buffer.extract_dup(0, buffer.get_size())
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()

    def _publish_frame(self, frame: np.ndarray, *, persist: bool) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        jpeg = encoded.tobytes() if ok else None
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_jpeg = jpeg
            now = time.monotonic()
            if jpeg and now >= self._next_history_at:
                # Keep a short temporal window for the VLM.  This is sampled,
                # not a second camera reader, so audio and preview stay stable.
                self._recent_jpegs.append((now, jpeg))
                self._next_history_at = now + 0.75
        if persist and jpeg:
            self._live_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._live_path.with_suffix(f".jpg.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(jpeg)
                os.replace(temporary, self._live_path)
            except OSError:
                # The preview window may briefly have the old JPEG open.
                # A missed preview frame must never stop the camera thread.
                temporary.unlink(missing_ok=True)

    def _capture_loop(self) -> None:
        next_persist = 0.0
        while not self._stop_capture.is_set():
            frame = self._pull_frame()
            if frame is None:
                continue
            now = time.monotonic()
            persist = now >= next_persist
            if persist:
                # Five preview frames per second are fluid enough for monitoring
                # without wasting CPU on JPEG encoding and disk replacement.
                next_persist = now + 0.20
            self._publish_frame(frame, persist=persist)
            self._stop_capture.wait(0.08)

    def read(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def read_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def recent_jpegs(
        self, *, window_seconds: float = 8.0, max_frames: int = 3
    ) -> list[bytes]:
        """Return evenly spaced recent keyframes in chronological order."""
        if max_frames <= 0:
            return []
        cutoff = time.monotonic() - max(0.5, window_seconds)
        with self._frame_lock:
            frames = [jpeg for captured_at, jpeg in self._recent_jpegs if captured_at >= cutoff]
        if len(frames) <= max_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return [frames[int(index)] for index in indices]

    def close(self) -> None:
        self._stop_capture.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
        self.pipeline.set_state(Gst.State.NULL)


class DirectReachyMedia:
    """Camera + microphone + speaker owned by exactly one local process."""

    def __init__(self) -> None:
        self.camera = DirectReachyCamera()
        self.camera.open()
        self.audio = GStreamerAudio(log_level="WARNING")

    def get_frame_jpeg(self) -> bytes | None:
        return self.camera.read_jpeg()

    def get_recent_frame_jpegs(
        self, *, window_seconds: float = 8.0, max_frames: int = 3
    ) -> list[bytes]:
        return self.camera.recent_jpegs(
            window_seconds=window_seconds, max_frames=max_frames
        )

    def get_input_audio_samplerate(self) -> int:
        return self.audio.get_input_audio_samplerate()

    def get_output_audio_samplerate(self) -> int:
        return self.audio.get_output_audio_samplerate()

    def get_output_channels(self) -> int:
        return self.audio.get_output_channels()

    def get_input_channels(self) -> int:
        return self.audio.get_input_channels()

    def get_DoA(self) -> tuple[float, bool] | None:
        return self.audio.get_DoA()

    def start_recording(self) -> None:
        self.audio.start_recording()

    def get_audio_sample(self):
        return self.audio.get_audio_sample()

    def stop_recording(self) -> None:
        self.audio.stop_recording()

    def start_playing(self) -> None:
        self.audio.start_playing()

    def push_audio_sample(self, sample) -> None:
        self.audio.push_audio_sample(sample)

    def stop_playing(self) -> None:
        self.audio.stop_playing()

    def close(self) -> None:
        self.camera.close()
        self.audio.cleanup()
