from __future__ import annotations

import time
from pathlib import Path


def capture_frame(output_path: Path | str, timeout_seconds: float = 8.0) -> Path:
    """Capture one frame through Reachy's local IPC feed.

    This deliberately does not open the USB camera with OpenCV and does not call
    release_media(), so the daemon keeps ownership of both camera and microphone.
    """
    from reachy_mini import ReachyMini

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        log_level="WARNING",
    ) as reachy:
        while time.monotonic() < deadline:
            jpeg = reachy.media.get_frame_jpeg()
            if jpeg:
                target.write_bytes(jpeg)
                return target
            time.sleep(0.1)
    raise TimeoutError("Reachy IPC 摄像头在限定时间内没有返回画面")
