from __future__ import annotations

import sys

from gi.repository import Gst


Gst.init(None)
kind, device = sys.argv[1:3]
seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
description = f'{kind} device="{device}" ! audioconvert ! fakesink'
print(description)
pipeline = Gst.parse_launch(description)
state = pipeline.set_state(Gst.State.PLAYING)
print("state", state.value_nick)
try:
    message = pipeline.get_bus().timed_pop_filtered(
        seconds * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS
    )
    print("message", str(message.type) if message else "timeout")
    if message and message.type == Gst.MessageType.ERROR:
        print(message.parse_error())
finally:
    pipeline.set_state(Gst.State.NULL)
