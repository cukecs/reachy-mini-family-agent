from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\ReachyAI\app")

from reachy_ai.identity import FaceIdentifier, VoiceIdentifier
from reachy_ai.memory import MemoryStore
from reachy_ai.robot_agent import capture_reachy_frame
from reachy_ai.speech import SpeechEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll one Reachy household identity")
    parser.add_argument("--user", required=True, help="Stable internal user ID, e.g. owner")
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument("--face", action="store_true")
    parser.add_argument("--voice", action="store_true")
    args = parser.parse_args()
    if not args.face and not args.voice:
        parser.error("Choose --face, --voice, or both")

    from reachy_mini import ReachyMini

    store = MemoryStore()
    store.ensure_user(args.user, args.name)
    speech = SpeechEngine()
    face = FaceIdentifier(store) if args.face else None
    voice = VoiceIdentifier(store) if args.voice else None

    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        log_level="WARNING",
    ) as reachy:
        if face:
            print("Look at Reachy and slowly turn your head a little left and right.")
            enrolled = 0
            for index in range(5):
                frame = capture_reachy_frame(reachy.media, timeout_seconds=4.0)
                try:
                    embedding, quality = face.embedding(frame)
                except LookupError:
                    print(f"Face sample {index + 1}: no face")
                else:
                    store.add_biometric(args.user, "face", embedding, quality)
                    enrolled += 1
                    print(f"Face sample {index + 1}: saved (quality={quality:.3f})")
                time.sleep(0.4)
            if enrolled < 3:
                raise RuntimeError("Fewer than three usable face samples were captured")

        if voice:
            print("Speak naturally for at least five seconds after the prompt.")
            text, audio, sample_rate = speech.listen_once_with_audio(
                reachy.media, timeout_seconds=12.0
            )
            print(f"ASR heard: {text}")
            embedding = voice.embedding(audio, sample_rate)
            store.add_biometric(args.user, "voice", embedding, quality=1.0)
            print("Voice sample saved.")

    store.upsert_fact(args.user, "daily_english_enabled", "true")
    store.upsert_fact(args.user, "english_level", "middle_school")
    print(f"Enrollment complete: {args.user} ({args.name})")


if __name__ == "__main__":
    main()
