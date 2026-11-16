from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ReachyAI\app")

from reachy_ai.speech import SpeechEngine, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 Reachy AI 的中英双语语音链路")
    parser.add_argument("--text", default="你好，我是 Reachy Mini。Hello from Reachy Mini.")
    parser.add_argument("--output", type=Path, default=Path(r"D:\ReachyAI\data\tts-test.wav"))
    parser.add_argument("--hardware", action="store_true", help="通过 Reachy 扬声器播放并监听一句话")
    parser.add_argument("--show-pronunciation", action="store_true", help="显示实际送入语音合成的逐字读音")
    args = parser.parse_args()

    engine = SpeechEngine()
    samples, sample_rate = engine.synthesize(args.text)
    output = write_wav(args.output, samples, sample_rate)
    print(f"TTS 正常：{output} ({sample_rate} Hz, {len(samples) / sample_rate:.2f} 秒)")
    if args.show_pronunciation and engine._pronunciation is not None:
        if engine._pronunciation.backend == "dictionary-fallback":
            print("读音：已降级到原生词典，无法显示逐字确定的读音。")
        else:
            print("读音：", " ".join(engine._pronunciation.last_readings))

    if args.hardware:
        from reachy_mini import ReachyMini

        with ReachyMini(
            connection_mode="localhost_only",
            media_backend="default",
            log_level="WARNING",
        ) as reachy:
            engine.speak(reachy.media, args.text)
            print("请说一句话……")
            print("ASR：", engine.listen_once(reachy.media))


if __name__ == "__main__":
    main()
