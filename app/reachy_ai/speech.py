from __future__ import annotations

import math
import json
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sherpa_onnx
from scipy.signal import resample_poly

from .pronunciation import prepare_melo_lexicon
from .contextual_pronunciation import ContextualPronunciation
from .voice_analysis import SpeechAnalysis, SpeechRecognitionDetail, analyze_speech


SETTINGS_PATH = Path(r"D:\ReachyAI\config\settings.json")
SPEECH_CONTROL_PATH = Path(r"D:\ReachyAI\data\speech-control.json")
DOA_MAX_MEDIAN_ABSOLUTE_DEVIATION = np.deg2rad(20.0)
DOA_MAX_RANGE = np.deg2rad(55.0)


def stable_doa_direction(directions: list[float]) -> float | None:
    """Return a DoA only when one utterance has spatially stable samples.

    Reachy's documented range is [0, pi]: 0=left, pi/2=front/back,
    pi=right. Front/back cannot be resolved from DoA alone.
    """
    values = np.asarray(
        [
            float(value)
            for value in directions
            if np.isfinite(value) and -0.02 <= float(value) <= np.pi + 0.02
        ],
        dtype=np.float64,
    )
    if values.size < 2:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    spread = float(np.max(values) - np.min(values))
    if (
        mad > DOA_MAX_MEDIAN_ABSOLUTE_DEVIATION
        or spread > DOA_MAX_RANGE
    ):
        return None
    return max(0.0, min(float(np.pi), median))


@dataclass(frozen=True)
class SpeechPaths:
    root: Path = Path(r"D:\ReachyAI\models\speech")

    @property
    def asr(self) -> Path:
        return self.root / "asr"

    @property
    def sense_voice(self) -> Path:
        return (
            self.root
            / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
        )

    @property
    def tts(self) -> Path:
        return self.root / "tts"

    @property
    def vad(self) -> Path:
        return self.root / "vad" / "silero_vad.onnx"


@dataclass(frozen=True)
class VoiceActivityResult:
    accepted: bool
    speech_seconds: float
    longest_speech_seconds: float
    speech_ratio: float
    reason: str


@dataclass(frozen=True)
class SpeechFilterConfig:
    enabled: bool = True
    threshold: float = 0.58
    min_speech_seconds: float = 0.20
    min_contiguous_seconds: float = 0.128
    min_rms: float = 0.004

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "SpeechFilterConfig":
        defaults = cls()
        try:
            values = json.loads(path.read_text(encoding="utf-8")).get("speech_filter", {})
        except (OSError, ValueError, TypeError):
            return defaults
        return cls(
            enabled=bool(values.get("enabled", defaults.enabled)),
            threshold=max(0.05, min(0.95, float(values.get("threshold", defaults.threshold)))),
            min_speech_seconds=max(
                0.05,
                min(2.0, float(values.get("min_speech_seconds", defaults.min_speech_seconds))),
            ),
            min_contiguous_seconds=max(
                0.032,
                min(1.0, float(values.get("min_contiguous_seconds", defaults.min_contiguous_seconds))),
            ),
            min_rms=max(
                0.0,
                min(0.2, float(values.get("min_rms", defaults.min_rms))),
            ),
        )


def _find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"在 {root} 中找不到 {filename}")
    return matches[0]


_SENTENCE_SPLIT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]*")
_CLAUSE_SPLIT_RE = re.compile(r"[^，,、:：]+[，,、:：]*")


def split_speech_segments(text: str, max_chars: int = 44) -> list[str]:
    """Split text into speakable sentence/clause chunks for low-latency TTS."""
    cleaned = str(text).strip()
    if not cleaned:
        return []
    raw_sentences = [
        chunk.strip() for chunk in _SENTENCE_SPLIT_RE.findall(cleaned) if chunk.strip()
    ]
    if not raw_sentences:
        return [cleaned]

    segments: list[str] = []
    for sentence in raw_sentences:
        if len(sentence) <= max_chars:
            segments.append(sentence)
            continue
        clauses = [
            chunk.strip() for chunk in _CLAUSE_SPLIT_RE.findall(sentence) if chunk.strip()
        ]
        if len(clauses) <= 1:
            segments.append(sentence)
            continue
        buffer = ""
        for clause in clauses:
            if not buffer:
                buffer = clause
            elif len(buffer) < 10 or len(buffer) + len(clause) <= max_chars:
                buffer += clause
            else:
                segments.append(buffer)
                buffer = clause
        if buffer:
            segments.append(buffer)
    return segments


def build_progressive_slices(display_text: str, spoken_segments: list[str]) -> list[str]:
    """Map spoken TTS segments onto cumulative display prefixes."""
    full_display = str(display_text).strip()
    if not spoken_segments:
        return []
    if not full_display:
        cumulative = ""
        slices: list[str] = []
        for seg in spoken_segments:
            cumulative = f"{cumulative} {seg}".strip() if cumulative else seg
            slices.append(cumulative)
        return slices
    if len(spoken_segments) == 1:
        return [full_display]

    display_segments = split_speech_segments(full_display)
    if len(display_segments) == len(spoken_segments):
        slices = []
        end = 0
        for seg in display_segments:
            pos = full_display.find(seg, end)
            if pos != -1:
                end = pos + len(seg)
            else:
                end = min(len(full_display), end + len(seg))
            slices.append(full_display[:end].strip())
        slices[-1] = full_display
        return slices

    total_spoken = max(1, sum(len(seg) for seg in spoken_segments))
    total_display = len(full_display)
    acc = 0
    slices = []
    for idx, seg in enumerate(spoken_segments):
        acc += len(seg)
        if idx == len(spoken_segments) - 1:
            slices.append(full_display)
            break
        cutoff = max(1, min(total_display, round(total_display * (acc / total_spoken))))
        while cutoff < total_display and full_display[cutoff - 1].isascii() and full_display[cutoff].isascii() and full_display[cutoff].isalnum():
            cutoff += 1
        slices.append(full_display[:cutoff].strip())
    return slices


def _interpolate_progressive_text(
    previous_text: str, target_text: str, fraction: float
) -> str:
    """Reveal characters smoothly between previous_text and target_text."""
    if fraction >= 0.96 or len(target_text) <= len(previous_text) + 3:
        return target_text
    if fraction <= 0.0:
        return previous_text or target_text[:1]
    start_len = len(previous_text)
    delta = len(target_text) - start_len
    cutoff = start_len + max(1, round(delta * fraction))
    cutoff = min(len(target_text), max(start_len + 1, cutoff))
    while (
        cutoff < len(target_text)
        and target_text[cutoff - 1].isascii()
        and target_text[cutoff].isascii()
        and target_text[cutoff].isalnum()
    ):
        cutoff += 1
    return target_text[:cutoff].strip()


class SpeechEngine:
    """CPU-only multilingual ASR/TTS used through Reachy's shared media pipeline."""

    def __init__(
        self,
        paths: SpeechPaths | None = None,
        num_threads: int = 4,
        filter_config: SpeechFilterConfig | None = None,
    ):
        self.paths = paths or SpeechPaths()
        self.num_threads = num_threads
        self.filter_config = filter_config or SpeechFilterConfig.load()
        self._recognizer: sherpa_onnx.OnlineRecognizer | None = None
        self._offline_recognizer: sherpa_onnx.OfflineRecognizer | None = None
        self._tts: sherpa_onnx.OfflineTts | None = None
        self._pronunciation: ContextualPronunciation | None = None
        self._pronunciation_backend_logged: str | None = None
        self._vad: sherpa_onnx.VadModel | None = None
        self.last_speech_direction: float | None = None
        self.last_speech_direction_samples: tuple[float, ...] = ()
        self.last_voice_activity: VoiceActivityResult | None = None
        self.last_speech_detail: SpeechRecognitionDetail | None = None
        self.last_speech_analysis: SpeechAnalysis | None = None
        self._speech_interrupt_seen = self._interrupt_token()

    @staticmethod
    def _interrupt_token() -> str:
        try:
            payload = json.loads(SPEECH_CONTROL_PATH.read_text(encoding="utf-8"))
            return str(payload.get("interrupt", ""))
        except (OSError, ValueError, TypeError, AttributeError):
            return ""

    def _consume_interrupt(self) -> bool:
        token = self._interrupt_token()
        if token == self._speech_interrupt_seen:
            return False
        self._speech_interrupt_seen = token
        return True

    def _wait_interruptibly(self, seconds: float) -> bool:
        """Return False when a desktop-console interrupt arrives."""
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self._consume_interrupt():
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return not self._consume_interrupt()

    @property
    def recognizer(self) -> sherpa_onnx.OnlineRecognizer:
        if self._recognizer is None:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=str(_find_one(self.paths.asr, "tokens.txt")),
                encoder=str(_find_one(self.paths.asr, "encoder.int8.onnx")),
                decoder=str(_find_one(self.paths.asr, "decoder.int8.onnx")),
                num_threads=self.num_threads,
                provider="cpu",
                sample_rate=16000,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=20.0,
            )
        return self._recognizer

    @property
    def offline_recognizer(self) -> sherpa_onnx.OfflineRecognizer:
        if self._offline_recognizer is None:
            model_dir = self.paths.sense_voice
            self._offline_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_dir / "model.int8.onnx"),
                tokens=str(model_dir / "tokens.txt"),
                num_threads=self.num_threads,
                provider="cpu",
                language="auto",
                use_itn=True,
            )
        return self._offline_recognizer

    @property
    def tts(self) -> sherpa_onnx.OfflineTts:
        if self._tts is None:
            model_dir = next(
                (p.parent for p in self.paths.tts.rglob("model.onnx")),
                None,
            )
            if model_dir is None:
                raise FileNotFoundError(f"在 {self.paths.tts} 中找不到 TTS model.onnx")

            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / "model.onnx"),
                lexicon=str(prepare_melo_lexicon(model_dir)),
                tokens=str(model_dir / "tokens.txt"),
                data_dir="",
                dict_dir="",
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
            )
            model = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits,
                num_threads=self.num_threads,
                provider="cpu",
                debug=False,
            )
            config = sherpa_onnx.OfflineTtsConfig(
                model=model,
                rule_fsts=",".join(
                    str(model_dir / name)
                    for name in ("phone.fst", "date.fst", "number.fst")
                ),
                max_num_sentences=2,
                silence_scale=0.15,
            )
            if not config.validate():
                raise RuntimeError(f"TTS 模型配置无效：{model_dir}")
            self._tts = sherpa_onnx.OfflineTts(config)
            tokens = {line.split()[0] for line in
                      (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines() if line.split()}
            self._pronunciation = ContextualPronunciation(self.paths.root / "g2pw", tokens)
        return self._tts

    @property
    def vad(self) -> sherpa_onnx.VadModel:
        if self._vad is None:
            if not self.paths.vad.exists():
                raise FileNotFoundError(f"缺少人声检测模型：{self.paths.vad}")
            silero = sherpa_onnx.SileroVadModelConfig(
                model=str(self.paths.vad),
                threshold=self.filter_config.threshold,
                min_silence_duration=0.45,
                min_speech_duration=0.20,
                window_size=512,
                max_speech_duration=20.0,
            )
            config = sherpa_onnx.VadModelConfig(
                silero_vad=silero,
                sample_rate=16000,
                num_threads=1,
                provider="cpu",
                debug=False,
            )
            if not config.validate():
                raise RuntimeError(f"人声检测模型配置无效：{self.paths.vad}")
            self._vad = sherpa_onnx.VadModel.create(config)
        return self._vad

    def analyze_voice_activity(
        self, samples: np.ndarray, sample_rate: int
    ) -> VoiceActivityResult:
        """Reject impulsive environmental sounds before they reach ASR."""
        if not self.filter_config.enabled:
            return VoiceActivityResult(True, 0.0, 0.0, 1.0, "filter_disabled")
        mono = np.asarray(samples, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        mono = np.nan_to_num(mono.reshape(-1), copy=False)
        if sample_rate <= 0 or mono.size < int(sample_rate * 0.12):
            return VoiceActivityResult(False, 0.0, 0.0, 0.0, "audio_too_short")
        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            mono = resample_poly(
                mono, 16000 // divisor, sample_rate // divisor
            ).astype(np.float32)
        mono = np.clip(mono, -1.0, 1.0)
        detector = self.vad
        detector.reset()
        flags: list[bool] = []
        voiced_frames: list[np.ndarray] = []
        window = detector.window_size()
        for offset in range(0, len(mono), window):
            frame = mono[offset : offset + window]
            if len(frame) < window:
                frame = np.pad(frame, (0, window - len(frame)))
            is_voiced = bool(detector.is_speech(frame))
            flags.append(is_voiced)
            if is_voiced:
                voiced_frames.append(frame)
        frame_seconds = window / 16000.0
        speech_frames = sum(flags)
        longest = 0
        run = 0
        for is_speech in flags:
            if is_speech:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        speech_seconds = speech_frames * frame_seconds
        longest_seconds = longest * frame_seconds
        ratio = speech_frames / max(1, len(flags))
        voiced_rms = (
            float(np.sqrt(np.mean(np.square(np.concatenate(voiced_frames)))))
            if voiced_frames
            else 0.0
        )
        accepted = (
            speech_seconds >= self.filter_config.min_speech_seconds
            and longest_seconds >= self.filter_config.min_contiguous_seconds
            and voiced_rms >= self.filter_config.min_rms
        )
        if accepted:
            reason = "human_voice"
        elif voiced_frames and voiced_rms < self.filter_config.min_rms:
            reason = "too_quiet_ambient"
        else:
            reason = "non_speech_or_impulse"
        return VoiceActivityResult(
            accepted,
            round(speech_seconds, 3),
            round(longest_seconds, 3),
            round(ratio, 3),
            reason,
        )

    def transcribe_rich(
        self, samples: np.ndarray, sample_rate: int = 16000
    ) -> SpeechRecognitionDetail:
        mono = np.asarray(samples, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        mono = np.clip(mono.reshape(-1), -1.0, 1.0)
        if mono.size == 0:
            return SpeechRecognitionDetail(text="")
        stream = self.offline_recognizer.create_stream()
        stream.accept_waveform(sample_rate, mono)
        self.offline_recognizer.decode_stream(stream)
        res = stream.result
        return SpeechRecognitionDetail(
            text=str(getattr(res, "text", "") or "").strip(),
            lang=str(getattr(res, "lang", "") or ""),
            emotion=str(getattr(res, "emotion", "") or ""),
            event=str(getattr(res, "event", "") or ""),
            tokens=tuple(str(t) for t in (getattr(res, "tokens", None) or ())),
            timestamps=tuple(float(t) for t in (getattr(res, "timestamps", None) or ())),
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        return self.transcribe_rich(samples, sample_rate).text

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        tts = self.tts
        prepared = self._pronunciation.prepare(text) if self._pronunciation is not None else text
        if (self._pronunciation is not None and self._pronunciation.backend != "not-loaded"
                and self._pronunciation.backend != self._pronunciation_backend_logged):
            self._pronunciation_backend_logged = self._pronunciation.backend
            print(f"TTS pronunciation frontend: {self._pronunciation.backend} (CPU)")
        audio = tts.generate(prepared, sid=0, speed=speed)
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)

    def listen_once(self, media, timeout_seconds: float = 15.0) -> str:
        text, _, _ = self.listen_once_with_audio(media, timeout_seconds)
        return text

    def listen_once_with_audio(
        self,
        media,
        timeout_seconds: float = 15.0,
        *,
        abort_check: Callable[[], bool] | None = None,
    ) -> tuple[str, np.ndarray, int]:
        """Listen via Reachy MediaManager without opening the USB mic directly."""
        recognizer = self.recognizer
        stream = recognizer.create_stream()
        sample_rate = media.get_input_audio_samplerate()
        deadline = time.monotonic() + timeout_seconds
        last_text = ""
        captured: list[np.ndarray] = []
        directions: list[float] = []
        next_direction_sample = 0.0
        self.last_speech_direction = None
        self.last_speech_direction_samples = ()
        media.start_recording()
        aborted = False
        try:
            while time.monotonic() < deadline:
                if abort_check is not None and abort_check():
                    aborted = True
                    break
                chunk = media.get_audio_sample()
                if chunk is None:
                    time.sleep(0.01)
                    continue
                samples = np.asarray(chunk, dtype=np.float32)
                now = time.monotonic()
                if now >= next_direction_sample:
                    next_direction_sample = now + 0.10
                    get_direction = getattr(media, "get_DoA", None)
                    if callable(get_direction):
                        direction = get_direction()
                        if direction is not None and direction[1]:
                            directions.append(float(direction[0]))
                if samples.ndim == 2:
                    samples = samples.mean(axis=1)
                captured.append(samples.copy())
                stream.accept_waveform(sample_rate, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                text = recognizer.get_result(stream).strip()
                if text:
                    last_text = text
                if last_text and recognizer.is_endpoint(stream):
                    break
        finally:
            media.stop_recording()

        if aborted:
            return "", np.empty(0, dtype=np.float32), sample_rate

        stream.accept_waveform(sample_rate, np.zeros(int(sample_rate * 0.5), np.float32))
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        audio = np.concatenate(captured) if captured else np.empty(0, dtype=np.float32)
        self.last_voice_activity = (
            self.analyze_voice_activity(audio, sample_rate)
            if audio.size
            else VoiceActivityResult(False, 0.0, 0.0, 0.0, "empty_audio")
        )
        self.last_speech_detail = None
        self.last_speech_analysis = None
        if audio.size and self.last_voice_activity.accepted:
            detail = self.transcribe_rich(audio, sample_rate)
            text = detail.text
            if not text:
                online_text = (recognizer.get_result(stream).strip() or last_text).strip()
                text = online_text
                detail = SpeechRecognitionDetail(
                    text=text,
                    lang=detail.lang,
                    emotion=detail.emotion,
                    event=detail.event,
                    tokens=detail.tokens,
                    timestamps=detail.timestamps,
                )
            self.last_speech_detail = detail
            if text:
                self.last_speech_analysis = analyze_speech(audio, sample_rate, detail)
        else:
            text = ""
        if directions:
            self.last_speech_direction_samples = tuple(directions)
            self.last_speech_direction = stable_doa_direction(directions)
        return text, audio, sample_rate

    def listen_melody_once_with_audio(
        self,
        media,
        *,
        timeout_seconds: float = 12.0,
        maximum_singing_seconds: float = 10.0,
        trailing_silence_seconds: float = 1.0,
    ) -> tuple[str, np.ndarray, int]:
        """Capture singing even when ASR cannot produce words.

        Unlike the speech endpoint detector, this ends from acoustic silence,
        so humming does not have to wait for the full recognition timeout.
        """
        sample_rate = media.get_input_audio_samplerate()
        deadline = time.monotonic() + timeout_seconds
        captured: list[np.ndarray] = []
        directions: list[float] = []
        noise_levels: list[float] = []
        next_direction_sample = 0.0
        listening_started = time.monotonic()
        singing_started: float | None = None
        last_voiced: float | None = None
        self.last_speech_direction = None
        self.last_speech_direction_samples = ()
        media.start_recording()
        try:
            while time.monotonic() < deadline:
                chunk = media.get_audio_sample()
                if chunk is None:
                    time.sleep(0.01)
                    continue
                samples = np.asarray(chunk, dtype=np.float32)
                now = time.monotonic()
                if now >= next_direction_sample:
                    next_direction_sample = now + 0.10
                    get_direction = getattr(media, "get_DoA", None)
                    if callable(get_direction):
                        direction = get_direction()
                        if direction is not None and direction[1]:
                            directions.append(float(direction[0]))
                if samples.ndim == 2:
                    samples = samples.mean(axis=1)
                samples = np.clip(samples.reshape(-1), -1.0, 1.0)
                captured.append(samples.copy())
                rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
                if now - listening_started <= 0.35 and rms <= 0.012:
                    noise_levels.append(rms)
                noise_floor = float(np.median(noise_levels)) if noise_levels else 0.0
                start_threshold = max(0.008, noise_floor * 3.0)
                continue_threshold = max(0.005, noise_floor * 1.8)
                if singing_started is None:
                    if rms >= start_threshold:
                        singing_started = now
                        last_voiced = now
                elif rms >= continue_threshold:
                    last_voiced = now
                if singing_started is not None:
                    if now - singing_started >= maximum_singing_seconds:
                        break
                    if last_voiced is not None and now - last_voiced >= trailing_silence_seconds:
                        break
        finally:
            media.stop_recording()

        audio = np.concatenate(captured) if captured else np.empty(0, dtype=np.float32)
        text = self.transcribe(audio, sample_rate) if audio.size else ""
        self.last_voice_activity = None
        if directions:
            self.last_speech_direction_samples = tuple(directions)
            self.last_speech_direction = stable_doa_direction(directions)
        return text, audio, sample_rate

    @staticmethod
    def _output_channels(media) -> int:
        channels = 2
        get_channels = getattr(media, "get_output_channels", None)
        if callable(get_channels):
            channels = int(get_channels())
        return max(1, channels)

    @staticmethod
    def _prepare_playback_samples(
        samples: np.ndarray,
        source_rate: int,
        output_rate: int,
        channels: int,
        *,
        tail_seconds: float = 0.40,
    ) -> np.ndarray:
        mono = np.asarray(samples, dtype=np.float32)
        if source_rate != output_rate and source_rate > 0 and output_rate > 0:
            divisor = math.gcd(source_rate, output_rate)
            mono = resample_poly(
                mono,
                output_rate // divisor,
                source_rate // divisor,
            ).astype(np.float32)

        mono = np.clip(mono.reshape(-1), -1.0, 1.0)
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        if peak > 0.88:
            mono = mono * (0.88 / peak)

        tail_frames = int(output_rate * max(0.0, tail_seconds))
        if tail_frames > 0:
            mono = np.concatenate([mono, np.zeros(tail_frames, dtype=np.float32)])

        # Reachy's Windows GStreamer output is stereo. Passing a 1-D mono
        # buffer makes it interpret adjacent samples as L/R pairs, producing
        # the fast, garbled "wah-wah" sound reported by the user.
        return (
            np.repeat(mono[:, None], channels, axis=1)
            if channels > 1
            else mono
        ).astype(np.float32, copy=False)

    def speak(
        self,
        media,
        text: str,
        speed: float = 1.0,
        *,
        display_text: str | None = None,
        on_text_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Speak through Reachy's audio output with sentence pipelining and synced captions."""
        # Consume an old click made while the robot was silent.  A new click
        # arriving during synthesis remains visible and cancels this reply.
        self._consume_interrupt()
        segments = split_speech_segments(text)
        if not segments:
            return
        if len(segments) == 1 and on_text_progress is None:
            samples, source_rate = self.synthesize(segments[0], speed=speed)
            self.play_samples(media, samples, source_rate)
            return

        output_rate = media.get_output_audio_samplerate()
        channels = self._output_channels(media)
        slices = build_progressive_slices(display_text or text, segments)
        synth_queue: queue.Queue[tuple[int, np.ndarray, float] | Exception | None] = queue.Queue(
            maxsize=2
        )
        stop_synthesis = threading.Event()

        def _producer() -> None:
            try:
                for idx, segment in enumerate(segments):
                    if stop_synthesis.is_set():
                        return
                    samples, source_rate = self.synthesize(segment, speed=speed)
                    if stop_synthesis.is_set():
                        return
                    tail = 0.40 if idx == len(segments) - 1 else 0.08
                    playback = self._prepare_playback_samples(
                        samples,
                        source_rate,
                        output_rate,
                        channels,
                        tail_seconds=tail,
                    )
                    speech_seconds = (
                        max(0.05, (len(playback) - int(output_rate * tail)) / output_rate)
                        if output_rate > 0
                        else 0.1
                    )
                    while not stop_synthesis.is_set():
                        try:
                            synth_queue.put((idx, playback, speech_seconds), timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                synth_queue.put(exc)
            finally:
                while not stop_synthesis.is_set():
                    try:
                        synth_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        def _next_item():
            while True:
                if self._consume_interrupt():
                    stop_synthesis.set()
                    return None, True
                try:
                    item = synth_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if isinstance(item, Exception):
                    stop_synthesis.set()
                    raise item
                return item, False

        first_item, interrupted = _next_item()
        if interrupted or first_item is None:
            stop_synthesis.set()
            return

        block_frames = max(1, int(output_rate * 0.20))
        lead_seconds = 0.60
        timeline: list[tuple[float, float, str, str]] = []
        last_emitted = ""
        last_emit_at = 0.0
        total_queued_seconds = 0.0

        def _sync_ui(started_at: float) -> None:
            nonlocal last_emitted, last_emit_at
            if on_text_progress is None or not timeline:
                return
            elapsed = max(0.0, time.monotonic() - started_at)
            active = timeline[0]
            for entry in timeline:
                if elapsed >= entry[0]:
                    active = entry
                else:
                    break
            seg_start, seg_end, prev_slice, target_slice = active
            seg_span = max(0.15, seg_end - seg_start)
            fraction = min(1.0, max(0.0, (elapsed - seg_start) / seg_span))
            candidate = _interpolate_progressive_text(prev_slice, target_slice, fraction)
            now = time.monotonic()
            if (
                candidate
                and candidate != last_emitted
                and (
                    not last_emitted
                    or fraction >= 0.96
                    or now - last_emit_at >= 0.16
                )
            ):
                last_emitted = candidate
                last_emit_at = now
                on_text_progress(candidate)

        def _wait_synced(until_time: float, started_at: float) -> bool:
            while time.monotonic() < until_time:
                if self._consume_interrupt():
                    return False
                _sync_ui(started_at)
                step = min(0.08, max(0.0, until_time - time.monotonic()))
                if step > 0:
                    time.sleep(step)
            _sync_ui(started_at)
            return not self._consume_interrupt()

        media.start_playing()
        started = time.monotonic()
        aborted = False
        try:
            current_item = first_item
            while current_item is not None:
                idx, playback, speech_seconds = current_item
                prev_slice = slices[idx - 1] if idx > 0 else ""
                target_slice = slices[idx]
                seg_start_sec = total_queued_seconds
                seg_end_sec = seg_start_sec + speech_seconds
                timeline.append((seg_start_sec, seg_end_sec, prev_slice, target_slice))
                seg_total_sec = (len(playback) / output_rate) if output_rate > 0 else 0.0

                for offset in range(0, len(playback), block_frames):
                    if self._consume_interrupt():
                        aborted = True
                        break
                    block = np.ascontiguousarray(playback[offset : offset + block_frames])
                    media.push_audio_sample(block)
                    _sync_ui(started)
                    queued_until = seg_start_sec + (
                        (offset + len(block)) / output_rate if output_rate > 0 else 0.0
                    )
                    target_elapsed = max(0.0, queued_until - lead_seconds)
                    if not _wait_synced(started + target_elapsed, started):
                        aborted = True
                        break

                if aborted:
                    stop_synthesis.set()
                    break
                total_queued_seconds += seg_total_sec
                current_item, interrupted = _next_item()
                if interrupted:
                    aborted = True
                    break

            if not aborted:
                _wait_synced(started + total_queued_seconds + 0.35, started)
                if on_text_progress is not None and slices and last_emitted != slices[-1]:
                    on_text_progress(slices[-1])
        finally:
            stop_synthesis.set()
            media.stop_playing()

    def play_wav(self, media, path: Path | str) -> None:
        """Play a PCM WAV through Reachy's shared speaker pipeline."""
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        if sample_width != 2:
            raise ValueError(f"只支持 16-bit PCM WAV，实际位宽为 {sample_width * 8} bit")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        self.play_samples(media, samples, sample_rate)

    def play_samples(
        self, media, samples: np.ndarray, source_rate: int
    ) -> None:
        """Play mono float audio with resampling and an underrun-safe buffer."""
        output_rate = media.get_output_audio_samplerate()
        # A stop click made while TTS was still synthesizing also cancels the
        # pending playback instead of starting a stale, unwanted reply.
        if self._consume_interrupt():
            return
        channels = self._output_channels(media)
        playback = self._prepare_playback_samples(
            samples,
            source_rate,
            output_rate,
            channels,
            tail_seconds=0.40,
        )

        # Keep about 600 ms queued ahead of real time. The previous 50 ms
        # lock-step feed could underrun whenever Windows delayed this thread.
        block_frames = max(1, int(output_rate * 0.20))
        lead_seconds = 0.60
        duration = len(playback) / output_rate if output_rate else 0.0
        media.start_playing()
        started = time.monotonic()
        try:
            for offset in range(0, len(playback), block_frames):
                if self._consume_interrupt():
                    break
                block = np.ascontiguousarray(playback[offset : offset + block_frames])
                media.push_audio_sample(block)
                queued_until = (offset + len(block)) / output_rate
                target_elapsed = max(0.0, queued_until - lead_seconds)
                wait = started + target_elapsed - time.monotonic()
                if wait > 0 and not self._wait_interruptibly(wait):
                    break
            else:
                wait = started + duration + 0.35 - time.monotonic()
                if wait > 0:
                    self._wait_interruptibly(wait)
        finally:
            media.stop_playing()


def write_wav(path: Path | str, samples: np.ndarray, sample_rate: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(target), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm16.tobytes())
    return target
