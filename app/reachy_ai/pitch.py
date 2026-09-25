from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import correlate, find_peaks, medfilt, resample_poly


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEFAULT_STATE_PATH = Path(r"D:\ReachyAI\data\last-pitch-analysis.json")


@dataclass(frozen=True)
class PitchNote:
    name: str
    midi: int
    frequency_hz: float
    duration_seconds: float
    cents_offset: float


@dataclass(frozen=True)
class PitchAnalysis:
    total_seconds: float
    voiced_seconds: float
    confidence: float
    notes: tuple[PitchNote, ...]

    @property
    def reliable(self) -> bool:
        return self.voiced_seconds >= 0.45 and bool(self.notes) and self.confidence >= 0.50

    @property
    def lowest_note(self) -> str | None:
        return min(self.notes, key=lambda item: item.midi).name if self.notes else None

    @property
    def highest_note(self) -> str | None:
        return max(self.notes, key=lambda item: item.midi).name if self.notes else None

    def model_context(self) -> str:
        if not self.reliable:
            return "[本地音高分析：没有检测到足够稳定的歌声音高。]"
        sequence = "、".join(
            f"{note.name}({note.duration_seconds:.2f}秒,{note.cents_offset:+.0f}音分)"
            for note in self.notes[:24]
        )
        return (
            "[本地歌声音高分析；这是麦克风信号分析结果，不是语音识别猜测："
            f"稳定有声音高约{self.voiced_seconds:.2f}秒；音域{self.lowest_note}到{self.highest_note}；"
            f"音符序列{sequence}；整体置信度{self.confidence:.2f}。"
            "请根据这些音高事实回答，不要声称知道未提供的原曲标准旋律。]"
        )

    def display_summary(self) -> str:
        if not self.reliable:
            return "🎤 没有检测到足够稳定的音高"
        sequence = " → ".join(note.name for note in self.notes[:16])
        return (
            f"🎤 音高：{sequence}｜音域 {self.lowest_note}–{self.highest_note}｜"
            f"有效 {self.voiced_seconds:.1f} 秒"
        )

    def write_json(self, path: Path = DEFAULT_STATE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_seconds": self.total_seconds,
            "voiced_seconds": self.voiced_seconds,
            "confidence": self.confidence,
            "reliable": self.reliable,
            "lowest_note": self.lowest_note,
            "highest_note": self.highest_note,
            "notes": [asdict(note) for note in self.notes],
        }
        temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


def _note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _frame_pitch(frame: np.ndarray, sample_rate: int, minimum_hz: float, maximum_hz: float) -> tuple[float, float]:
    frame = np.asarray(frame, dtype=np.float64)
    frame = (frame - np.mean(frame)) * np.hanning(len(frame))
    energy = np.square(frame)
    if float(np.sum(energy)) <= 1e-10:
        return 0.0, 0.0
    minimum_lag = max(2, int(sample_rate / maximum_hz))
    maximum_lag = min(len(frame) // 2, int(sample_rate / minimum_hz))
    autocorrelation = correlate(frame, frame, mode="full", method="fft")[len(frame) - 1 :]
    prefix = np.concatenate(([0.0], np.cumsum(energy)))
    lags = np.arange(minimum_lag, maximum_lag + 1)
    left_energy = prefix[len(frame) - lags]
    right_energy = prefix[len(frame)] - prefix[lags]
    denominator = np.sqrt(np.maximum(left_energy * right_energy, 1e-12))
    normalized = autocorrelation[lags] / denominator
    peaks, _ = find_peaks(normalized)
    if not len(peaks):
        return 0.0, 0.0
    peak_values = normalized[peaks]
    strongest = float(np.max(peak_values))
    eligible = peaks[peak_values >= max(0.50, strongest * 0.90)]
    if not len(eligible):
        return 0.0, strongest
    relative_index = int(eligible[0])
    lag = float(lags[relative_index])
    if 0 < relative_index < len(normalized) - 1:
        left, center, right = normalized[relative_index - 1 : relative_index + 2]
        curvature = left - 2.0 * center + right
        if abs(curvature) > 1e-9:
            lag += float(0.5 * (left - right) / curvature)
    return float(sample_rate / max(lag, 1.0)), strongest


def analyze_pitch(
    samples: np.ndarray,
    sample_rate: int,
    *,
    minimum_hz: float = 65.0,
    maximum_hz: float = 1000.0,
) -> PitchAnalysis:
    """Extract a monophonic singing contour locally from microphone PCM."""
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    mono = np.nan_to_num(mono.reshape(-1), copy=False)
    total_seconds = len(mono) / sample_rate if sample_rate > 0 else 0.0
    if sample_rate <= 0 or len(mono) < int(sample_rate * 0.20):
        return PitchAnalysis(total_seconds, 0.0, 0.0, ())
    target_rate = 16000
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = target_rate
    mono = np.clip(mono, -1.0, 1.0)
    frame_length = 2048
    hop_length = 320
    if len(mono) < frame_length:
        mono = np.pad(mono, (0, frame_length - len(mono)))
    starts = range(0, len(mono) - frame_length + 1, hop_length)
    frames = [mono[start : start + frame_length] for start in starts]
    rms = np.asarray([float(np.sqrt(np.mean(np.square(frame)))) for frame in frames])
    upper_rms = float(np.percentile(rms, 75)) if len(rms) else 0.0
    threshold = max(0.004, float(np.percentile(rms, 15)) * 2.2)
    if upper_rms > 0:
        threshold = min(threshold, upper_rms * 0.55)

    frequencies = np.full(len(frames), np.nan, dtype=np.float64)
    confidences = np.zeros(len(frames), dtype=np.float64)
    for index, frame in enumerate(frames):
        if rms[index] < threshold:
            continue
        frequency, confidence = _frame_pitch(frame, sample_rate, minimum_hz, maximum_hz)
        if confidence >= 0.50 and minimum_hz <= frequency <= maximum_hz:
            frequencies[index] = frequency
            confidences[index] = confidence

    valid = np.isfinite(frequencies)
    if not np.any(valid):
        return PitchAnalysis(total_seconds, 0.0, 0.0, ())
    midi = np.full_like(frequencies, np.nan)
    midi[valid] = 69.0 + 12.0 * np.log2(frequencies[valid] / 440.0)
    valid_values = midi[valid]
    if len(valid_values) >= 3:
        midi[valid] = medfilt(valid_values, kernel_size=3)

    events: list[PitchNote] = []
    current: list[float] = []
    current_confidences: list[float] = []
    gap_frames = 0

    def finish_event() -> None:
        nonlocal current, current_confidences, gap_frames
        duration = len(current) * hop_length / sample_rate
        if current and duration >= 0.12:
            median_midi = float(np.median(current))
            rounded = int(round(median_midi))
            events.append(
                PitchNote(
                    name=_note_name(rounded),
                    midi=rounded,
                    frequency_hz=float(440.0 * (2.0 ** ((median_midi - 69.0) / 12.0))),
                    duration_seconds=round(duration, 3),
                    cents_offset=round((median_midi - rounded) * 100.0, 1),
                )
            )
        current = []
        current_confidences = []
        gap_frames = 0

    for value, confidence in zip(midi, confidences):
        if not np.isfinite(value):
            gap_frames += 1
            if gap_frames > 2:
                finish_event()
            continue
        gap_frames = 0
        if current and abs(value - float(np.median(current[-8:]))) > 0.75:
            finish_event()
        current.append(float(value))
        current_confidences.append(float(confidence))
    finish_event()

    voiced_seconds = float(np.count_nonzero(valid) * hop_length / sample_rate)
    confidence = float(np.median(confidences[valid]))
    return PitchAnalysis(
        total_seconds=round(total_seconds, 3),
        voiced_seconds=round(voiced_seconds, 3),
        confidence=round(confidence, 3),
        notes=tuple(events),
    )
