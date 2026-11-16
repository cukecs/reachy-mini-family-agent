from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt, resample_poly

from .pitch import _frame_pitch

_TAG_RE = re.compile(r"^<\|([^|>]+)\|>$")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")

_EMOTION_ZH = {
    "HAPPY": "开心/轻快",
    "SAD": "低落/难过",
    "ANGRY": "生气/急躁",
    "NEUTRAL": "平静",
    "FEARFUL": "紧张/害怕",
    "DISGUSTED": "厌烦",
    "SURPRISED": "惊讶",
}

_EVENT_ZH = {
    "SPEECH": "正常说话",
    "LAUGHTER": "笑声",
    "APPLAUSE": "掌声",
    "CRY": "哭声",
    "COUGH": "咳嗽",
    "SNEEZE": "打喷嚏",
    "BGM": "背景音乐",
}


def clean_sense_tag(raw: str) -> str:
    value = str(raw or "").strip()
    match = _TAG_RE.match(value)
    if match:
        value = match.group(1).strip()
    return value


def normalize_emotion(raw: str) -> str:
    tag = clean_sense_tag(raw).upper()
    if not tag or tag in {"EMO_UNKNOWN", "UNKNOWN", "NONE"}:
        return "neutral"
    return tag.lower()


def normalize_event(raw: str) -> str:
    tag = clean_sense_tag(raw).upper()
    if not tag or tag in {"EVENT_UNKNOWN", "UNKNOWN", "NONE"}:
        return "speech"
    return tag.lower()


def normalize_lang(raw: str) -> str:
    tag = clean_sense_tag(raw).lower()
    return tag or "auto"


@dataclass(frozen=True)
class SpeechRecognitionDetail:
    text: str
    lang: str = "auto"
    emotion: str = ""
    event: str = ""
    tokens: tuple[str, ...] = ()
    timestamps: tuple[float, ...] = ()


@dataclass(frozen=True)
class SpeechAnalysis:
    text: str
    lang: str
    emotion: str
    event: str
    audio_seconds: float
    active_seconds: float
    rate_per_second: float
    rate_unit: str
    pause_count: int
    longest_pause_seconds: float
    pause_after_tokens: tuple[str, ...]
    mean_pitch_hz: float
    pitch_range_semitones: float
    ending_contour: str

    @property
    def has_notable_prosody(self) -> bool:
        return (
            self.emotion != "neutral"
            or self.event != "speech"
            or self.pause_count > 0
            or self.ending_contour in {"rising", "falling"}
            or bool(_EN_WORD_RE.search(self.text))
        )

    def model_context(self) -> str:
        if not self.text.strip():
            return ""
        parts: list[str] = []
        emo_zh = _EMOTION_ZH.get(self.emotion.upper(), self.emotion)
        if self.emotion and self.emotion != "neutral":
            parts.append(f"说话语气={emo_zh}")
        else:
            parts.append("说话语气=平静")
        if self.event and self.event != "speech":
            parts.append(f"伴随声音={_EVENT_ZH.get(self.event.upper(), self.event)}")
        if self.rate_per_second > 0:
            parts.append(f"语速约{self.rate_per_second:.1f}{self.rate_unit}")
        if self.pause_count > 0:
            after = "、".join(self.pause_after_tokens[:3])
            pause_desc = (
                f"句中明显停顿{self.pause_count}次（最长{self.longest_pause_seconds:.2f}秒"
            )
            if after:
                pause_desc += f"，多出现在“{after}”后"
            pause_desc += "）"
            parts.append(pause_desc)
        if self.mean_pitch_hz > 0:
            contour_zh = {
                "rising": "句尾语调上扬",
                "falling": "句尾语调下降",
                "level": "句尾语调较平",
            }.get(self.ending_contour, "")
            pitch_desc = (
                f"基频均值{self.mean_pitch_hz:.0f}Hz/音域{self.pitch_range_semitones:.1f}半音"
            )
            if contour_zh:
                pitch_desc += f"（{contour_zh}）"
            parts.append(pitch_desc)
        return "[本地语音声学特征（来自麦克风原声，非纯文本猜测）：" + "；".join(parts) + "。]"

    def display_summary(self) -> str:
        emo_zh = _EMOTION_ZH.get(self.emotion.upper(), self.emotion)
        contour_zh = {
            "rising": "尾音上扬",
            "falling": "尾音下降",
            "level": "尾音平直",
        }.get(self.ending_contour, "语调平稳")
        return (
            f"语气：{emo_zh}｜语速：{self.rate_per_second:.1f}{self.rate_unit}"
            f"｜停顿：{self.pause_count}次｜{contour_zh}"
        )


def _analyze_intonation(samples: np.ndarray, sample_rate: int) -> tuple[float, float, str]:
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    mono = np.nan_to_num(mono.reshape(-1), copy=False)
    if sample_rate <= 0 or mono.size < int(sample_rate * 0.25):
        return 0.0, 0.0, "unknown"
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        mono = resample_poly(
            mono, 16000 // divisor, sample_rate // divisor
        ).astype(np.float32)
        sample_rate = 16000

    frame_size = 640  # 40ms at 16kHz
    hop_size = 160    # 10ms
    if mono.size < frame_size:
        return 0.0, 0.0, "unknown"

    pitches: list[float] = []
    for offset in range(0, len(mono) - frame_size + 1, hop_size):
        frame = mono[offset : offset + frame_size]
        rms = float(np.sqrt(np.mean(np.square(frame))))
        if rms < 0.008:
            continue
        freq, conf = _frame_pitch(frame, sample_rate, 75.0, 420.0)
        if freq > 0.0 and conf >= 0.55:
            pitches.append(freq)

    if len(pitches) < 5:
        return 0.0, 0.0, "unknown"

    smoothed = medfilt(np.asarray(pitches, dtype=np.float64), kernel_size=5 if len(pitches) >= 5 else 3)
    mean_hz = float(np.median(smoothed))
    low_hz = float(np.percentile(smoothed, 10))
    high_hz = float(np.percentile(smoothed, 90))
    range_semi = 12.0 * math.log2(max(high_hz, 1e-6) / max(low_hz, 1e-6))

    tail_count = max(3, len(smoothed) // 4)
    head_median = float(np.median(smoothed[:-tail_count])) if len(smoothed) > tail_count else mean_hz
    tail_median = float(np.median(smoothed[-tail_count:]))
    semitone_shift = 12.0 * math.log2(max(tail_median, 1e-6) / max(head_median, 1e-6))
    if semitone_shift >= 1.4:
        contour = "rising"
    elif semitone_shift <= -1.4:
        contour = "falling"
    else:
        contour = "level"
    return round(mean_hz, 1), round(range_semi, 1), contour


def analyze_speech(
    samples: np.ndarray,
    sample_rate: int,
    detail: SpeechRecognitionDetail,
) -> SpeechAnalysis:
    """Compute speaker emotion, speech rate, pauses, and pitch contour."""
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    audio_seconds = float(mono.size / sample_rate) if sample_rate > 0 else 0.0

    lang = normalize_lang(detail.lang)
    emotion = normalize_emotion(detail.emotion)
    event = normalize_event(detail.event)

    tokens = detail.tokens
    timestamps = detail.timestamps
    pause_count = 0
    longest_pause = 0.0
    pause_after: list[str] = []
    if len(timestamps) >= 2:
        active_seconds = max(0.25, float(timestamps[-1] - timestamps[0]))
        for idx in range(len(timestamps) - 1):
            gap = float(timestamps[idx + 1] - timestamps[idx])
            if gap >= 0.45:
                pause_count += 1
                longest_pause = max(longest_pause, gap)
                if idx < len(tokens):
                    tok = tokens[idx].strip()
                    if tok:
                        pause_after.append(tok)
    else:
        active_seconds = max(0.25, audio_seconds)

    en_words = _EN_WORD_RE.findall(detail.text)
    cjk_chars = _CJK_CHAR_RE.findall(detail.text)
    if lang == "en" or (len(en_words) >= 2 and len(en_words) >= len(cjk_chars)):
        rate_per_second = len(en_words) / active_seconds
        rate_unit = "词/秒"
    else:
        total_units = len(cjk_chars) + len(en_words)
        rate_per_second = total_units / active_seconds
        rate_unit = "字/秒"

    mean_hz, range_semi, contour = _analyze_intonation(mono, sample_rate)
    return SpeechAnalysis(
        text=detail.text,
        lang=lang,
        emotion=emotion,
        event=event,
        audio_seconds=round(audio_seconds, 2),
        active_seconds=round(active_seconds, 2),
        rate_per_second=round(rate_per_second, 2),
        rate_unit=rate_unit,
        pause_count=pause_count,
        longest_pause_seconds=round(longest_pause, 2),
        pause_after_tokens=tuple(pause_after[:4]),
        mean_pitch_hz=mean_hz,
        pitch_range_semitones=range_semi,
        ending_contour=contour,
    )
