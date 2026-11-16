from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .voice_analysis import SpeechAnalysis

DEFAULT_PHONEME_MODEL_DIR = Path(r"D:\ReachyAI\models\speech\english-phoneme")
DEFAULT_LEXICON_PATH = Path(r"D:\ReachyAI\models\speech\tts\vits-melo-tts-zh_en\lexicon.txt")

_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Map MeloTTS / CMU ARPAbet phones to eSpeak IPA symbols used by wav2vec2-espeak.
_ARPABET_TO_IPA: dict[str, str] = {
    "aa": "ɑː",
    "ae": "æ",
    "ah": "ʌ",
    "ah0": "ə",
    "ao": "ɔ",
    "aw": "aʊ",
    "ax": "ə",
    "ay": "aɪ",
    "b": "b",
    "ch": "tʃ",
    "d": "d",
    "dh": "ð",
    "eh": "ɛ",
    "er": "ɚ",
    "ey": "eɪ",
    "f": "f",
    "g": "ɡ",
    "hh": "h",
    "ih": "ɪ",
    "iy": "iː",
    "jh": "dʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "ng": "ŋ",
    "ow": "oʊ",
    "oy": "ɔɪ",
    "p": "p",
    "r": "ɹ",
    "s": "s",
    "sh": "ʃ",
    "t": "t",
    "th": "θ",
    "uh": "ʊ",
    "uw": "uː",
    "v": "v",
    "w": "w",
    "y": "j",
    "z": "z",
    "zh": "ʒ",
}

_IPA_FALLBACKS: dict[str, tuple[str, ...]] = {
    "ɑː": ("ɑ", "a"),
    "iː": ("i", "ɪ"),
    "uː": ("u", "ʊ"),
    "ɚ": ("ɜː", "ɜ", "ə", "ɹ"),
    "oʊ": ("oː", "o"),
    "ɔɪ": ("ɔ",),
    "ɡ": ("g",),
    "ɹ": ("r",),
}

# Natural stress reductions and broad/narrow transcription equivalents
# (e.g. MeloTTS uses /ʌ/ for unstressed schwa /ə/, and /ɔ/ for /ɔː/).
_ACCEPTABLE_IPA_VARIANTS: dict[str, tuple[str, ...]] = {
    "ʌ": ("ə", "ɐ", "əl"),
    "ə": ("ʌ", "ɐ", "ᵻ", "ɪ"),
    "ɔ": ("ɔː", "ɑː", "ɑ"),
    "ɑː": ("ɑ", "a", "ɔː"),
    "iː": ("i",),
    "uː": ("u",),
    "ɚ": ("ɜː", "ɜ", "ɹ", "ə"),
    "eɪ": ("eː", "e"),
    "oʊ": ("oː", "o"),
    "ɪ": ("ᵻ", "i"),
    "t": ("ɾ",),
    "d": ("ɾ",),
}


@dataclass(frozen=True)
class PhonemeAssessment:
    expected_ipa: str
    heard_ipa: str
    score: float
    accepted: bool


@dataclass(frozen=True)
class WordPronunciationAssessment:
    word: str
    expected_ipa: str
    score: float
    issue: str = ""
    phonemes: tuple[PhonemeAssessment, ...] = ()


@dataclass(frozen=True)
class EnglishPronunciationResult:
    backend: str
    target_text: str
    heard_text: str
    overall_score: float
    heard_ipa: str
    word_assessments: tuple[WordPronunciationAssessment, ...]
    speech_analysis: SpeechAnalysis | None = None

    @property
    def has_phoneme_model(self) -> bool:
        return self.backend.startswith("onnx-")

    def model_context(self) -> str:
        lines = [
            f"[本地英语发音评测（引擎：{self.backend}；请仅依据以下实测结果点评，不要编造未测出的细节）：",
            f"- 参照文本：{self.target_text or '（自由口语）'}",
            f"- 本地听写结果：{self.heard_text or '（未识别出有效英文）'}",
        ]
        if self.has_phoneme_model:
            lines.append(f"- 综合发音准确度：{self.overall_score:.0f}/100")
            if self.heard_ipa:
                lines.append(f"- 实测音素流（IPA）：/{self.heard_ipa}/")
            weak_words = [w for w in self.word_assessments if w.issue or w.score < 68.0]
            if weak_words:
                details = "；".join(
                    f"{w.word}（目标/{w.expected_ipa}/，得分{w.score:.0f}"
                    + (f"，{w.issue}" if w.issue else "")
                    + "）"
                    for w in weak_words[:6]
                )
                lines.append(f"- 需重点纠正单词：{details}")
            else:
                lines.append("- 各单词核心音素均基本到位。")
        else:
            lines.append(
                "- 说明：当前未安装 ONNX 音素评测模型，仅基于本地 ASR 对比与词典音标分析；"
                "若本地听写与目标词不一致或有明显卡顿，请温和指出可能混淆的读音，并可提醒用户运行 scripts/prepare-english-phoneme.py 启用音素级打分。"
            )
            weak_words = [w for w in self.word_assessments if w.issue]
            if weak_words:
                details = "；".join(
                    f"{w.word}（参考/{w.expected_ipa}/：{w.issue}）"
                    for w in weak_words[:6]
                )
                lines.append(f"- 词级差异与参考音标：{details}")
        if self.speech_analysis is not None:
            lines.append(f"- 语音节奏与语调：{self.speech_analysis.display_summary()}")
        lines.append("请用简短友好的中文指出具体发音/重音/语调问题，给出易懂的口型或对比提示，并领读一遍。]")
        return "\n".join(lines)

    def display_summary(self) -> str:
        weak = [w.word for w in self.word_assessments if w.issue or w.score < 68.0]
        if self.has_phoneme_model:
            note = f"需注意：{', '.join(weak[:4])}" if weak else "发音清晰"
            return f"🗣️ 英语发音 {self.overall_score:.0f}/100｜{note}"
        note = f"疑似差异：{', '.join(weak[:4])}" if weak else "词形匹配正常"
        return f"🗣️ 英语口语分析（基础模式）｜{note}"


def extract_english_words(text: str) -> list[str]:
    return [word for word in _EN_WORD_RE.findall(text) if word]


def load_english_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> dict[str, list[str]]:
    """Load English word -> ARPAbet phone list from MeloTTS lexicon."""
    mapping: dict[str, list[str]] = {}
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        word = parts[0].lower()
        if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", word):
            continue
        if word in mapping:
            continue
        phones: list[str] = []
        for token in parts[1:]:
            if token.isdigit():
                break
            phones.append(token.lower())
        if phones:
            mapping[word] = phones
    return mapping


def arpabet_to_ipa(phones: list[str]) -> list[str]:
    result: list[str] = []
    for phone in phones:
        base = re.sub(r"[012]$", "", phone.lower())
        ipa = _ARPABET_TO_IPA.get(base)
        if ipa:
            result.append(ipa)
    return result


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _ctc_forced_align(
    log_probs: np.ndarray,
    target_ids: list[int],
    blank_id: int = 0,
) -> list[tuple[int, int]]:
    """Return (start_frame, end_frame) per target token via Viterbi CTC alignment."""
    num_frames = log_probs.shape[0]
    if not target_ids or num_frames == 0:
        return []
    # Blank-interleaved state sequence: [b, t0, b, t1, b, ..., b]
    states: list[int] = [blank_id]
    for tid in target_ids:
        states.append(tid)
        states.append(blank_id)
    num_states = len(states)
    if num_frames < len(target_ids):
        return [(0, num_frames) for _ in target_ids]

    neg_inf = -1e9
    dp = np.full((num_frames, num_states), neg_inf, dtype=np.float64)
    back = np.full((num_frames, num_states), -1, dtype=np.int32)

    dp[0, 0] = log_probs[0, states[0]]
    if num_states > 1:
        dp[0, 1] = log_probs[0, states[1]]

    for t in range(1, num_frames):
        frame_lp = log_probs[t]
        for s in range(num_states):
            best_prev = dp[t - 1, s]
            best_idx = s
            if s > 0 and dp[t - 1, s - 1] > best_prev:
                best_prev = dp[t - 1, s - 1]
                best_idx = s - 1
            if (
                s > 1
                and states[s] != blank_id
                and states[s] != states[s - 2]
                and dp[t - 1, s - 2] > best_prev
            ):
                best_prev = dp[t - 1, s - 2]
                best_idx = s - 2
            dp[t, s] = best_prev + float(frame_lp[states[s]])
            back[t, s] = best_idx

    last_s = num_states - 1
    if num_states >= 2 and dp[num_frames - 1, num_states - 2] > dp[num_frames - 1, last_s]:
        last_s = num_states - 2

    path = [0] * num_frames
    cur = last_s
    for t in range(num_frames - 1, -1, -1):
        path[t] = cur
        if t > 0:
            cur = int(back[t, cur])
            if cur < 0:
                cur = 0

    spans: list[tuple[int, int]] = []
    for idx in range(len(target_ids)):
        target_state = 2 * idx + 1
        frames = [t for t, st in enumerate(path) if st == target_state]
        if frames:
            spans.append((frames[0], frames[-1] + 1))
        else:
            approx = min(num_frames - 1, int(idx * num_frames / max(1, len(target_ids))))
            spans.append((approx, approx + 1))
    return spans


class EnglishPronunciationAssessor:
    """CPU-friendly English pronunciation evaluator with optional ONNX Wav2Vec2 IPA."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_PHONEME_MODEL_DIR,
        lexicon_path: Path = DEFAULT_LEXICON_PATH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.lexicon_path = Path(lexicon_path)
        self._lexicon: dict[str, list[str]] | None = None
        self._session: Any = None
        self._vocab: dict[str, int] | None = None
        self._inv_vocab: dict[int, str] | None = None
        self._load_attempted = False

    @property
    def lexicon(self) -> dict[str, list[str]]:
        if self._lexicon is None:
            self._lexicon = load_english_lexicon(self.lexicon_path)
        return self._lexicon

    def _ensure_onnx_loaded(self) -> bool:
        if self._load_attempted:
            return self._session is not None and self._vocab is not None
        self._load_attempted = True
        model_path = next(
            (
                candidate
                for candidate in (
                    self.model_dir / "model_int8.onnx",
                    self.model_dir / "model.onnx",
                )
                if candidate.exists()
            ),
            None,
        )
        vocab_path = self.model_dir / "vocab.json"
        if model_path is None or not vocab_path.exists():
            return False
        try:
            import onnxruntime as ort

            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            if not isinstance(vocab, dict) or not vocab:
                return False
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._vocab = {str(k): int(v) for k, v in vocab.items()}
            self._inv_vocab = {v: k for k, v in self._vocab.items()}
            return True
        except Exception:
            self._session = None
            self._vocab = None
            self._inv_vocab = None
            return False

    def _resolve_vocab_id(self, ipa_symbol: str) -> int | None:
        if not self._vocab:
            return None
        if ipa_symbol in self._vocab:
            return self._vocab[ipa_symbol]
        for alt in _IPA_FALLBACKS.get(ipa_symbol, ()):
            if alt in self._vocab:
                return self._vocab[alt]
        return None

    def word_ipa(self, word: str) -> list[str]:
        phones = self.lexicon.get(word.lower(), [])
        return arpabet_to_ipa(phones)

    def _run_onnx(self, samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, str] | None:
        if not self._ensure_onnx_loaded() or self._session is None or self._inv_vocab is None:
            return None
        mono = np.asarray(samples, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        mono = np.nan_to_num(mono.reshape(-1), copy=False)
        if sample_rate <= 0 or mono.size < int(sample_rate * 0.15):
            return None
        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            mono = resample_poly(
                mono, 16000 // divisor, sample_rate // divisor
            ).astype(np.float32)
        mean = float(np.mean(mono))
        std = float(np.std(mono))
        if std < 1e-6:
            return None
        normed = ((mono - mean) / math.sqrt(std * std + 1e-7)).astype(np.float32)[None, :]
        input_name = self._session.get_inputs()[0].name
        inputs: dict[str, Any] = {input_name: normed}
        for inp in self._session.get_inputs()[1:]:
            if "attention" in inp.name:
                inputs[inp.name] = np.ones_like(normed, dtype=np.int64)
        logits = self._session.run(None, inputs)[0][0]
        log_probs = _log_softmax(np.asarray(logits, dtype=np.float64))

        # Greedy CTC collapse for heard IPA string
        blank_id = self._vocab.get("<pad>", 0) if self._vocab else 0
        special_ids = {
            blank_id,
            self._vocab.get("<s>", -1) if self._vocab else -1,
            self._vocab.get("</s>", -1) if self._vocab else -1,
            self._vocab.get("<unk>", -1) if self._vocab else -1,
        }
        pred_ids = np.argmax(log_probs, axis=-1)
        heard_tokens: list[str] = []
        prev = -1
        for pid in pred_ids:
            idx = int(pid)
            if idx != prev and idx not in special_ids:
                sym = self._inv_vocab.get(idx, "")
                if sym:
                    heard_tokens.append(sym)
            prev = idx
        return log_probs, " ".join(heard_tokens)

    def assess(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        heard_text: str,
        target_text: str | None = None,
        speech_analysis: SpeechAnalysis | None = None,
    ) -> EnglishPronunciationResult:
        reference_text = (target_text or heard_text).strip()
        target_words = extract_english_words(reference_text)
        heard_words = extract_english_words(heard_text)

        onnx_out = self._run_onnx(samples, sample_rate)
        if onnx_out is not None and target_words:
            log_probs, heard_ipa = onnx_out
            return self._assess_with_onnx(
                log_probs=log_probs,
                heard_ipa=heard_ipa,
                target_words=target_words,
                reference_text=reference_text,
                heard_text=heard_text,
                speech_analysis=speech_analysis,
            )

        return self._assess_fallback(
            target_words=target_words,
            heard_words=heard_words,
            reference_text=reference_text,
            heard_text=heard_text,
            speech_analysis=speech_analysis,
        )

    def _assess_with_onnx(
        self,
        *,
        log_probs: np.ndarray,
        heard_ipa: str,
        target_words: list[str],
        reference_text: str,
        heard_text: str,
        speech_analysis: SpeechAnalysis | None,
    ) -> EnglishPronunciationResult:
        assert self._vocab is not None and self._inv_vocab is not None
        blank_id = self._vocab.get("<pad>", 0)
        word_phone_seqs: list[tuple[str, list[tuple[str, int]]]] = []
        flat_ids: list[int] = []

        for word in target_words:
            ipa_list = self.word_ipa(word)
            pairs: list[tuple[str, int]] = []
            for sym in ipa_list:
                tid = self._resolve_vocab_id(sym)
                if tid is not None:
                    pairs.append((sym, tid))
                    flat_ids.append(tid)
            word_phone_seqs.append((word, pairs))

        if not flat_ids:
            return self._assess_fallback(
                target_words=target_words,
                heard_words=extract_english_words(heard_text),
                reference_text=reference_text,
                heard_text=heard_text,
                speech_analysis=speech_analysis,
            )

        spans = _ctc_forced_align(log_probs, flat_ids, blank_id=blank_id)
        span_cursor = 0
        word_results: list[WordPronunciationAssessment] = []
        all_scores: list[float] = []

        for word, pairs in word_phone_seqs:
            if not pairs:
                word_results.append(
                    WordPronunciationAssessment(word=word, expected_ipa="", score=75.0)
                )
                continue
            phone_results: list[PhonemeAssessment] = []
            weak_notes: list[str] = []
            for sym, tid in pairs:
                start_f, end_f = spans[span_cursor]
                span_cursor += 1
                seg = log_probs[start_f:end_f]
                if seg.size == 0:
                    score = 50.0
                    heard_sym = "?"
                else:
                    max_lp = np.max(seg, axis=-1)
                    # Find strongest non-blank phoneme in this segment
                    seg_no_blank = seg.copy()
                    seg_no_blank[:, blank_id] = -1e9
                    best_id = int(np.argmax(np.mean(seg_no_blank, axis=0)))
                    heard_sym = self._inv_vocab.get(best_id, sym)
                    allowed_ids = [tid]
                    for variant in _ACCEPTABLE_IPA_VARIANTS.get(sym, ()):
                        vid = self._vocab.get(variant)
                        if vid is not None:
                            allowed_ids.append(vid)
                    best_target_lp = np.max(seg[:, allowed_ids], axis=-1)
                    gop = float(np.mean(best_target_lp - max_lp))
                    score = round(100.0 * math.exp(max(-6.0, gop) * 0.45), 1)
                is_variant_match = heard_sym in _ACCEPTABLE_IPA_VARIANTS.get(sym, ())
                accepted = score >= 62.0 or (is_variant_match and score >= 45.0)
                phone_results.append(
                    PhonemeAssessment(
                        expected_ipa=sym,
                        heard_ipa=heard_sym,
                        score=score,
                        accepted=accepted,
                    )
                )
                if not accepted and heard_sym != sym and not is_variant_match:
                    weak_notes.append(f"/{sym}/听起来像/{heard_sym}/")
                elif not accepted:
                    weak_notes.append(f"/{sym}/偏弱")

            word_score = round(sum(p.score for p in phone_results) / len(phone_results), 1)
            all_scores.append(word_score)
            expected_ipa = "".join(sym for sym, _ in pairs)
            word_results.append(
                WordPronunciationAssessment(
                    word=word,
                    expected_ipa=expected_ipa,
                    score=word_score,
                    issue="、".join(weak_notes[:2]),
                    phonemes=tuple(phone_results),
                )
            )

        overall = round(sum(all_scores) / max(1, len(all_scores)), 1)
        return EnglishPronunciationResult(
            backend="onnx-wav2vec2-ipa",
            target_text=reference_text,
            heard_text=heard_text,
            overall_score=overall,
            heard_ipa=heard_ipa,
            word_assessments=tuple(word_results),
            speech_analysis=speech_analysis,
        )

    def _assess_fallback(
        self,
        *,
        target_words: list[str],
        heard_words: list[str],
        reference_text: str,
        heard_text: str,
        speech_analysis: SpeechAnalysis | None,
    ) -> EnglishPronunciationResult:
        lower_target = [w.lower() for w in target_words]
        lower_heard = [w.lower() for w in heard_words]
        matcher = difflib.SequenceMatcher(None, lower_target, lower_heard)
        issues_by_index: dict[int, str] = {}
        matched = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                matched += i2 - i1
            elif tag == "replace":
                heard_slice = " ".join(heard_words[j1:j2])
                for idx in range(i1, i2):
                    issues_by_index[idx] = f"被识别为“{heard_slice}”"
            elif tag == "delete":
                for idx in range(i1, i2):
                    issues_by_index[idx] = "可能漏读或含糊"

        word_results: list[WordPronunciationAssessment] = []
        for idx, word in enumerate(target_words):
            ipa = "".join(self.word_ipa(word))
            issue = issues_by_index.get(idx, "")
            score = 55.0 if issue else 82.0
            word_results.append(
                WordPronunciationAssessment(
                    word=word,
                    expected_ipa=ipa,
                    score=score,
                    issue=issue,
                )
            )
        overall = (
            round(100.0 * matched / len(target_words), 1)
            if target_words
            else 0.0
        )
        return EnglishPronunciationResult(
            backend="asr-lexicon-fallback",
            target_text=reference_text,
            heard_text=heard_text,
            overall_score=overall,
            heard_ipa="",
            word_assessments=tuple(word_results),
            speech_analysis=speech_analysis,
        )
