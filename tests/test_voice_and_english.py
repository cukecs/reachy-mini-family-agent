from __future__ import annotations

import unittest

import numpy as np

from reachy_ai.english_pronunciation import (
    EnglishPronunciationAssessor,
    _ctc_forced_align,
    arpabet_to_ipa,
)
from reachy_ai.voice_analysis import SpeechRecognitionDetail, analyze_speech


class VoiceAndEnglishTests(unittest.TestCase):
    def test_analyze_speech_detects_emotion_pauses_and_rising_contour(self) -> None:
        sr = 16000
        duration = 1.2
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        freq = np.linspace(140.0, 240.0, len(t), dtype=np.float32)
        phase = 2.0 * np.pi * np.cumsum(freq) / sr
        audio = 0.25 * np.sin(phase).astype(np.float32)

        detail = SpeechRecognitionDetail(
            text="Do you think so?",
            lang="<|en|>",
            emotion="<|HAPPY|>",
            event="<|Speech|>",
            tokens=("Do", "you", "think", "so?"),
            timestamps=(0.10, 0.28, 0.85, 1.05),
        )
        analysis = analyze_speech(audio, sr, detail)
        self.assertEqual(analysis.lang, "en")
        self.assertEqual(analysis.emotion, "happy")
        self.assertEqual(analysis.pause_count, 1)
        self.assertEqual(analysis.pause_after_tokens, ("you",))
        self.assertEqual(analysis.ending_contour, "rising")
        self.assertIn("开心", analysis.model_context())

    def test_arpabet_to_ipa_and_forced_align(self) -> None:
        ipa = arpabet_to_ipa(["th", "ih", "ng", "k"])
        self.assertEqual(ipa, ["θ", "ɪ", "ŋ", "k"])

        # Synthetic log-probs peaking at blank(0), phoneA(1), blank(0), phoneB(2)
        log_probs = np.full((8, 3), -4.0, dtype=np.float64)
        log_probs[0, 0] = -0.1
        log_probs[1:3, 1] = -0.1
        log_probs[3:5, 0] = -0.1
        log_probs[5:8, 2] = -0.1

        spans = _ctc_forced_align(log_probs, [1, 2], blank_id=0)
        self.assertEqual(len(spans), 2)
        self.assertLess(spans[0][0], spans[0][1])
        self.assertLessEqual(spans[0][1], spans[1][0])

    def test_english_assessor_fallback_and_mocked_onnx(self) -> None:
        assessor = EnglishPronunciationAssessor()
        audio = np.zeros(16000, dtype=np.float32)

        fallback = assessor.assess(
            audio,
            16000,
            heard_text="I sink so",
            target_text="I think so",
        )
        self.assertEqual(fallback.backend, "asr-lexicon-fallback")
        self.assertIn("think", [w.word for w in fallback.word_assessments])
        think_item = next(w for w in fallback.word_assessments if w.word == "think")
        self.assertIn("sink", think_item.issue)

        # Mock ONNX logits where "think" (/θ ɪ ŋ k/) gets /s/ instead of /θ/.
        assessor._vocab = {"<pad>": 0, "θ": 1, "s": 2, "ɪ": 3, "ŋ": 4, "k": 5}
        assessor._inv_vocab = {v: k for k, v in assessor._vocab.items()}
        log_probs = np.full((12, 6), -5.0, dtype=np.float64)
        log_probs[0, 0] = -0.05
        log_probs[1:3, 2] = -0.05  # heard /s/ instead of /θ/
        log_probs[3:6, 3] = -0.05  # /ɪ/
        log_probs[6:9, 4] = -0.05  # /ŋ/
        log_probs[9:12, 5] = -0.05  # /k/

        assessor._lexicon = {"think": ["th", "ih", "ng", "k"]}
        assessor._run_onnx = lambda *_args, **_kwargs: (log_probs, "s ɪ ŋ k")  # type: ignore[method-assign]
        onnx_res = assessor.assess(
            audio,
            16000,
            heard_text="think",
            target_text="think",
        )
        self.assertEqual(onnx_res.backend, "onnx-wav2vec2-ipa")
        self.assertEqual(len(onnx_res.word_assessments), 1)
        self.assertIn("/θ/听起来像/s/", onnx_res.word_assessments[0].issue)


if __name__ == "__main__":
    unittest.main()
