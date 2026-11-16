from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from reachy_ai.contextual_pronunciation import ContextualPronunciation
from reachy_ai.pronunciation import melo_aliases, melo_entry, prepare_melo_lexicon

TOKENS = set("b p m f d t n l g k h j q x r z c s zh ch sh y w AA EE OO a ai an ang ao e ei en eng er o ou i ia ian iang iao ie in ing iong iu ong u ua uai uan uang ui un uo v van ve vn ir i0 En E".split())
MODEL = Path(r"D:\ReachyAI\models\speech\tts\vits-melo-tts-zh_en")
G2PW = Path(r"D:\ReachyAI\models\speech\g2pw")


class ContextualPronunciationTests(unittest.TestCase):
    def make_frontend(self, *, overrides_path=Path("not-present.json")):
        return ContextualPronunciation(Path("no-model"), TOKENS, overrides_path=overrides_path)

    def test_aliases_are_unique_and_have_explicit_phonemes(self):
        aliases = melo_aliases(frozenset(TOKENS))
        self.assertEqual(len(aliases), len(set(aliases.values())))
        self.assertNotEqual(aliases["chang2"], aliases["zhang3"])
        self.assertNotEqual(aliases["chong2"], aliases["zhong4"])
        for syllable in ["chang2", "zhang3", "chong2", "zhong4", "hang2", "xing2", "hai2", "huan2"]:
            self.assertTrue(0xE000 <= ord(aliases[syllable]) < 0xF900)
            self.assertEqual(len(melo_entry("一", syllable, TOKENS).split()), 5)

    def test_english_numbers_and_punctuation_are_unchanged(self):
        frontend = self.make_frontend()
        with patch.object(frontend, "_load", return_value=False):
            self.assertEqual(frontend.prepare("Hello, 123.5!"), "Hello, 123.5!")
            result = frontend.prepare("你好Hello 123！")
        self.assertEqual(len(result), len("你好Hello 123！"))
        self.assertTrue(result.endswith("Hello 123！"))

    def test_missing_or_failed_model_reports_degraded_backend(self):
        frontend = self.make_frontend()
        with self.assertLogs("reachy_ai.contextual_pronunciation", level="WARNING"):
            result = frontend.prepare("你好")
        self.assertEqual(len(result), 2)
        self.assertEqual(result, "你好")
        self.assertEqual(frontend.backend, "dictionary-fallback")
        with patch.object(frontend, "_load", return_value=False):
            self.assertEqual(len(frontend.prepare("你好")), 2)

    def test_explicit_user_override_is_applied_without_changing_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            override = Path(temporary) / "tts-pronunciation.json"
            override.write_text(json.dumps({"重读": "zhong4 du2"}), encoding="utf-8")
            frontend = self.make_frontend(overrides_path=override)
            original = "这个词要重读。"
            with patch.object(frontend, "_load", return_value=False):
                prepared = frontend.prepare(original)
            self.assertEqual(original, "这个词要重读。")
            self.assertEqual(frontend.last_readings[4:6], ["zhong4", "du2"])
            self.assertEqual(prepared[4], frontend.aliases["zhong4"])

    def test_old_common_phrase_defaults_do_not_override_context_model(self):
        frontend = self.make_frontend()
        with patch.object(frontend, "readings", return_value=["zhong4", "du2"]):
            result = frontend.prepare("重读")
        self.assertEqual(result, frontend.aliases["zhong4"] + frontend.aliases["du2"])

    def test_real_speech_engine_passes_aliases_not_original_text(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import Mock
        from reachy_ai.speech import SpeechEngine

        engine = SpeechEngine()
        fake_tts = Mock()
        fake_tts.generate.return_value = SimpleNamespace(samples=np.zeros(4), sample_rate=44100)
        engine._tts = fake_tts
        engine._pronunciation = self.make_frontend()
        with patch.object(engine._pronunciation, "_load", return_value=False):
            samples, rate = engine.synthesize("还给我。")
        sent = fake_tts.generate.call_args.args[0]
        self.assertEqual(sent[:3], "".join(engine._pronunciation.aliases[p] for p in ["huan2", "gei3", "wo3"]))
        self.assertEqual(sent[-1], "。")
        self.assertEqual(rate, 44100)


@unittest.skipUnless(os.environ.get("RUN_TTS_MODEL_TESTS") == "1", "requires local g2pW and MeloTTS weights")
class ActualContextualModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokens = {line.split()[0] for line in (MODEL / "tokens.txt").read_text(encoding="utf-8").splitlines() if line.split()}
        cls.frontend = ContextualPronunciation(G2PW, cls.tokens)

    def test_same_characters_change_with_sentence_context(self):
        cases = [
            ("这根绳子有三米长。", {7: "chang2"}),
            ("孩子又长了三厘米。", {3: "zhang3"}),
            ("我还没有把书还给你。", {1: "hai2", 6: "huan2"}),
            ("银行行长正在行走。", {1: "hang2", 2: "hang2", 3: "zhang3", 6: "xing2"}),
        ]
        for text, expected in cases:
            readings = self.frontend.readings(text)
            self.assertEqual(self.frontend.backend, "g2pw-int8")
            self.assertEqual(len(text), len(readings))
            for index, reading in expected.items():
                with self.subTest(text=text, index=index):
                    self.assertEqual(readings[index], reading)

    def test_native_synthesis_consumes_the_selected_phones(self):
        lexicon = prepare_melo_lexicon(MODEL)
        text = "这根绳子有三米长。孩子又长了三厘米。我还没有把书还给你。"
        prepared = self.frontend.prepare(text)
        code = f"""
import sherpa_onnx
from pathlib import Path
d = Path({str(MODEL)!r})
v = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d/'model.onnx'),lexicon={str(lexicon)!r},tokens=str(d/'tokens.txt'))
m = sherpa_onnx.OfflineTtsModelConfig(vits=v,num_threads=4,debug=True)
c = sherpa_onnx.OfflineTtsConfig(model=m,rule_fsts=','.join(str(d/n) for n in ['phone.fst','date.fst','number.fst']))
a = sherpa_onnx.OfflineTts(c).generate({prepared!r})
assert len(a.samples) > a.sample_rate
print('audio-ok')
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("audio-ok", result.stdout)
        for reading in ["chang2", "zhang3", "hai2", "huan2"]:
            alias = self.frontend.aliases[reading]
            phones = " ".join(melo_entry("一", reading, self.tokens).split()[1:])
            self.assertIn(f"{alias}: {phones}", result.stderr)
        self.assertNotIn("Ignore OOV", result.stderr)


if __name__ == "__main__":
    unittest.main()
