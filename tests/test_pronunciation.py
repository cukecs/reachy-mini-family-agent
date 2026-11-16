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

from reachy_ai.pronunciation import common_phrases, melo_entry, prepare_melo_lexicon


TOKENS = set("b p m f d t n l g k h j q x r z c s zh ch sh y w AA EE OO a ai an ang ao e ei en eng er o ou i ia ian iang iao ie in ing iong iu ong u ua uai uan uang ui un uo v van ve vn ir i0 En E".split())


class PronunciationTests(unittest.TestCase):
    def test_polyphones_have_opposite_readings(self):
        cases = {
            "重新": "重新 ch ong x in 2 2 1 1",
            "重要": "重要 zh ong y ao 4 4 4 4",
            "很长": "很长 h en ch ang 3 3 2 2",
            "长高": "长高 zh ang g ao 3 3 1 1",
            "行长": "行长 h ang zh ang 2 2 3 3",
            "还要": "还要 h ai y ao 2 2 4 4",
            "还书": "还书 h uan sh u 2 2 1 1",
            "音乐": "音乐 y in y ve 1 1 4 4",
            "快乐": "快乐 k uai l e 4 4 4 4",
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(melo_entry(word, common_phrases()[word], TOKENS), expected)

    def test_all_builtin_phrases_are_valid(self):
        for word, reading in common_phrases().items():
            with self.subTest(word=word):
                row = melo_entry(word, reading, TOKENS).split()
                self.assertEqual(len(row), 1 + len(word) * 4)

    def test_special_chinese_phones(self):
        self.assertEqual(melo_entry("知识", "zhi1 shi2", TOKENS), "知识 zh ir sh ir 1 1 2 2")
        self.assertEqual(melo_entry("女子", "nv3 zi3", TOKENS), "女子 n v z i0 3 3 3 3")
        self.assertEqual(melo_entry("月夜", "yue4 ye4", TOKENS), "月夜 y ve y E 4 4 4 4")

    def test_invalid_entries_are_rejected(self):
        for word, reading in [("长", "chang"), ("长高", "zhang3"),
                              ("长", "chang9"), ("长。", "chang2"),
                              ("长", "invalid2")]:
            with self.subTest(word=word, reading=reading):
                with self.assertRaises(ValueError):
                    melo_entry(word, reading, TOKENS)

    @patch("reachy_ai.pronunciation.phrases_dict", {"长高": [["zhǎng"], ["gāo"]]})
    def test_merge_cache_override_and_original_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            original = "hello HH AH0 6 6\n重 zh ong 4 4\n长 zh ang 3 3\n"
            (model / "lexicon.txt").write_text(original, encoding="utf-8")
            (model / "tokens.txt").write_text(
                "\n".join(f"{token} {i}" for i, token in enumerate(sorted(TOKENS))), encoding="utf-8"
            )
            overrides = root / "overrides.json"
            overrides.write_text(json.dumps({"重读": "zhong4 du2"}), encoding="utf-8")
            cache = root / "cache"
            first = prepare_melo_lexicon(model, overrides_path=overrides, cache_dir=cache)
            contents = first.read_text(encoding="utf-8")
            self.assertIn("hello HH AH0 6 6\n", contents)
            self.assertIn("重读 zh ong d u 4 4 2 2\n", contents)
            self.assertIn("很重要 h en zh ong y ao 3 3 4 4 4 4\n", contents)
            self.assertIn("很长大 h en zh ang d a 3 3 3 3 4 4\n", contents)
            self.assertEqual((model / "lexicon.txt").read_text(encoding="utf-8"), original)
            modified = first.stat().st_mtime_ns
            self.assertEqual(first, prepare_melo_lexicon(model, overrides_path=overrides, cache_dir=cache))
            self.assertEqual(first.stat().st_mtime_ns, modified)
            overrides.write_text(json.dumps({"重读": "chong2 du2"}), encoding="utf-8")
            second = prepare_melo_lexicon(model, overrides_path=overrides, cache_dir=cache)
            self.assertNotEqual(first, second)
            self.assertIn("重读 ch ong d u 2 2 2 2\n", second.read_text(encoding="utf-8"))
            # The model lexicon must contain no duplicate words: native sherpa
            # keeps the first occurrence and would ignore an appended override.
            words = [line.split()[0] for line in contents.splitlines()]
            self.assertEqual(len(words), len(set(words)))

    @unittest.skipUnless(os.environ.get("RUN_TTS_MODEL_TESTS") == "1", "requires local MeloTTS weights")
    def test_real_native_frontend_and_audio(self):
        # Inspect actual phones/tones used by native synthesis, not only our
        # dictionary. This catches longest-match conflicts and integration bugs.
        model = Path(r"D:\ReachyAI\models\speech\tts\vits-melo-tts-zh_en")
        lexicon = prepare_melo_lexicon(model)
        code = f"""
import sherpa_onnx
from pathlib import Path
d = Path({str(model)!r})
v = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d/'model.onnx'), lexicon={str(lexicon)!r}, tokens=str(d/'tokens.txt'))
m = sherpa_onnx.OfflineTtsModelConfig(vits=v, num_threads=4, debug=True)
c = sherpa_onnx.OfflineTtsConfig(model=m, rule_fsts=','.join(str(d/n) for n in ['phone.fst','date.fst','number.fst']))
a = sherpa_onnx.OfflineTts(c).generate('重新启动很重要。重试一下。这句话很长，孩子长高了。重庆银行行长还要还书。音乐很快乐。')
assert len(a.samples) > a.sample_rate
print('audio-ok')
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("audio-ok", result.stdout)
        actual = "\n".join(line for line in result.stderr.splitlines() if "ConvertTextToTokenIds:139" in line)
        for expected in ["重新: ch ong x in 2 2 1 1", "很重要: h en zh ong y ao 3 3 4 4 4 4",
                         "重试: ch ong sh ir 2 2 4 4", "很长: h en ch ang 3 3 2 2",
                         "长高: zh ang g ao 3 3 1 1", "银行行长: y in h ang h ang zh ang 2 2 2 2 2 2 3 3",
                         "还要: h ai y ao 2 2 4 4", "还书: h uan sh u 2 2 1 1",
                         "音乐: y in y ve 1 1 4 4", "快乐: k uai l e 4 4 4 4"]:
            with self.subTest(expected=expected):
                self.assertIn(expected, actual)


if __name__ == "__main__":
    unittest.main()
