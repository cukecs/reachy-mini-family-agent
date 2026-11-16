"""Local sentence-context Mandarin disambiguation and explicit Melo phones.

Uses official g2pW ONNX assets with NumPy batches and a BERT tokenizer. No
PyTorch, network calls or chat-model requests run in the speech hot path.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from pypinyin import Style, lazy_pinyin

from .pronunciation import DEFAULT_OVERRIDES, melo_aliases

logger = logging.getLogger(__name__)
HAN = re.compile(r"[\u3400-\u9fff]")


class ContextualPronunciation:
    def __init__(self, model_dir: Path, tokens: set[str], *, overrides_path=DEFAULT_OVERRIDES):
        self.model_dir = model_dir
        self.aliases = melo_aliases(frozenset(tokens))
        self._session = None
        self._initialization_attempted = False
        self.backend = "not-loaded"
        self.last_readings: list[str] = []
        self.overrides = {}
        if overrides_path.exists():
            custom = json.loads(overrides_path.read_text(encoding="utf-8-sig"))
            for word, reading in custom.items():
                values = reading.split()
                if len(values) != len(word) or any(value not in self.aliases for value in values):
                    raise ValueError(f"Invalid TTS pronunciation override: {word!r}")
                self.overrides[word] = values

    def _load(self) -> bool:
        if self._initialization_attempted:
            return self._session is not None
        self._initialization_attempted = True
        try:
            import onnxruntime as ort
            from tokenizers import BertWordPieceTokenizer

            pairs = [line.split("\t") for line in
                     (self.model_dir / "POLYPHONIC_CHARS.txt").read_text(encoding="utf-8").splitlines() if line]
            self.labels = sorted({reading for _, reading in pairs})
            label_ids = {label: index for index, label in enumerate(self.labels)}
            self.allowed: dict[str, list[int]] = {}
            for char, reading in pairs:
                self.allowed.setdefault(char, []).append(label_ids[reading])
            self.char_ids = {char: index for index, char in enumerate(sorted(self.allowed))}
            self.to_pinyin = json.loads((self.model_dir / "bopomofo_to_pinyin_wo_tune_dict.json").read_text(encoding="utf-8"))
            self.s2t = dict(line.split("\t") for line in
                            (self.model_dir / "bert-base-chinese_s2t_dict.txt").read_text(encoding="utf-8").splitlines() if line)
            self.tokenizer = BertWordPieceTokenizer(str(self.model_dir / "vocab.txt"), lowercase=True)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            model = self.model_dir / "g2pw.int8.onnx"
            if not model.is_file():
                model = self.model_dir / "g2pw.onnx"
            self._session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
            self.backend = "g2pw-int8" if "int8" in model.name else "g2pw"
            logger.info("TTS contextual pronunciation: %s (CPU)", self.backend)
        except Exception:
            self.backend = "dictionary-fallback"
            logger.warning("g2pW unavailable; TTS uses limited dictionary fallback. Run scripts/prepare-g2pw.py", exc_info=True)
        return self._session is not None

    def readings(self, text: str) -> list[str]:
        # errors=list preserves one output slot for every English character,
        # digit and punctuation mark, keeping character-to-phone offsets exact.
        readings = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors=list)
        if len(readings) != len(text):
            raise ValueError("Mandarin pronunciation lost character alignment")
        if self._load():
            translated = "".join(self.s2t.get(char, char) for char in text)
            queries = []
            for index, char in enumerate(translated):
                if char not in self.allowed or not HAN.fullmatch(text[index]):
                    continue
                # g2pW v2's documented training window is 32 characters.
                start, end = max(0, index - 16), min(len(text), index + 16)
                encoded = self.tokenizer.encode(translated[start:end])
                local_index = index - start
                position = next((i for i, (left, right) in enumerate(encoded.offsets)
                                 if left <= local_index < right), None)
                if position is not None:
                    queries.append((index, char, encoded.ids, position))
            for offset in range(0, len(queries), 8):
                batch = queries[offset:offset + 8]
                size = len(batch)
                width = max(len(item[2]) for item in batch)
                ids = np.zeros((size, width), dtype=np.int64)
                attention = np.zeros_like(ids)
                masks = np.zeros((size, len(self.labels)), dtype=np.float32)
                chars = np.empty(size, dtype=np.int64)
                positions = np.empty(size, dtype=np.int64)
                for row, (_, char, values, position) in enumerate(batch):
                    ids[row, :len(values)] = values
                    attention[row, :len(values)] = 1
                    masks[row, self.allowed[char]] = 1
                    chars[row] = self.char_ids[char]
                    positions[row] = position
                probabilities = self._session.run(None, {
                    "input_ids": ids, "token_type_ids": np.zeros_like(ids),
                    "attention_mask": attention, "phoneme_mask": masks,
                    "char_ids": chars, "position_ids": positions,
                })[0]
                # Restrict to the query character's legal readings even if a
                # changed export stops applying its own phoneme mask.
                for row, (index, char, _, _) in enumerate(batch):
                    allowed = self.allowed[char]
                    label = self.labels[max(allowed, key=lambda i: probabilities[row, i])]
                    body = self.to_pinyin.get(label[:-1])
                    if body and body + label[-1] in self.aliases:
                        readings[index] = body + label[-1]
        # Only explicit private overrides take precedence over the contextual
        # model. The old generic common-phrase patch must not override meaning.
        index = 0
        words = sorted(self.overrides, key=len, reverse=True)
        while index < len(text):
            word = next((word for word in words if text.startswith(word, index)), None)
            if word:
                readings[index:index + len(word)] = self.overrides[word]
                index += len(word)
            else:
                index += 1
        self.last_readings = readings
        return readings

    def prepare(self, text: str) -> str:
        if not HAN.search(text):
            self.last_readings = list(text)
            return text
        try:
            readings = self.readings(text)
        except Exception:
            logger.exception("Contextual TTS inference failed; reverting to dictionary pronunciation")
            self._session = None
            self._initialization_attempted = True
            self.backend = "dictionary-fallback"
            readings = self.readings(text)
        if self.backend == "dictionary-fallback":
            # Preserve the previous, patched native lexicon on degradation;
            # bare pypinyin would reintroduce the missing-phrase failures.
            return text
        # Each syllable gets a private-use lexicon symbol with explicit phones
        # and tones. This bypasses sherpa's longest-word guessing completely,
        # while preserving English, numbers and punctuation for native handling.
        return "".join(self.aliases.get(reading, char) if HAN.fullmatch(char) else char
                       for char, reading in zip(text, readings))
