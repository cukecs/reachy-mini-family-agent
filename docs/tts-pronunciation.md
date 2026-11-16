# 中文多音字 / Mandarin polyphones

## 中文

### 当前处理链路（2026-10-29）

此前仅补词组的方案不能完成上下文消歧。现在主要链路为：原文 → 本地 g2pW 判断多音字 → 逐字带声调拼音 → 明确音素/声调 → 原 MeloTTS 中文女声。合成器通过缓存词典中的私用区符号接收指定读音，避免重新按最长词组猜音。原文、字幕和记忆不做替换。

g2pW 在 CPU 运行，使用本地 INT8 ONNX 权重（约 360 MiB），不需要 PyTorch、Transformers、CUDA 或云端 API；模型仅在第一次中文合成时加载。当前模型的上下文窗口为目标字附近 32 个字符，并非通用语言大模型；歧义表达和少见人名仍可能读错，不保证每句完美。

首次准备（下载官方约 589 MB 压缩包，校验 SHA256，再生成 INT8 模型；原文件和运行模型均保存在 D 盘，不提交 Git）：

```powershell
D:\ReachyAI\.venv\Scripts\python.exe -m pip install -r D:\ReachyAI\requirements-speech.txt
D:\ReachyAI\.venv\Scripts\python.exe D:\ReachyAI\scripts\prepare-g2pw.py
```

重启 Agent 后，首次中文合成的日志应显示 `TTS pronunciation frontend: g2pw-int8 (CPU)`。如果资源缺失或推理失败，会记录警告并回退到旧的修补词典，不会静默声称正在做上下文消歧。运行过程中不会联网下载任何模型。

诊断实际发音（不连接机器人）：

```powershell
D:\ReachyAI\.venv\Scripts\python.exe D:\ReachyAI\scripts\test-speech.py --text "这根绳子有三米长。孩子又长了三厘米。我还没有把书还给你。" --show-pronunciation
```

可选私有词典覆盖仍优先于模型；旧代码中的一般词组规则不再覆盖正常模型推理出的读音。模型不可用时才保留下面的原生词典降级机制。

### 原生词典降级与私有覆盖

本地中文女声使用 sherpa-onnx / MeloTTS。其前端按最长词组查找读音，未命中的词组会退回单字默认读音；例如未收录的“很长”可能使用“长大”的 zhǎng，而不是 cháng。

Agent 启动 TTS 时会保留原模型词典，补入 pypinyin 的缺失词组，并覆盖常用多音词、口语短语：重新/重要、长高/很长、银行行长、还要/还书、音乐/快乐、调节/音调等。合并结果缓存于 `D:\ReachyAI\cache\tts-pronunciation`，不会改写模型文件、对话原文、字幕或记忆；也不会把所有“重”“长”强制改成同一个音。

词典降级不是语义消歧模型。同一短语本身有多个合理读音（例如“重读”表示重新朗读或加重读音），以及人名、地名、孤立单字，仍可能需要更具体上下文或读音设置。

可选：复制 `config/tts-pronunciation.example.json` 为 `config/tts-pronunciation.json`，按“原词组: 空格分隔的带数字声调拼音”添加读音，再重启 Agent。每个汉字对应一个拼音，1–4 为声调、5 为轻声；词组最多 10 个汉字。真实人名等私有读音只放在实际配置文件，不要写进公开模板。实际配置和缓存不会提交 Git。

安装依赖及验证（验证不会启动机器人或播放声音）：

```powershell
D:\ReachyAI\.venv\Scripts\python.exe -m pip install -r D:\ReachyAI\requirements-speech.txt
$env:RUN_TTS_MODEL_TESTS = '1'
D:\ReachyAI\.venv\Scripts\python.exe -m unittest discover -s D:\ReachyAI\tests -p test_pronunciation.py -v
```

测试检查实际原生 TTS 前端使用的音素与声调，并执行合成；仅在具备本地模型权重时设置 `RUN_TTS_MODEL_TESTS=1`。新上下文处理层另运行 `-p test_contextual_pronunciation.py`，包括真实 g2pW 上下文推理及传到原生合成器的音素验证。

## English

The primary pipeline now uses local g2pW ONNX CPU inference to disambiguate Mandarin polyphones, then feeds explicit phoneme/tone aliases to MeloTTS instead of letting its longest-word frontend guess again. Captions, conversation and memory retain original text. The INT8 model is about 360 MiB, uses the model's 32-character context window, and requires neither PyTorch/Transformers nor a GPU/cloud API. It can still mispronounce ambiguous wording and uncommon names.

Run `scripts/prepare-g2pw.py` once after installing dependencies. Assets are downloaded from the official g2pW repository/model host, the model archive SHA256 is checked, and files remain under `D:\ReachyAI\models\speech\g2pw`. The runtime does not download anything. Its startup diagnostic must say `g2pw-int8`; missing assets/inference errors log a warning and degrade to the previous patched native dictionary.

Use `scripts/test-speech.py --text "..." --show-pronunciation` for a waveform and the chosen syllables. `test_contextual_pronunciation.py` tests actual contextual inference and native synthesis phones with `RUN_TTS_MODEL_TESTS=1`. Explicit private pronunciation overrides take precedence over the neural model.

The local female Mandarin voice uses sherpa-onnx / MeloTTS's longest-phrase lexicon frontend. Missing phrases fall back to single-character defaults. The Agent creates a cached merged lexicon, preserving model assets and English entries, adding missing pypinyin phrases and explicit common Mandarin pronunciation corrections. Conversation text, captions and memory remain unchanged.

This is dictionary-based disambiguation, not full semantic understanding. Ambiguous phrases, names and isolated characters can still need explicit pronunciation. Optional private overrides belong in `config/tts-pronunciation.json`: map up to ten Chinese characters to space-separated tone-number pinyin, one syllable per character, with 5 for neutral tone. Restart the Agent after editing. Private overrides and cached lexicons are Git-ignored.

Set `RUN_TTS_MODEL_TESTS=1` and run the command above to verify actual native frontend phones/tones and waveform generation. It does not connect to the robot or play sound.

Sources: [g2pW official project](https://github.com/GitYCC/g2pW), [sherpa-onnx MeloTTS frontend](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.8/sherpa-onnx/csrc/melo-tts-lexicon.cc), [pypinyin](https://pypi.org/project/pypinyin/0.55.0/), [MeloTTS phone conventions](https://github.com/myshell-ai/MeloTTS/blob/main/melo/text/opencpop-strict.txt).
