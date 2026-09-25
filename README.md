# Reachy Mini Lite Family Agent / Reachy Mini 家庭机器人 Agent

[中文](#中文) · [English](#english)

## 中文

这是一个在 Windows 上运行的 Reachy Mini 有线版实验项目。桌面控制台启动机器人后台、模型路由和对话 Agent，显示实时摄像头预览与语音字幕，并提供软关机、重启 Agent、停止语音输出及模型/API 配置。代码包含语音识别与合成、视觉问答、联网搜索、机器人动作、家庭成员识别和持久记忆等功能。

项目仍在开发中，需要自行准备 Reachy Mini、官方 Python SDK、模型文件、推理运行时和 API 服务。仓库不包含任何模型权重、声库、个人资料或可执行文件。

### 架构

```text
Reachy Mini USB 摄像头/麦克风/扬声器/电机
        │
Reachy daemon ── Robot Agent ── 本地 SQLite 记忆与身份特征
                       │
                       ├─ 本地 ASR / VAD / TTS
                       ├─ 摄像头视觉前端
                       ├─ 搜索、天气与音乐工具
                       └─ 两路对话模型路由
                          ├─ 默认模型
                          └─ 复杂/云端模型

桌面控制台 ── 实时画面、字幕、服务控制、模型/API 列表
```

模型列表可以保存多个本地或云端 OpenAI 兼容端点；对话路由每次只使用选中的“默认模型”和“复杂/云端模型”。视觉前端可以单独指向支持图像输入的本地端点。复杂模型如果由 AutoWake 等服务按需启动，浅健康检查不会唤醒它。

### 目录与隐私

- `app/reachy_ai/`：机器人 Agent、路由、语音、视觉、记忆、身份与动作代码。
- `scripts/`：Windows 启动、维护、测试和控制台脚本。
- `tests/`：使用虚构人物资料的自动化测试。
- `config/settings.example.json`、`config/local-model.example.json`：可公开的配置模板。
- `data/`：对话历史、声纹和人脸特征、摄像头帧、音频及 SQLite 数据库；整个目录被 Git 忽略。
- `models/`、`runtime/`、`cache/`、`logs/`、`dist/`：本地权重、运行时、缓存、日志及 EXE；不提交。

API Key 通过各模型条目的 `api_key_env` 指定 Windows 用户环境变量。请勿将真实密钥、家庭成员姓名、声纹/人脸模板或运行记录放入 `*.example.json`、源码或 Git 提交。

### 准备与启动

1. 安装 Python 3.13，准备 Reachy Mini 官方 SDK、项目依赖、llama.cpp 或其他 OpenAI 兼容服务，以及所需的 ASR/TTS/视觉模型。`requirements-speech.txt` 只列出语音相关依赖，并非完整环境锁文件。
2. 将两个 `config/*.example.json` 分别复制为 `config/settings.json` 和 `config/local-model.json`，修改模型地址、名称、路径与端口。两个实际配置文件不会提交。项目脚本默认安装根目录是 `D:\ReachyAI`，换目录时需同步修改路径。
3. 在 Windows 用户环境变量中设置所选模型使用的 API Key。例如：

   ```powershell
   [Environment]::SetEnvironmentVariable('MY_MODEL_API_KEY', '<your-key>', 'User')
   ```

4. 连接并给机器人通电，再启动完整服务：

   ```powershell
   powershell -ExecutionPolicy Bypass -File D:\ReachyAI\scripts\start-autonomous-agent.ps1
   ```

5. 可选：运行 `scripts/build-desktop-exe.ps1` 构建控制台，然后打开 `dist/ReachyAI-Control.exe`。控制台会随应用启动后台；“立即停止说话”可打断语音，“重启 Agent”会重新启动对话循环。

路由 API 默认监听 `http://127.0.0.1:8077`，提供 `/health`、`/route` 和 OpenAI 兼容的 `/v1/chat/completions`。对机器人的软关机使用 `scripts/soft-shutdown-reachy.ps1`。控制台使用说明见 [docs/desktop-console.md](docs/desktop-console.md)。

### 验证与限制

```powershell
D:\ReachyAI\.venv\Scripts\python.exe -m unittest discover -s D:\ReachyAI\tests -q
```

测试不等于真实硬件验证。人脸和声纹识别会误判，尤其在多人同框、噪声或短语音时；不要把它用于安全认证。摄像头通常只能由一个程序占用。歌曲播放和歌声合成依赖各自的服务、可用音源或乐谱，效果取决于素材与模型。第三方模型、曲谱及音乐服务须由使用者自行安装并遵守其许可和服务条款。

## English

This is an experimental Windows project for the wired Reachy Mini. Its desktop console starts the robot daemon, model router, and conversation agent; shows a live camera preview and speech captions; and provides soft shutdown, agent restart, speech interruption, and model/API settings. The source includes ASR and TTS, visual question answering, web tools, robot motions, family member recognition, and persistent memory.

This is not a turnkey installer. You need a Reachy Mini, the official Python SDK, model files, inference runtimes, and any API services you choose. Model weights, voice banks, personal data, and built executables are not included.

### Design

The agent uses the Reachy daemon for media and motion. Local components handle voice activity detection, speech, vision, and a SQLite memory store. The model catalog can hold multiple OpenAI-compatible local or cloud endpoints. Two selected entries serve conversation: the default model and the complex/cloud model. A separate vision endpoint can analyze camera frames before a text-only model answers. Shallow health checks do not wake a sleeping backend.

### Setup

1. Install Python 3.13, the official Reachy Mini SDK, the project dependencies, and your chosen local inference and speech models. `requirements-speech.txt` covers speech dependencies only.
2. Copy `config/settings.example.json` to `config/settings.json` and `config/local-model.example.json` to `config/local-model.json`. Edit endpoint names, URLs, ports, and model paths. Scripts currently assume the project lives at `D:\ReachyAI`.
3. Put API keys in Windows user environment variables named by each endpoint's `api_key_env`. Never commit real keys.
4. Connect and power the robot, then run `scripts/start-autonomous-agent.ps1`, or build and launch the desktop console with `scripts/build-desktop-exe.ps1`.
5. Run tests with `D:\ReachyAI\.venv\Scripts\python.exe -m unittest discover -s D:\ReachyAI\tests -q`.

The router listens on `http://127.0.0.1:8077` by default and exposes `/health`, `/route`, and `/v1/chat/completions`. See [desktop console documentation](docs/desktop-console.md) for controls and configuration.

### Privacy and limitations

`data/` contains conversation history, camera frames, audio, face and voice embeddings, and the SQLite memory database. It is excluded from Git together with actual configuration, logs, models, runtimes, caches, and build outputs. The committed tests use fictional people. Review any new files before publishing changes.

Face and voice matching can fail, especially with multiple people, noise, or short utterances; it is not suitable for security authentication. The camera may be exclusive to one application. Music features depend on separate services and lawful, usable source material. Hardware behavior still requires testing on the actual robot.
