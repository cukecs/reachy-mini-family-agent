# Reachy AI 控制台 / Desktop Console

控制台是 Windows 入口。接好 Reachy Mini 的 USB 和电源后启动 `ReachyAI-Control.exe`，程序会尝试启动 daemon、模型路由和对话 Agent。关闭窗口不会自动软关机。

The Windows console starts the Reachy daemon, model router, and conversation agent after the robot is connected and powered. Closing the window does not soft-shutdown the robot.

## 控制 / Controls

| 按钮 Button | 功能 Behavior |
| --- | --- |
| 唤醒机器人 / Wake | 启动后台并唤醒机器人 / Start services and wake the robot |
| 软关机 / Soft shutdown | 停止 Agent，执行睡眠姿态并卸力 / Stop Agent, pose for sleep, release motors |
| 启动 AI 服务 / Start AI | 启动本机 llama.cpp 与路由器 / Start local llama.cpp and router |
| 停止 AI 服务 / Stop AI | 停止 Agent、路由器与本机模型 / Stop Agent, router, and local model |
| 重启 Agent / Restart Agent | 重启对话循环 / Restart the conversation loop |
| 立即停止说话 / Stop speech | 打断正在合成或播放的语音 / Interrupt pending or playing speech |

“实时画面与字幕”页显示机器人摄像头预览和转录/回复。摄像头由 Agent 访问；不要同时打开会占用摄像头的其他程序。字幕和画面保存在本机 `data/`，不会进入 Git 仓库。

The Live tab shows the camera preview and conversation captions. Other applications may conflict with Agent camera access. Captions and frames stay in the local `data/` directory, which is excluded from Git.

## 模型与 API / Models and APIs

在“模型 / API 列表”中添加任意 OpenAI 兼容端点，填写显示名称、ID、API 地址、接口模型名、超时、输出长度与高级 JSON 参数。API Key 存在对应的 Windows 用户环境变量中。选择“默认模型”和“复杂 / 云端模型”后，点击“保存并重启 AI”。列表可保存更多候选，但对话路由只使用这两个选中的端点。

Add OpenAI-compatible endpoints to the model catalog. Each entry contains an ID, display name, API URL, model name, timeout, output limit, and optional JSON parameters. The API key is stored in the specified Windows user environment variable. Select the default and complex/cloud models, then choose Save and Restart AI. Other catalog entries remain available but are not active conversation routes.

本机视觉端点由 `config/settings.json` 的 `router.vision_model_id` 指定；它应支持图像输入。`config/local-model.json` 存放 llama.cpp 的本机路径与显存参数。实际配置、备份与密钥均不提交。

The local visual endpoint is selected with `router.vision_model_id` in `config/settings.json` and should support image input. `config/local-model.json` holds local llama.cpp paths and runtime parameters. Actual configuration, backups, and credentials are not committed.

## 注意 / Notes

- 项目脚本默认根目录为 `D:\ReachyAI`。移动项目后需修改脚本路径。 / Scripts currently assume `D:\ReachyAI`.
- `data/` 含家庭成员记忆、声纹/人脸特征、摄像头帧和音频，不要手动加入 Git。 / `data/` contains private family memory, biometric templates, frames, and audio; never add it to Git.
- EXE 不包含模型、声库或官方 SDK。 / The EXE does not bundle model files, voice banks, or the official SDK.
