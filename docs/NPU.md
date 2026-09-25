# Ryzen 7 8845H NPU 使用边界

这台电脑已检测到 `NPU Compute Accelerator Device`，硬件是 Hawk Point/XDNA 1。

在 Reachy AI 中的分工：

- NPU：后续承载 INT8 CNN，例如人脸检测、目标检测和图像分类。
- RTX 4060：Agents-A1-4B 文本模型与按需视觉模型。
- CPU：Reachy daemon、音频、VAD、ASR、TTS、路由和状态机。
- R9700：经 AutoWake 按需唤醒的复杂推理模型。

Hawk Point 不支持 Ryzen AI 当前的 OGA LLM NPU 路径，也不支持当前 Whisper
全编码器 NPU 卸载。因此不把 LLM、VLM、ASR 或 TTS 强行迁移到 NPU。这可以避免
驱动兼容问题，也不会挤占 RTX 4060 的显存。

## 驱动

当前旧驱动为 `32.0.201.215`；Ryzen AI 1.8 要求至少 `32.0.203.280`。
官方 WHQL `.203.376` 安装包已经下载并校验到 D 盘，但安装需要一次管理员 UAC：

```powershell
powershell -ExecutionPolicy Bypass -File D:\ReachyAI\scripts\install-npu-driver.ps1
```

安装后重启，再检查：

```powershell
powershell -ExecutionPolicy Bypass -File D:\ReachyAI\scripts\check-npu.ps1
```

Ryzen AI SDK/Windows ML 的 Python 环境应另建 Python 3.11 或 3.12 环境，不与
当前 Reachy Python 3.13 环境混装。
