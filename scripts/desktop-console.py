from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
import webbrowser
import winreg
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from PIL import Image, ImageTk


ROOT = Path(r"D:\ReachyAI")
SETTINGS_PATH = ROOT / "config" / "settings.json"
LOCAL_MODEL_PATH = ROOT / "config" / "local-model.json"
STATE_PATH = ROOT / "data" / "runtime-state.json"
CAMERA_PATH = ROOT / "data" / "live-camera.jpg"
SPEECH_CONTROL_PATH = ROOT / "data" / "speech-control.json"
SCRIPTS = ROOT / "scripts"
DEFAULT_SCORE_ROOT = ROOT / "data" / "scores"
_control_mutex_handle = None
_external_launch_lock = threading.Lock()


def acquire_control_single_instance() -> bool:
    """Prevent two GUI launchers from starting competing backend workflows."""
    global _control_mutex_handle
    if os.name != "nt":
        return True
    _control_mutex_handle = ctypes.windll.kernel32.CreateMutexW(
        None, True, "Local\\ReachyAI_ControlConsole"
    )
    return ctypes.windll.kernel32.GetLastError() != 183


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _registry_env(name: str) -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value)
    except FileNotFoundError:
        return os.environ.get(name, "")


def _set_registry_env(name: str, value: str) -> None:
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        if value:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
            os.environ[name] = value
        else:
            try:
                winreg.DeleteValue(key, name)
            except FileNotFoundError:
                pass
            os.environ.pop(name, None)


def _broadcast_environment_change() -> None:
    SMTO_ABORTIFHUNG = 0x0002
    WM_SETTINGCHANGE = 0x001A
    ctypes.windll.user32.SendMessageTimeoutW(
        0xFFFF, WM_SETTINGCHANGE, 0, "Environment", SMTO_ABORTIFHUNG, 3000, None
    )


def _sanitized_external_env() -> dict[str, str]:
    """Build an environment safe for non-PyInstaller child processes."""
    env = os.environ.copy()
    mei = str(getattr(sys, "_MEIPASS", "") or "").strip()
    if mei:
        mei_norm = os.path.normcase(os.path.abspath(mei))
        cleaned_path: list[str] = []
        for entry in env.get("PATH", "").split(os.pathsep):
            if not entry:
                continue
            try:
                entry_norm = os.path.normcase(os.path.abspath(entry))
            except OSError:
                entry_norm = os.path.normcase(entry)
            if entry_norm == mei_norm or entry_norm.startswith(mei_norm + os.sep):
                continue
            cleaned_path.append(entry)
        env["PATH"] = os.pathsep.join(cleaned_path)

    # PyInstaller uses these internally. External long-running services must
    # never inherit a pointer to the GUI's temporary extraction directory.
    env.pop("_MEIPASS2", None)
    env.pop("_PYI_APPLICATION_HOME_DIR", None)
    return env


def _run_external_hidden(
    args: list[str],
    *,
    cwd: str,
    timeout: float,
) -> tuple[int, str, str]:
    """Run a long-lived-service launcher without leaking onefile DLL state."""
    env = _sanitized_external_env()
    mei = str(getattr(sys, "_MEIPASS", "") or "").strip()

    with _external_launch_lock:
        if os.name == "nt":
            # PyInstaller onefile sets the process DLL directory to _MEIPASS.
            # If inherited, llama-server loads VCRUNTIME*.dll from _MEI and
            # keeps the GUI's temp directory locked after the GUI exits.
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        try:
            process = subprocess.Popen(
                args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            if os.name == "nt" and mei:
                ctypes.windll.kernel32.SetDllDirectoryW(mei)

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise TimeoutError(
            f"外部启动脚本运行超过 {timeout:.0f} 秒"
        )
    return process.returncode, stdout, stderr


class ReachyConsole(tk.Tk):
    COLORS = {
        "bg": "#111722",
        "panel": "#192231",
        "panel2": "#202c3d",
        "text": "#eef5ff",
        "muted": "#9fb0c5",
        "accent": "#55d6be",
        "blue": "#62a8ff",
        "danger": "#ff7272",
        "warning": "#f6c85f",
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("Reachy AI 控制台")
        self.geometry("1320x860")
        self.minsize(1050, 700)
        self.configure(bg=self.COLORS["bg"])
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._camera_mtime = 0.0
        self._state_mtime = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._settings_vars: dict[str, tk.StringVar] = {}
        self._runtime_vars: dict[str, tk.StringVar] = {}
        self._secret_vars: dict[str, tk.StringVar] = {}
        self._models: list[dict] = []
        self._model_vars: dict[str, tk.StringVar] = {}
        self._model_secret_values: dict[str, str] = {}
        self._default_model_var = tk.StringVar()
        self._cloud_model_var = tk.StringVar()
        self._network_vars: dict[str, tk.StringVar] = {}
        self._singing_vars: dict[str, tk.StringVar] = {}
        self._speech_filter_vars: dict[str, tk.StringVar] = {}
        self._launch_start_requested = False
        self._configure_style()
        self._build_ui()
        self._load_settings()
        self.after(120, self._poll_camera)
        self.after(250, self._poll_runtime_state)
        self.after(200, self._drain_events)
        self.after(300, self._refresh_status_async)
        self.after(800, self._start_backends_on_launch)

    def _start_backends_on_launch(self) -> None:
        """Start the daemon, AI stack, and robot agent once the UI is visible."""
        if self._launch_start_requested:
            return
        self._launch_start_requested = True
        self._run_script(
            "start-autonomous-agent.ps1",
            "APP 已启动，正在连接机器人并启动 AI 后台",
        )

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=self.COLORS["bg"])
        style.configure("Panel.TFrame", background=self.COLORS["panel"])
        style.configure(
            "TLabel", background=self.COLORS["bg"], foreground=self.COLORS["text"]
        )
        style.configure(
            "Muted.TLabel", background=self.COLORS["panel"], foreground=self.COLORS["muted"]
        )
        style.configure(
            "Title.TLabel",
            background=self.COLORS["bg"],
            foreground=self.COLORS["text"],
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=self.COLORS["panel"],
            foreground=self.COLORS["accent"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "TButton", font=("Microsoft YaHei UI", 10), padding=(12, 8)
        )
        style.configure("Accent.TButton", foreground="#0b1918", background=self.COLORS["accent"])
        style.map("Accent.TButton", background=[("active", "#7ce4d1")])
        style.configure("Danger.TButton", foreground="#251010", background=self.COLORS["danger"])
        style.map("Danger.TButton", background=[("active", "#ff9696")])
        style.configure("TNotebook", background=self.COLORS["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.COLORS["panel"],
            foreground=self.COLORS["muted"],
            padding=(18, 9),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.COLORS["panel2"])],
            foreground=[("selected", self.COLORS["text"])],
        )
        style.configure(
            "TLabelframe",
            background=self.COLORS["panel"],
            foreground=self.COLORS["text"],
        )
        style.configure(
            "TLabelframe.Label",
            background=self.COLORS["panel"],
            foreground=self.COLORS["accent"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure("TCheckbutton", background=self.COLORS["panel"], foreground=self.COLORS["text"])

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(18, 14, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Reachy AI 控制台", style="Title.TLabel").pack(side="left")
        self.summary_var = tk.StringVar(value="正在检测服务状态…")
        ttk.Label(header, textvariable=self.summary_var).pack(side="right", pady=8)

        toolbar = ttk.Frame(self, style="Panel.TFrame", padding=10)
        toolbar.pack(fill="x", padx=18, pady=(0, 12))
        ttk.Button(
            toolbar,
            text="唤醒机器人",
            style="Accent.TButton",
            command=lambda: self._run_script("start-autonomous-agent.ps1", "正在唤醒机器人"),
        ).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            text="软关机（缩脖子）",
            style="Danger.TButton",
            command=lambda: self._run_script("soft-shutdown-reachy.ps1", "正在让机器人软关机"),
        ).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            text="启动 AI 服务",
            command=lambda: self._run_script("start-stack-hidden.ps1", "正在启动模型与路由器"),
        ).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            text="停止 AI 服务",
            command=lambda: self._run_script("stop-ai-stack.ps1", "正在停止 AI 服务"),
        ).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            text="重启 Agent",
            command=lambda: self._run_script("restart-robot-agent.ps1", "正在重启对话 Agent"),
        ).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            text="立即停止说话",
            style="Danger.TButton",
            command=self._interrupt_speech,
        ).pack(side="left", padx=4)
        self.robot_status = tk.StringVar(value="机器人：检测中")
        self.ai_status = tk.StringVar(value="AI：检测中")
        ttk.Label(toolbar, textvariable=self.robot_status, style="Status.TLabel").pack(
            side="right", padx=12
        )
        ttk.Label(toolbar, textvariable=self.ai_status, style="Status.TLabel").pack(
            side="right", padx=12
        )

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        live_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
        settings_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
        logs_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
        notebook.add(live_tab, text="实时画面与字幕")
        notebook.add(settings_tab, text="模型与 API 配置")
        notebook.add(logs_tab, text="运行日志")
        self._build_live_tab(live_tab)
        self._build_settings_tab(settings_tab)
        self._build_logs_tab(logs_tab)

    def _build_live_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="机器人视野（约 5 FPS）").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(parent, text="实时语音字幕").grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 8))
        self.camera_label = tk.Label(
            parent,
            text="机器人启动后将在这里显示摄像头画面",
            bg="#070a0f",
            fg=self.COLORS["muted"],
            font=("Microsoft YaHei UI", 12),
            bd=0,
        )
        self.camera_label.grid(row=1, column=0, sticky="nsew")
        self.transcript = scrolledtext.ScrolledText(
            parent,
            state="disabled",
            wrap="word",
            bg="#101722",
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            padx=14,
            pady=12,
            font=("Microsoft YaHei UI", 11),
        )
        self.transcript.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        self.transcript.tag_configure("user", foreground=self.COLORS["blue"], spacing1=8)
        self.transcript.tag_configure("assistant", foreground=self.COLORS["text"], spacing1=8)
        self.transcript.tag_configure("meta", foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 9))

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, bg=self.COLORS["panel"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, style="Panel.TFrame")
        body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=1)
        catalog = ttk.LabelFrame(body, text="模型 / API 列表", padding=12)
        catalog.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        catalog.columnconfigure(0, weight=3)
        catalog.columnconfigure(1, weight=2)

        self.model_tree = ttk.Treeview(
            catalog,
            columns=("name", "id", "model", "url", "key"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        for column, label, width in (
            ("name", "显示名称", 150),
            ("id", "ID", 105),
            ("model", "接口模型名", 155),
            ("url", "API 地址", 260),
            ("key", "Key 环境变量", 145),
        ):
            self.model_tree.heading(column, text=label)
            self.model_tree.column(column, width=width, minwidth=80)
        self.model_tree.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        self.model_tree.bind("<<TreeviewSelect>>", self._select_model)

        editor = ttk.Frame(catalog, style="Panel.TFrame")
        editor.grid(row=0, column=1, sticky="nsew")
        model_fields = [
            ("id", "模型 ID（英文/数字/-/_）", False),
            ("name", "显示名称", False),
            ("base_url", "API 地址", False),
            ("model", "接口模型名", False),
            ("api_key_env", "API Key 环境变量", False),
            ("api_key", "API Key（只存 Windows 用户环境）", True),
            ("timeout_seconds", "超时（秒）", False),
            ("max_tokens", "最大输出 tokens", False),
            ("temperature", "温度", False),
            ("extra_body", "高级参数（JSON）", False),
        ]
        for row, (field, label, secret) in enumerate(model_fields):
            ttk.Label(editor, text=label, style="Muted.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=3
            )
            variable = tk.StringVar()
            self._model_vars[field] = variable
            ttk.Entry(editor, textvariable=variable, show="●" if secret else "").grid(
                row=row, column=1, sticky="ew", pady=3
            )
        editor.columnconfigure(1, weight=1)
        model_buttons = ttk.Frame(catalog, style="Panel.TFrame")
        model_buttons.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Button(model_buttons, text="新增 / 更新", command=self._upsert_model).pack(side="left", padx=4)
        ttk.Button(model_buttons, text="清空表单", command=self._clear_model_form).pack(side="left", padx=4)
        ttk.Button(model_buttons, text="删除选中", style="Danger.TButton", command=self._delete_model).pack(side="left", padx=4)

        selectors = ttk.Frame(catalog, style="Panel.TFrame")
        selectors.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Label(selectors, text="默认模型", style="Muted.TLabel").pack(side="left")
        self.default_model_combo = ttk.Combobox(
            selectors, textvariable=self._default_model_var, state="readonly", width=26
        )
        self.default_model_combo.pack(side="left", padx=(6, 18))
        ttk.Label(selectors, text="复杂 / 云端模型", style="Muted.TLabel").pack(side="left")
        self.cloud_model_combo = ttk.Combobox(
            selectors, textvariable=self._cloud_model_var, state="readonly", width=26
        )
        self.cloud_model_combo.pack(side="left", padx=6)
        ttk.Label(
            selectors,
            text="路由只在这两个选中的模型之间切换。",
            style="Muted.TLabel",
        ).pack(side="left", padx=14)

        runtime = ttk.LabelFrame(body, text="本地 llama.cpp 运行参数", padding=12)
        runtime.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
        runtime_fields = [
            ("server_path", "llama-server 路径"),
            ("model_path", "GGUF 模型路径"),
            ("mmproj_path", "视觉 mmproj 路径"),
            ("alias", "服务模型别名"),
            ("host", "监听地址"),
            ("port", "端口"),
            ("context_size", "上下文长度"),
            ("gpu_layers", "GPU 层数"),
            ("cache_type_k", "K Cache 类型"),
            ("cache_type_v", "V Cache 类型"),
            ("parallel", "并行槽位"),
            ("image_min_tokens", "最少图像 tokens"),
            ("image_max_tokens", "最多图像 tokens"),
            ("reasoning_budget", "推理预算 tokens"),
            ("flash_attention", "Flash Attention（true/false）"),
            ("mmproj_offload", "视觉模块使用GPU（true/false）"),
        ]
        for index, (field, label) in enumerate(runtime_fields):
            row, column = divmod(index, 3)
            cell = ttk.Frame(runtime, style="Panel.TFrame")
            cell.grid(row=row, column=column, sticky="ew", padx=6, pady=5)
            runtime.columnconfigure(column, weight=1)
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")
            variable = tk.StringVar()
            self._runtime_vars[field] = variable
            line = ttk.Frame(cell, style="Panel.TFrame")
            line.pack(fill="x")
            ttk.Entry(line, textvariable=variable).pack(side="left", fill="x", expand=True)
            if field in {"server_path", "model_path", "mmproj_path"}:
                ttk.Button(line, text="浏览", command=lambda f=field: self._browse_file(f)).pack(side="left", padx=(4, 0))

        router = ttk.LabelFrame(body, text="智能路由与 Agent 上下文", padding=12)
        router.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
        for index, (section, field, label) in enumerate(
            [
                ("router", "confidence_threshold", "置信度阈值"),
                ("router", "classify_timeout_seconds", "分类超时秒数"),
                ("router", "long_input_chars", "长输入升级字符数"),
                ("agent", "history_token_budget", "默认模型历史 token 预算"),
                ("agent", "cloud_history_token_budget", "复杂/云端历史 token 预算"),
                ("agent", "history_max_messages", "持久对话消息上限"),
            ]
        ):
            row, column = divmod(index, 3)
            cell = ttk.Frame(router, style="Panel.TFrame")
            cell.grid(row=row, column=column, sticky="ew", padx=6, pady=5)
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")
            variable = tk.StringVar()
            self._settings_vars[f"{section}.{field}"] = variable
            ttk.Entry(cell, textvariable=variable, width=20).pack(fill="x", pady=(2, 0))
            router.columnconfigure(column, weight=1)

        network = ttk.LabelFrame(body, text="联网工具（DDGS + SearXNG + Open-Meteo）", padding=12)
        network.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
        network_fields = [
            ("enabled", "启用联网工具"),
            ("search.provider", "网页搜索提供方"),
            ("search.searxng_url", "SearXNG 地址"),
            ("search.url", "网页搜索 MCP 回退地址"),
            ("time.url", "时间 MCP 地址"),
        ]
        for row, (field, label) in enumerate(network_fields):
            ttk.Label(network, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", padx=6, pady=4)
            variable = tk.StringVar()
            self._network_vars[field] = variable
            ttk.Entry(network, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        network.columnconfigure(1, weight=1)

        singing = ttk.LabelFrame(body, text="中文女声与真实乐谱", padding=12)
        singing.grid(row=4, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
        singing_fields = [
            ("enabled", "启用歌声"),
            ("score_root", "乐谱库目录"),
            ("speaker", "女声风格"),
            ("max_chars", "单次最多歌词字数"),
            ("pitch_steps", "音高推理步数"),
            ("variance_steps", "表现力推理步数"),
            ("acoustic_steps", "声学推理步数"),
            ("timeout_seconds", "渲染超时（秒）"),
        ]
        for index, (field, label) in enumerate(singing_fields):
            row, column = divmod(index, 4)
            cell = ttk.Frame(singing, style="Panel.TFrame")
            cell.grid(row=row, column=column, sticky="ew", padx=6, pady=5)
            singing.columnconfigure(column, weight=1)
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")
            variable = tk.StringVar()
            self._singing_vars[field] = variable
            ttk.Entry(cell, textvariable=variable).pack(fill="x")
        score_controls = ttk.Frame(singing, style="Panel.TFrame")
        score_controls.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=(10, 2))
        ttk.Button(score_controls, text="导入乐谱 / 音频", command=self._import_score).pack(side="left", padx=4)
        ttk.Button(score_controls, text="打开乐谱库", command=self._open_score_library).pack(side="left", padx=4)
        ttk.Button(score_controls, text="搜索中国曲谱网", command=self._search_public_scores).pack(side="left", padx=4)
        ttk.Label(
            score_controls,
            text="优先本地；其次中国曲谱网和其他可直接访问来源；合成前校验主旋律、节拍与歌词。",
            style="Muted.TLabel",
        ).pack(side="left", padx=12)

        speech_filter = ttk.LabelFrame(body, text="麦克风人声过滤", padding=12)
        speech_filter.grid(row=5, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
        filter_fields = [
            ("enabled", "启用过滤"),
            ("threshold", "人声阈值（高=更严格）"),
            ("min_speech_seconds", "最短累计人声（秒）"),
            ("min_contiguous_seconds", "最短连续人声（秒）"),
        ]
        for column, (field, label) in enumerate(filter_fields):
            cell = ttk.Frame(speech_filter, style="Panel.TFrame")
            cell.grid(row=0, column=column, sticky="ew", padx=6, pady=5)
            speech_filter.columnconfigure(column, weight=1)
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")
            variable = tk.StringVar()
            self._speech_filter_vars[field] = variable
            ttk.Entry(cell, textvariable=variable).pack(fill="x")

        controls = ttk.Frame(body, style="Panel.TFrame", padding=(6, 10))
        controls.grid(row=6, column=0, columnspan=3, sticky="ew")
        ttk.Button(controls, text="重新载入", command=self._load_settings).pack(side="left", padx=4)
        ttk.Button(controls, text="仅保存", command=self._save_settings).pack(side="left", padx=4)
        ttk.Button(
            controls,
            text="保存并重启 AI",
            style="Accent.TButton",
            command=self._save_and_restart,
        ).pack(side="left", padx=4)
        ttk.Label(
            controls,
            text="API Key 写入当前 Windows 用户环境；模型权重继续保存在 D 盘。",
            style="Muted.TLabel",
        ).pack(side="right", padx=8)

    def _build_logs_tab(self, parent: ttk.Frame) -> None:
        self.logs = scrolledtext.ScrolledText(
            parent,
            state="disabled",
            bg="#0d131d",
            fg="#d6e2f0",
            insertbackground="#d6e2f0",
            relief="flat",
            font=("Cascadia Mono", 10),
        )
        self.logs.pack(fill="both", expand=True)

    def _browse_file(self, field: str) -> None:
        initial = Path(self._runtime_vars[field].get() or ROOT)
        filename = filedialog.askopenfilename(initialdir=initial.parent if initial.suffix else initial)
        if filename:
            self._runtime_vars[field].set(filename.replace("/", "\\"))

    @staticmethod
    def _legacy_models(settings: dict) -> list[dict]:
        names = {
            "local": "本地模型",
            "medium": "局域网 / AutoWake 模型",
            "cloud": "云端模型",
        }
        result: list[dict] = []
        for endpoint_id in ("local", "medium", "cloud"):
            value = settings.get(endpoint_id)
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item["id"] = endpoint_id
            item["name"] = names[endpoint_id]
            result.append(item)
        return result

    def _refresh_model_catalog(self) -> None:
        self.model_tree.delete(*self.model_tree.get_children())
        for item in self._models:
            self.model_tree.insert(
                "",
                "end",
                iid=str(item["id"]),
                values=(
                    item.get("name", ""),
                    item.get("id", ""),
                    item.get("model", ""),
                    item.get("base_url", ""),
                    item.get("api_key_env", ""),
                ),
            )
        model_ids = [str(item["id"]) for item in self._models]
        self.default_model_combo.configure(values=model_ids)
        self.cloud_model_combo.configure(values=model_ids)
        if model_ids and self._default_model_var.get() not in model_ids:
            self._default_model_var.set(model_ids[0])
        if model_ids and self._cloud_model_var.get() not in model_ids:
            self._cloud_model_var.set(model_ids[-1])

    def _clear_model_form(self) -> None:
        defaults = {
            "timeout_seconds": "120",
            "max_tokens": "2048",
            "temperature": "0.7",
            "extra_body": "{}",
        }
        for field, variable in self._model_vars.items():
            variable.set(defaults.get(field, ""))
        for selected in self.model_tree.selection():
            self.model_tree.selection_remove(selected)

    def _select_model(self, _event=None) -> None:
        selection = self.model_tree.selection()
        if not selection:
            return
        endpoint_id = selection[0]
        item = next((model for model in self._models if str(model["id"]) == endpoint_id), None)
        if item is None:
            return
        for field in (
            "id",
            "name",
            "base_url",
            "model",
            "api_key_env",
            "timeout_seconds",
            "max_tokens",
            "temperature",
        ):
            self._model_vars[field].set(str(item.get(field, "")))
        self._model_vars["extra_body"].set(
            json.dumps(item.get("extra_body") or {}, ensure_ascii=False, separators=(",", ":"))
        )
        env_name = str(item.get("api_key_env", ""))
        self._model_vars["api_key"].set(
            self._model_secret_values.get(env_name, _registry_env(env_name) if env_name else "")
        )

    def _upsert_model(self, *, silent: bool = False) -> bool:
        try:
            endpoint_id = self._model_vars["id"].get().strip()
            if not endpoint_id:
                if silent:
                    return False
                raise ValueError("请填写模型 ID。")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", endpoint_id):
                raise ValueError("模型 ID 只能包含英文、数字、下划线和短横线。")
            advanced = json.loads(self._model_vars["extra_body"].get().strip() or "{}")
            if not isinstance(advanced, dict):
                raise ValueError("高级参数必须是 JSON 对象。")
            item = {
                "id": endpoint_id,
                "name": self._model_vars["name"].get().strip() or endpoint_id,
                "base_url": self._model_vars["base_url"].get().strip().rstrip("/"),
                "model": self._model_vars["model"].get().strip(),
                "api_key_env": self._model_vars["api_key_env"].get().strip(),
                "timeout_seconds": self._convert_number(
                    self._model_vars["timeout_seconds"].get(), float, "超时"
                ),
                "max_tokens": self._convert_number(
                    self._model_vars["max_tokens"].get(), int, "最大输出 tokens"
                ),
                "temperature": self._convert_number(
                    self._model_vars["temperature"].get(), float, "温度"
                ),
                "extra_body": advanced,
            }
            if not item["base_url"] or not item["model"]:
                raise ValueError("API 地址和接口模型名不能为空。")
            existing = next(
                (index for index, model in enumerate(self._models) if model["id"] == endpoint_id),
                None,
            )
            if existing is None:
                self._models.append(item)
            else:
                self._models[existing] = item
            if item["api_key_env"]:
                self._model_secret_values[item["api_key_env"]] = self._model_vars[
                    "api_key"
                ].get().strip()
            self._refresh_model_catalog()
            self.model_tree.selection_set(endpoint_id)
            self.model_tree.see(endpoint_id)
            if not silent:
                self._log(f"模型列表已更新：{item['name']} ({endpoint_id})")
            return True
        except Exception as exc:
            if silent:
                raise
            messagebox.showerror("模型配置无效", str(exc))
            return False

    def _delete_model(self) -> None:
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showinfo("删除模型", "请先在列表中选择一个模型。")
            return
        endpoint_id = selection[0]
        if len(self._models) <= 1:
            messagebox.showerror("删除模型", "至少保留一个模型。")
            return
        self._models = [model for model in self._models if model["id"] != endpoint_id]
        self._clear_model_form()
        self._refresh_model_catalog()

    def _interrupt_speech(self) -> None:
        try:
            SPEECH_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                SPEECH_CONTROL_PATH,
                {"interrupt": time.time_ns(), "source": "desktop-console"},
            )
            self._log("已发送停止语音输出指令。")
        except Exception as exc:
            messagebox.showerror("停止语音失败", str(exc))

    def _score_root(self) -> Path:
        value = self._singing_vars.get("score_root")
        return Path(value.get().strip()) if value and value.get().strip() else DEFAULT_SCORE_ROOT

    def _open_score_library(self) -> None:
        root = self._score_root()
        root.mkdir(parents=True, exist_ok=True)
        os.startfile(root)

    def _import_score(self) -> None:
        source_name = filedialog.askopenfilename(
            title="选择真实乐谱",
            filetypes=[
                ("支持的素材", "*.musicxml *.xml *.mxl *.mid *.midi *.pdf *.png *.jpg *.jpeg *.tif *.tiff *.wav *.flac *.ogg *.mp3 *.m4a *.aac"),
                ("MusicXML", "*.musicxml *.xml *.mxl"),
                ("MIDI", "*.mid *.midi"),
                ("图片或 PDF 乐谱", "*.pdf *.png *.jpg *.jpeg *.tif *.tiff"),
                ("音频", "*.wav *.flac *.ogg *.mp3 *.m4a *.aac"),
            ],
        )
        if not source_name:
            return
        source = Path(source_name)
        title = simpledialog.askstring(
            "歌曲名称",
            "请输入以后对机器人说的准确歌名：",
            initialvalue=source.stem,
            parent=self,
        )
        if not title:
            return
        safe_title = re.sub(r'[<>:"/\\|?*]+', "_", title).strip(" .")
        if not safe_title:
            messagebox.showerror("导入失败", "歌名无效。")
            return
        root = self._score_root()
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{safe_title}{source.suffix.lower()}"
        if target.exists() and not messagebox.askyesno("覆盖乐谱", f"《{title}》已存在，是否覆盖？"):
            return
        shutil.copy2(source, target)
        sidecar = source.with_suffix(".txt")
        if sidecar.exists():
            shutil.copy2(sidecar, target.with_suffix(".txt"))
        self._log(f"已导入真实乐谱：{target}")
        messagebox.showinfo(
            "导入完成",
            f"已导入《{title}》。\n\n"
            "系统会从 MusicXML/MIDI、图片/PDF 或清晰单声部音频提取旋律。\n"
            "没有嵌入歌词时，请在素材旁放置同名 UTF-8 .txt 歌词文件。",
        )

    def _search_public_scores(self) -> None:
        title = simpledialog.askstring("搜索乐谱", "输入歌曲原文名称：", parent=self)
        if not title:
            return
        from urllib.parse import quote_plus

        url = "https://www.qupu123.com/Search?keys=" + quote_plus(title.strip())
        webbrowser.open(url)
        self._log(f"已在中国曲谱网搜索：{title.strip()}")

    def _load_settings(self) -> None:
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            runtime = json.loads(LOCAL_MODEL_PATH.read_text(encoding="utf-8"))
            raw_models = settings.get("models")
            self._models = (
                [dict(item) for item in raw_models if isinstance(item, dict) and item.get("id")]
                if isinstance(raw_models, list) and raw_models
                else self._legacy_models(settings)
            )
            router_settings = settings.get("router", {})
            legacy_default = str(router_settings.get("default_route", "local"))
            self._default_model_var.set(
                str(router_settings.get("default_model_id", legacy_default))
            )
            fallback_cloud = "medium" if any(m["id"] == "medium" for m in self._models) else "cloud"
            self._cloud_model_var.set(
                str(router_settings.get("cloud_model_id", fallback_cloud))
            )
            self._refresh_model_catalog()
            self._clear_model_form()
            for key, variable in self._settings_vars.items():
                section, field = key.split(".", 1)
                value = settings.get(section, {}).get(field, "")
                if field == "extra_body":
                    variable.set(json.dumps(value or {}, ensure_ascii=False, separators=(",", ":")))
                else:
                    variable.set(str(value))
            for field, variable in self._runtime_vars.items():
                value = runtime.get(field, "")
                variable.set(str(value).lower() if isinstance(value, bool) else str(value))
            for name, variable in self._secret_vars.items():
                variable.set(_registry_env(name))
            network = settings.get("network_tools", {})
            self._network_vars["enabled"].set(str(bool(network.get("enabled", True))).lower())
            self._network_vars["search.provider"].set(
                str(network.get("search", {}).get("provider", "ddgs"))
            )
            self._network_vars["search.searxng_url"].set(
                str(network.get("search", {}).get("searxng_url", ""))
            )
            for name in ("search", "time"):
                self._network_vars[f"{name}.url"].set(str(network.get(name, {}).get("url", "")))
            singing = settings.get("singing", {})
            defaults = {
                "enabled": True,
                "score_root": str(DEFAULT_SCORE_ROOT),
                "speaker": "Gentle",
                "max_chars": 16,
                "pitch_steps": 6,
                "variance_steps": 6,
                "acoustic_steps": 12,
                "timeout_seconds": 180,
            }
            for field, variable in self._singing_vars.items():
                value = singing.get(field, defaults[field])
                variable.set(str(value).lower() if isinstance(value, bool) else str(value))
            speech_filter = settings.get("speech_filter", {})
            filter_defaults = {
                "enabled": True,
                "threshold": 0.58,
                "min_speech_seconds": 0.20,
                "min_contiguous_seconds": 0.128,
            }
            for field, variable in self._speech_filter_vars.items():
                value = speech_filter.get(field, filter_defaults[field])
                variable.set(str(value).lower() if isinstance(value, bool) else str(value))
            self._log("已载入模型和 API 配置。")
        except Exception as exc:
            messagebox.showerror("配置读取失败", str(exc))

    @staticmethod
    def _convert_number(value: str, kind: type, label: str):
        try:
            return kind(value)
        except ValueError as exc:
            raise ValueError(f"{label} 的数值无效：{value}") from exc

    def _save_settings(self, *, notify: bool = True) -> bool:
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            runtime = json.loads(LOCAL_MODEL_PATH.read_text(encoding="utf-8"))
            # If the user edited the currently visible form, Save also commits
            # it; the separate Update button remains useful when editing many.
            if self._model_vars["id"].get().strip():
                self._upsert_model(silent=True)
            if not self._models:
                raise ValueError("模型列表不能为空。")
            model_ids = {str(model["id"]) for model in self._models}
            default_model_id = self._default_model_var.get().strip()
            cloud_model_id = self._cloud_model_var.get().strip()
            if default_model_id not in model_ids:
                raise ValueError("请选择有效的默认模型。")
            if cloud_model_id not in model_ids:
                raise ValueError("请选择有效的复杂 / 云端模型。")
            settings["models"] = self._models
            for legacy_key in ("local", "medium", "cloud"):
                settings.pop(legacy_key, None)
            router_settings = settings.setdefault("router", {})
            router_settings["default_model_id"] = default_model_id
            router_settings["cloud_model_id"] = cloud_model_id
            router_settings.pop("default_route", None)
            router_settings.pop("medium_sticky_seconds", None)
            settings["router"]["confidence_threshold"] = self._convert_number(
                self._settings_vars["router.confidence_threshold"].get(), float, "路由置信度"
            )
            settings["router"]["classify_timeout_seconds"] = self._convert_number(
                self._settings_vars["router.classify_timeout_seconds"].get(), float, "分类超时"
            )
            settings["router"]["long_input_chars"] = self._convert_number(
                self._settings_vars["router.long_input_chars"].get(), int, "长输入字符数"
            )
            agent = settings.setdefault("agent", {})
            agent["history_token_budget"] = self._convert_number(
                self._settings_vars["agent.history_token_budget"].get(), int, "本地历史 token 预算"
            )
            agent.pop("medium_history_token_budget", None)
            agent["cloud_history_token_budget"] = self._convert_number(
                self._settings_vars["agent.cloud_history_token_budget"].get(), int, "云端历史 token 预算"
            )
            agent["history_max_messages"] = self._convert_number(
                self._settings_vars["agent.history_max_messages"].get(), int, "持久对话消息上限"
            )
            if not 4096 <= agent["history_token_budget"] <= 57344:
                raise ValueError("本地历史 token 预算应在 4096 到 57344 之间")
            if not 65536 <= agent["cloud_history_token_budget"] <= 950000:
                raise ValueError("云端历史 token 预算应在 65536 到 950000 之间")
            if not 20 <= agent["history_max_messages"] <= 10000:
                raise ValueError("持久对话消息上限应在 20 到 10000 之间")
            network_enabled = self._network_vars["enabled"].get().strip().lower()
            if network_enabled not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError("联网工具开关必须填写 true 或 false")
            settings.setdefault("network_tools", {})["enabled"] = network_enabled in {"true", "1", "yes"}
            settings["network_tools"].setdefault("search", {})["provider"] = (
                self._network_vars["search.provider"].get().strip() or "ddgs"
            )
            settings["network_tools"]["search"]["searxng_url"] = (
                self._network_vars["search.searxng_url"].get().strip()
            )
            for name in ("search", "time"):
                settings["network_tools"].setdefault(name, {})["url"] = self._network_vars[f"{name}.url"].get().strip()

            singing_enabled = self._singing_vars["enabled"].get().strip().lower()
            if singing_enabled not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError("歌声开关必须填写 true 或 false")
            singing = settings.setdefault("singing", {})
            singing["enabled"] = singing_enabled in {"true", "1", "yes"}
            singing["score_root"] = self._singing_vars["score_root"].get().strip() or str(DEFAULT_SCORE_ROOT)
            singing["speaker"] = self._singing_vars["speaker"].get().strip() or "Gentle"
            for field in ("max_chars", "pitch_steps", "variance_steps", "acoustic_steps"):
                singing[field] = self._convert_number(self._singing_vars[field].get(), int, field)
            singing["timeout_seconds"] = self._convert_number(
                self._singing_vars["timeout_seconds"].get(), float, "歌声渲染超时"
            )
            Path(singing["score_root"]).mkdir(parents=True, exist_ok=True)

            filter_enabled = self._speech_filter_vars["enabled"].get().strip().lower()
            if filter_enabled not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError("人声过滤开关必须填写 true 或 false")
            speech_filter = settings.setdefault("speech_filter", {})
            speech_filter["enabled"] = filter_enabled in {"true", "1", "yes"}
            speech_filter["threshold"] = self._convert_number(
                self._speech_filter_vars["threshold"].get(), float, "人声阈值"
            )
            speech_filter["min_speech_seconds"] = self._convert_number(
                self._speech_filter_vars["min_speech_seconds"].get(), float, "最短累计人声"
            )
            speech_filter["min_contiguous_seconds"] = self._convert_number(
                self._speech_filter_vars["min_contiguous_seconds"].get(), float, "最短连续人声"
            )

            for field in ("server_path", "model_path", "mmproj_path", "alias", "host", "cache_type_k", "cache_type_v"):
                runtime[field] = self._runtime_vars[field].get().strip()
            for field in (
                "port",
                "context_size",
                "gpu_layers",
                "parallel",
                "image_min_tokens",
                "image_max_tokens",
                "reasoning_budget",
            ):
                runtime[field] = self._convert_number(self._runtime_vars[field].get(), int, field)
            for field, label in (
                ("flash_attention", "Flash Attention"),
                ("mmproj_offload", "视觉模块使用GPU"),
            ):
                value = self._runtime_vars[field].get().strip().lower()
                if value not in {"true", "false", "1", "0", "yes", "no"}:
                    raise ValueError(f"{label} 必须填写 true 或 false")
                runtime[field] = value in {"true", "1", "yes"}

            for field in ("server_path", "model_path", "mmproj_path"):
                if not Path(runtime[field]).exists():
                    raise ValueError(f"文件不存在：{runtime[field]}")

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            (SETTINGS_PATH.parent / f"settings.{stamp}.bak.json").write_text(
                SETTINGS_PATH.read_text(encoding="utf-8"), encoding="utf-8"
            )
            _atomic_json(SETTINGS_PATH, settings)
            _atomic_json(LOCAL_MODEL_PATH, runtime)
            for name, value in self._model_secret_values.items():
                _set_registry_env(name, value)
            _broadcast_environment_change()
            self._log("配置已保存；旧 settings.json 已备份。")
            if notify:
                messagebox.showinfo("保存成功", "配置已保存。运行中的服务需重启后读取新参数。")
            return True
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return False

    def _save_and_restart(self) -> None:
        if self._save_settings(notify=False):
            self._run_script(
                "apply-config-and-restart.ps1",
                "正在应用配置并重启模型、路由器与 Agent",
            )

    def _run_script(self, filename: str, description: str) -> None:
        self._log(description + "…")

        def worker() -> None:
            path = SCRIPTS / filename
            try:
                returncode, stdout, stderr = _run_external_hidden(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(path),
                    ],
                    cwd=str(ROOT),
                    timeout=180,
                )
                output = (stdout + "\n" + stderr).strip()
                if returncode:
                    self._events.put(("error", f"{filename} 失败（{returncode}）\n{output}"))
                else:
                    self._events.put(("log", output or f"{filename} 已完成"))
            except Exception as exc:
                self._events.put(("error", f"{filename} 执行失败：{exc}"))
            finally:
                self._events.put(("refresh", ""))

        threading.Thread(target=worker, daemon=True).start()

    def _log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.configure(state="normal")
        self.logs.insert("end", f"[{stamp}] {text.strip()}\n")
        self.logs.see("end")
        self.logs.configure(state="disabled")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, text = self._events.get_nowait()
                if kind == "error":
                    self._log(text)
                    messagebox.showerror("操作失败", text)
                elif kind == "log":
                    self._log(text)
                elif kind == "refresh":
                    self._refresh_status_async(schedule_next=False)
                elif kind == "status":
                    self._apply_status(text)
        except queue.Empty:
            pass
        self.after(200, self._drain_events)

    def _poll_camera(self) -> None:
        try:
            mtime = CAMERA_PATH.stat().st_mtime
            if mtime != self._camera_mtime:
                self._camera_mtime = mtime
                with Image.open(CAMERA_PATH) as source:
                    image = source.convert("RGB")
                width = max(320, self.camera_label.winfo_width())
                height = max(240, self.camera_label.winfo_height())
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
                self._photo = ImageTk.PhotoImage(image)
                self.camera_label.configure(image=self._photo, text="")
        except (FileNotFoundError, OSError):
            pass
        self.after(120, self._poll_camera)

    def _poll_runtime_state(self) -> None:
        try:
            mtime = STATE_PATH.stat().st_mtime
            if mtime != self._state_mtime:
                self._state_mtime = mtime
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                self.transcript.configure(state="normal")
                self.transcript.delete("1.0", "end")
                for message in state.get("messages", []):
                    role = message.get("role", "assistant")
                    who = "你" if role == "user" else "Reachy"
                    route = message.get("route")
                    meta = f" · {route}" if route else ""
                    self.transcript.insert("end", f"{who}{meta}  {message.get('time', '')}\n", "meta")
                    self.transcript.insert("end", message.get("text", "") + "\n", role)
                self.transcript.see("end")
                self.transcript.configure(state="disabled")
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        self.after(250, self._poll_runtime_state)

    @staticmethod
    def _url_json(url: str, timeout: float = 1.5) -> dict | None:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

    def _refresh_status_async(self, *, schedule_next: bool = True) -> None:
        def worker() -> None:
            daemon = self._url_json("http://127.0.0.1:8000/api/daemon/status")
            router = self._url_json("http://127.0.0.1:8077/health")
            camera_live = False
            try:
                camera_live = time.time() - CAMERA_PATH.stat().st_mtime < 2.5
            except OSError:
                pass
            daemon_state = daemon.get("state") if daemon else "offline"
            local_ok = bool(router and router.get("local", {}).get("ok"))
            robot_text = f"机器人：{daemon_state}"
            if camera_live:
                robot_text += " · 对话运行中"
            ai_text = "AI：本地模型在线" if local_ok else "AI：离线"
            self._events.put(("status", json.dumps([robot_text, ai_text, camera_live], ensure_ascii=False)))

        threading.Thread(target=worker, daemon=True).start()
        if schedule_next:
            self.after(3000, self._refresh_status_async)

    def _apply_status(self, payload: str) -> None:
        robot_text, ai_text, camera_live = json.loads(payload)
        self.robot_status.set(robot_text)
        self.ai_status.set(ai_text)
        self.summary_var.set("摄像头实时共享中" if camera_live else "等待机器人摄像头")

if __name__ == "__main__":
    if not acquire_control_single_instance():
        ctypes.windll.user32.MessageBoxW(
            None,
            "Reachy AI 控制台已经在运行。\n请使用已经打开的窗口。",
            "Reachy AI",
            0x40,
        )
        raise SystemExit(0)
    ReachyConsole().mainloop()
