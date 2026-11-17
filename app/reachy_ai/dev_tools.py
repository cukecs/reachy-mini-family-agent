from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(r"D:\ReachyAI")
SETTINGS_PATH = ROOT / "config" / "settings.json"
RUNTIME_STATE_PATH = ROOT / "data" / "runtime-state.json"
CAMERA_PATH = ROOT / "data" / "live-camera.jpg"
SPEECH_CONTROL_PATH = ROOT / "data" / "speech-control.json"
DEV_COMMAND_PATH = ROOT / "data" / "dev-command.json"
DEV_TOKEN_PATH = ROOT / "data" / "dev-token.txt"

_READ_ALLOWED_ROOTS = ("app", "scripts", "tests", "docs", "logs")
_WRITE_ALLOWED_ROOTS = ("app", "scripts", "tests", "docs")
_READ_ALLOWED_ROOT_FILES = {
    "README.md",
    "requirements-speech.txt",
    ".gitignore",
}
_BLOCKED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "models",
    "runtime",
    "cache",
    "dist",
    "dist-staging",
    "build",
    "data",
    "official",
    "vendor",
}


@dataclass(frozen=True)
class DevInterfaceConfig:
    enabled: bool = True
    listen_host: str = "127.0.0.1"
    listen_port: int = 8078
    token_env: str = "REACHY_DEV_TOKEN"
    allow_code_edit: bool = False
    allow_remote_control: bool = True

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "DevInterfaceConfig":
        defaults = cls()
        try:
            values = json.loads(path.read_text(encoding="utf-8")).get(
                "dev_interface", {}
            )
        except (OSError, ValueError, TypeError):
            return defaults
        if not isinstance(values, dict):
            return defaults
        port = int(values.get("listen_port", defaults.listen_port))
        return cls(
            enabled=bool(values.get("enabled", defaults.enabled)),
            listen_host=str(values.get("listen_host", defaults.listen_host)).strip()
            or defaults.listen_host,
            listen_port=max(1024, min(65535, port)),
            token_env=str(values.get("token_env", defaults.token_env)).strip()
            or defaults.token_env,
            allow_code_edit=bool(
                values.get("allow_code_edit", defaults.allow_code_edit)
            ),
            allow_remote_control=bool(
                values.get("allow_remote_control", defaults.allow_remote_control)
            ),
        )

    def resolve_token(self) -> str:
        env_val = os.environ.get(self.token_env, "").strip()
        if env_val:
            return env_val
        if os.name == "nt":
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                    reg_val = str(winreg.QueryValueEx(key, self.token_env)[0]).strip()
                    if reg_val:
                        return reg_val
            except (FileNotFoundError, OSError):
                pass
        try:
            file_val = DEV_TOKEN_PATH.read_text(encoding="utf-8").strip()
            if file_val:
                return file_val
        except OSError:
            pass
        token = secrets.token_urlsafe(24)
        DEV_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEV_TOKEN_PATH.write_text(token, encoding="utf-8")
        return token


def resolve_workspace_path(
    rel_path: str,
    *,
    root: Path = ROOT,
    allow_write: bool = False,
) -> Path:
    """Resolve a path strictly inside allowed workspace directories."""
    raw = str(rel_path or "").strip().replace("\\", "/")
    if not raw or raw in {".", "./"}:
        if allow_write:
            raise PermissionError("写入操作必须指定具体文件路径")
        return root.resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    try:
        rel = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PermissionError(f"路径超出工作区范围：{rel_path}") from exc

    parts = rel.parts
    if not parts:
        return root_resolved
    if any(part in _BLOCKED_PARTS or part.startswith(".env") for part in parts):
        raise PermissionError(f"禁止访问受保护目录或敏感文件：{rel_path}")

    top = parts[0]
    if len(parts) == 1 and not allow_write and top in _READ_ALLOWED_ROOT_FILES:
        return resolved
    if top == "config" and not allow_write:
        if len(parts) == 2 and parts[1].endswith(".example.json"):
            return resolved
        raise PermissionError("仅允许读取 config/*.example.json 公开模板")

    allowed_roots = _WRITE_ALLOWED_ROOTS if allow_write else _READ_ALLOWED_ROOTS
    if top not in allowed_roots:
        raise PermissionError(f"路径不在允许访问的目录内：{rel_path}")
    return resolved


def dev_list_files(
    subdir: str = ".",
    pattern: str = "*",
    limit: int = 120,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    target = resolve_workspace_path(subdir, root=root)
    if not target.exists():
        raise FileNotFoundError(f"目录不存在：{subdir}")
    root_resolved = root.resolve()
    results: list[dict[str, Any]] = []
    if target == root_resolved:
        search_roots = [root_resolved / name for name in _READ_ALLOWED_ROOTS if (root_resolved / name).exists()]
        for fn in sorted(_READ_ALLOWED_ROOT_FILES):
            fpath = root_resolved / fn
            if fpath.is_file() and fpath.match(pattern):
                results.append(
                    {"path": fn, "bytes": fpath.stat().st_size}
                )
    else:
        search_roots = [target]

    for base in search_roots:
        for entry in sorted(base.rglob(pattern)):
            if not entry.is_file():
                continue
            rel = entry.relative_to(root_resolved)
            if any(part in _BLOCKED_PARTS for part in rel.parts):
                continue
            results.append(
                {
                    "path": rel.as_posix(),
                    "bytes": entry.stat().st_size,
                }
            )
            if len(results) >= max(1, min(500, limit)):
                break
        if len(results) >= max(1, min(500, limit)):
            break
    return {"ok": True, "count": len(results), "files": results}


def dev_read_file(
    path: str,
    start_line: int = 1,
    end_line: int = 160,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    target = resolve_workspace_path(path, root=root)
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    start = max(1, int(start_line))
    end = min(total, max(start, int(end_line)))
    if end - start > 300:
        end = start + 300
    numbered = [
        f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1)
    ]
    return {
        "ok": True,
        "path": target.relative_to(root.resolve()).as_posix(),
        "total_lines": total,
        "start_line": start,
        "end_line": end,
        "content": "\n".join(numbered),
    }


def dev_search_code(
    query: str,
    subdir: str = "app",
    limit: int = 40,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    needle = str(query or "").strip()
    if not needle:
        raise ValueError("query 不能为空")
    target = resolve_workspace_path(subdir, root=root)
    root_resolved = root.resolve()
    try:
        regex = re.compile(needle, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(needle), re.IGNORECASE)

    matches: list[dict[str, Any]] = []
    files = [target] if target.is_file() else sorted(target.rglob("*.py"))
    if target.is_dir() and not files:
        files = sorted(target.rglob("*"))
    for file_path in files:
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root_resolved)
        if any(part in _BLOCKED_PARTS for part in rel.parts):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(
                    {
                        "path": rel.as_posix(),
                        "line": line_no,
                        "text": line[:200],
                    }
                )
                if len(matches) >= max(1, min(100, limit)):
                    return {"ok": True, "query": needle, "matches": matches, "truncated": True}
    return {"ok": True, "query": needle, "matches": matches, "truncated": False}


def dev_edit_file(
    path: str,
    target: str,
    replacement: str,
    *,
    allow_code_edit: bool = False,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not allow_code_edit:
        raise PermissionError("当前 dev_interface.allow_code_edit 为 false，禁止直接修改源码")
    file_path = resolve_workspace_path(path, root=root, allow_write=True)
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not target:
        raise ValueError("target 不能为空")
    original = file_path.read_text(encoding="utf-8")
    count = original.count(target)
    if count == 0:
        raise ValueError("未在文件中找到完全匹配的 target 文本")
    updated = original.replace(target, replacement)
    file_path.write_text(updated, encoding="utf-8")
    return {
        "ok": True,
        "path": file_path.relative_to(root.resolve()).as_posix(),
        "replacements": count,
    }


def dev_git_status(*, root: Path = ROOT) -> dict[str, Any]:
    status_proc = subprocess.run(
        ["git", "-C", str(root), "status", "-sb"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    log_proc = subprocess.run(
        ["git", "-C", str(root), "log", "-n", "5", "--oneline"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    return {
        "ok": status_proc.returncode == 0,
        "status": status_proc.stdout.strip(),
        "recent_commits": log_proc.stdout.strip().splitlines(),
        "error": status_proc.stderr.strip() or None,
    }


def dev_run_tests(pattern: str = "", *, root: Path = ROOT) -> dict[str, Any]:
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else "python"
    cmd = [python_exe, "-m", "unittest", "discover", "-s", str(root / "tests"), "-q"]
    if pattern.strip():
        cmd.extend(["-p", pattern.strip()])
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": combined[-4000:],
    }


def _probe_json(url: str, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def dev_diagnostics(*, root: Path = ROOT) -> dict[str, Any]:
    mem = psutil.virtual_memory()
    processes: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            name = str(proc.info.get("name") or "")
            if any(
                marker in cmdline.lower() or marker in name.lower()
                for marker in ("reachy", "llama-server", "robot-agent", "desktop-console")
            ):
                rss_mb = round((proc.info["memory_info"].rss if proc.info.get("memory_info") else 0) / (1024 * 1024), 1)
                processes.append({"pid": proc.info["pid"], "name": name, "rss_mb": rss_mb, "cmd": cmdline[:160]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    state_summary: dict[str, Any] = {}
    try:
        state_raw = json.loads((root / "data" / "runtime-state.json").read_text(encoding="utf-8"))
        state_summary = {
            "status": state_raw.get("status"),
            "detail": state_raw.get("detail"),
            "updated_at": state_raw.get("updated_at"),
            "last_messages": state_raw.get("messages", [])[-4:],
        }
    except (OSError, ValueError, TypeError):
        pass

    camera_age = None
    cam_path = root / "data" / "live-camera.jpg"
    try:
        camera_age = round(max(0.0, time.time() - cam_path.stat().st_mtime), 2)
    except OSError:
        pass

    return {
        "ok": True,
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "ram_used_gb": round((mem.total - mem.available) / (1024 ** 3), 2),
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "camera_frame_age_seconds": camera_age,
        "daemon": _probe_json("http://127.0.0.1:8000/api/daemon/status"),
        "router": _probe_json("http://127.0.0.1:8077/health"),
        "runtime_state": state_summary,
        "processes": processes,
    }


def dispatch_dev_command(
    action: str,
    text: str = "",
    *,
    root: Path = ROOT,
    wait_seconds: float = 12.0,
) -> dict[str, Any]:
    """Send a remote control/chat command to the running robot agent."""
    clean_action = str(action or "").strip().lower()
    if clean_action == "stop_speech":
        path = root / "data" / "speech-control.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"interrupt": str(time.time())}), encoding="utf-8")
        tmp.replace(path)
        return {"ok": True, "action": "stop_speech"}

    if clean_action not in {"say", "ask", "wake", "sleep"}:
        raise ValueError(f"不支持的远程动作：{action}")
    if clean_action in {"say", "ask"} and not str(text).strip():
        raise ValueError("say / ask 动作需要提供 text")

    cmd_path = root / "data" / "dev-command.json"
    cmd_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_id = f"{int(time.time() * 1000)}"
    payload = {
        "id": cmd_id,
        "action": clean_action,
        "text": str(text).strip(),
        "status": "pending",
        "created_at": time.time(),
    }
    tmp = cmd_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cmd_path)

    deadline = time.monotonic() + max(1.0, wait_seconds)
    while time.monotonic() < deadline:
        time.sleep(0.15)
        try:
            state = json.loads(cmd_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if state.get("id") == cmd_id and state.get("status") in {"done", "error"}:
            return state
    return {"ok": False, "id": cmd_id, "status": "timeout", "action": clean_action}
