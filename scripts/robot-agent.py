from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ReachyAI\app")

from reachy_ai.robot_agent import run_once, run_robot


_mutex_handle = None


def acquire_single_instance() -> bool:
    global _mutex_handle
    if sys.platform != "win32":
        return True
    _mutex_handle = ctypes.windll.kernel32.CreateMutexW(
        None, True, "Local\\ReachyAI_AutonomousRobotAgent"
    )
    return ctypes.windll.kernel32.GetLastError() != 183


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy Mini 本地多模态对话")
    parser.add_argument("--once-text", help="不连接机器人，只测试一次模型请求")
    parser.add_argument("--image", type=Path, help="与 --once-text 一起测试视觉")
    parser.add_argument("--user", help="单次测试时载入指定用户的持久记忆")
    args = parser.parse_args()
    if args.once_text:
        run_once(args.once_text, args.image, args.user)
    else:
        if not acquire_single_instance():
            print("Reachy autonomous agent is already running.")
            return
        run_robot()


if __name__ == "__main__":
    main()
