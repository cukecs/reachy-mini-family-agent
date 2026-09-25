from __future__ import annotations

import base64
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ReachyAI\app")

from reachy_ai.config import load_config
from reachy_ai.openai_client import OpenAICompatClient, assistant_text
from reachy_ai.reachy_camera import capture_frame


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=Path, help="使用已有图像；省略时从 Reachy IPC 拍摄")
args = parser.parse_args()

image_path = args.image or capture_frame(Path(r"D:\ReachyAI\data\camera-test.jpg"))
encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
client = OpenAICompatClient(load_config().local)
response = client.chat(
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "简短描述画面中最显眼的内容。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ],
    }],
    max_tokens=96,
    temperature=0.2,
    extra={"chat_template_kwargs": {"enable_thinking": False}},
)
print(json.dumps({"image": str(image_path), "answer": assistant_text(response)}, ensure_ascii=False))
