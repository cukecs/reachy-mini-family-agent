"""Install local CPU INT8 ONNX English phoneme model for pronunciation assessment."""
from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(r"D:\ReachyAI\models\speech\english-phoneme")
BASE_URL = "https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/resolve/main"
FILES = {
    "model_int8.onnx": f"{BASE_URL}/onnx/model_int8.onnx",
    "vocab.json": f"{BASE_URL}/vocab.json",
    "config.json": f"{BASE_URL}/config.json",
    "preprocessor_config.json": f"{BASE_URL}/preprocessor_config.json",
}


def download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as destination:
            for block in response.iter_content(1024 * 1024):
                if block:
                    destination.write(block)
    temporary.replace(target)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        target = ROOT / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"Already exists: {target}", flush=True)
            continue
        print(f"Downloading {filename}...", flush=True)
        download(url, target)
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "repo": "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX",
                "runtime": "CPU ONNX INT8 via onnxruntime; no torch",
                "files": FILES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Ready: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
