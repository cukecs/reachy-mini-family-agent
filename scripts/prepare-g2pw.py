"""Install official g2pW assets locally; no PyTorch/Transformers required."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests

ROOT = Path(r"D:\ReachyAI\models\speech\g2pw")
ZIP_URL = "https://huggingface.co/YC-Chen/g2pW/resolve/main/G2PWModel-v2-onnx.zip"
ZIP_SHA256 = "699f3c1fd7fb0e2c2d49ed2486826fd5bff233fee7759350a91c3b49aedc4ed2"
ASSETS = {
    "vocab.txt": "https://huggingface.co/google-bert/bert-base-chinese/resolve/8f23c25b06e129b6c986331a13d8d025a92cf0ea/vocab.txt",
    "bopomofo_to_pinyin_wo_tune_dict.json": "https://raw.githubusercontent.com/GitYCC/g2pW/master/g2pw/bopomofo_to_pinyin_wo_tune_dict.json",
    "bert-base-chinese_s2t_dict.txt": "https://raw.githubusercontent.com/GitYCC/g2pW/master/g2pw/bert-base-chinese_s2t_dict.txt",
    "LICENSE": "https://raw.githubusercontent.com/GitYCC/g2pW/master/LICENCE",
}


def download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with temporary.open("wb") as destination:
            for block in response.iter_content(1024 * 1024):
                destination.write(block)
    temporary.replace(target)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "G2PWModel-v2-onnx.zip"
    if not (ROOT / "g2pw.onnx").exists():
        if not archive.exists():
            print("Downloading official g2pW model (589 MB)...", flush=True)
            download(ZIP_URL, archive)
        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != ZIP_SHA256:
            raise RuntimeError("g2pW archive SHA256 mismatch; not extracting")
        print("Archive verified; extracting model assets...", flush=True)
        allowed = {"g2pw.onnx", "POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt", "config.py", "version"}
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                name = Path(entry.filename).name
                if not entry.is_dir() and name in allowed:
                    with bundle.open(entry) as source, (ROOT / name).open("wb") as destination:
                        shutil.copyfileobj(source, destination)
    for name, url in ASSETS.items():
        if not (ROOT / name).exists():
            download(url, ROOT / name)
    if not (ROOT / "g2pw.int8.onnx").exists():
        print("Preparing CPU INT8 model (one-time operation)...", flush=True)
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(
            str(ROOT / "g2pw.onnx"), str(ROOT / "g2pw.int8.onnx"),
            weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
        )
    (ROOT / "source.json").write_text(json.dumps({
        "model_url": ZIP_URL, "archive_sha256": ZIP_SHA256,
        "assets": ASSETS, "runtime": "CPU ONNX INT8; no torch",
    }, indent=2), encoding="utf-8")
    print(f"Ready: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
