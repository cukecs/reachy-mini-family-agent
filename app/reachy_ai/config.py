from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\ReachyAI")
DEFAULT_CONFIG = ROOT / "config" / "settings.json"


@dataclass(frozen=True)
class ModelEndpoint:
    endpoint_id: str
    display_name: str
    base_url: str
    model: str
    api_key_env: str = ""
    timeout_seconds: float = 120.0
    max_tokens: int = 1024
    temperature: float = 0.7
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        value = os.environ.get(self.api_key_env, "")
        if value or os.name != "nt":
            return value
        # The desktop console stores secrets in the current user's environment.
        # A running Agent may have been started before that environment changed,
        # so read the registry as a fallback instead of requiring a reboot.
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return str(winreg.QueryValueEx(key, self.api_key_env)[0])
        except (FileNotFoundError, OSError):
            return ""


@dataclass(frozen=True)
class RouterConfig:
    confidence_threshold: float = 0.68
    classify_timeout_seconds: float = 30.0
    long_input_chars: int = 1600


@dataclass(frozen=True)
class AppConfig:
    default_model: ModelEndpoint
    cloud: ModelEndpoint
    vision_model: ModelEndpoint
    router: RouterConfig = field(default_factory=RouterConfig)
    listen_host: str = "127.0.0.1"
    listen_port: int = 8077


def _endpoint(data: dict[str, Any], endpoint_id: str) -> ModelEndpoint:
    return ModelEndpoint(
        endpoint_id=endpoint_id,
        display_name=str(data.get("name", endpoint_id)),
        base_url=str(data["base_url"]).rstrip("/"),
        model=str(data["model"]),
        api_key_env=str(data.get("api_key_env", "")),
        timeout_seconds=float(data.get("timeout_seconds", 120)),
        max_tokens=int(data.get("max_tokens", 1024)),
        temperature=float(data.get("temperature", 0.7)),
        extra_body=dict(data.get("extra_body", {})),
    )


def load_config(path: Path | str = DEFAULT_CONFIG) -> AppConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    router_raw = raw.get("router", {})
    models_raw = raw.get("models")
    if isinstance(models_raw, list) and models_raw:
        models = {
            str(item["id"]): item
            for item in models_raw
            if isinstance(item, dict) and item.get("id")
        }
        default_id = str(router_raw.get("default_model_id", ""))
        cloud_id = str(router_raw.get("cloud_model_id", ""))
        if default_id not in models:
            default_id = next(iter(models))
        if cloud_id not in models:
            cloud_id = default_id
        vision_id = str(router_raw.get("vision_model_id", ""))
        if vision_id not in models:
            vision_id = next(
                (
                    model_id
                    for model_id, item in models.items()
                    if any(
                        host in str(item.get("base_url", "")).lower()
                        for host in ("127.0.0.1", "localhost")
                    )
                ),
                default_id,
            )
        default_model = _endpoint(models[default_id], default_id)
        cloud_model = _endpoint(models[cloud_id], cloud_id)
        vision_model = _endpoint(models[vision_id], vision_id)
    else:
        # Backward-compatible migration for pre-catalog settings files.
        legacy_default_id = str(router_raw.get("default_route", "local"))
        if legacy_default_id not in {"local", "medium", "cloud"}:
            legacy_default_id = "local"
        default_model = _endpoint(raw[legacy_default_id], legacy_default_id)
        legacy_cloud_id = "medium" if "medium" in raw else "cloud"
        cloud_model = _endpoint(raw[legacy_cloud_id], legacy_cloud_id)
        vision_model = _endpoint(raw["local"], "local")

    return AppConfig(
        default_model=default_model,
        cloud=cloud_model,
        vision_model=vision_model,
        router=RouterConfig(
            confidence_threshold=float(router_raw.get("confidence_threshold", 0.68)),
            classify_timeout_seconds=float(router_raw.get("classify_timeout_seconds", 30)),
            long_input_chars=int(router_raw.get("long_input_chars", 1600)),
        ),
        listen_host=str(raw.get("listen_host", "127.0.0.1")),
        listen_port=int(raw.get("listen_port", 8077)),
    )
