from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import ModelEndpoint


class ModelError(RuntimeError):
    pass


@dataclass
class OpenAICompatClient:
    endpoint: ModelEndpoint

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        request = urllib.request.Request(
            f"{self.endpoint.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout or self.endpoint.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ModelError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            raise ModelError(str(exc)) from exc

    def health(self, timeout: float = 3.0) -> dict[str, Any]:
        return self._request("GET", "/models", timeout=timeout)

    def chat(self, messages: list[dict[str, Any]], *, max_tokens: int | None = None,
             temperature: float = 0.4, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.endpoint.max_tokens,
            "stream": False,
        }
        if extra:
            payload.update(extra)
        return self._request("POST", "/chat/completions", payload)


def assistant_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError("模型返回结构缺少 choices[0].message.content") from exc
    if not isinstance(content, str):
        raise ModelError("模型返回 content 不是文本")
    return content
