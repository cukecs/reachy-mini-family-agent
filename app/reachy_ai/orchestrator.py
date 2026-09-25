from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .openai_client import ModelError, OpenAICompatClient
from .router import RouteDecision, SmartRouter


@dataclass
class Orchestrator:
    config: AppConfig

    def __post_init__(self) -> None:
        self.local = OpenAICompatClient(self.config.default_model)
        self.cloud = OpenAICompatClient(self.config.cloud)
        self.vision = OpenAICompatClient(self.config.vision_model)
        self.router = SmartRouter(self.local, self.config.router)

    def route(self, messages: list[dict[str, Any]], force_route: str | None = None) -> RouteDecision:
        return self.router.decide(messages, force_route)

    def chat(self, payload: dict[str, Any]) -> tuple[dict[str, Any], RouteDecision]:
        payload = copy.deepcopy(payload)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 必须是非空数组")
        forced_route = payload.pop("route", None)
        internal_vision = forced_route == "vision"
        decision = (
            RouteDecision("local", 1.0, "内部本机视觉前端", "vision")
            if internal_vision
            else self.route(messages, forced_route)
        )
        clients = {
            "local": self.local,
            "cloud": self.cloud,
        }
        chosen = self.vision if internal_vision else clients[decision.route]
        forwarded = payload
        forwarded.pop("messages", None)
        forwarded.pop("model", None)
        max_tokens = forwarded.pop("max_tokens", None)
        temperature = float(forwarded.pop("temperature", chosen.endpoint.temperature))
        forwarded.pop("stream", None)
        caller_extra = copy.deepcopy(forwarded)

        def call(client: OpenAICompatClient) -> dict[str, Any]:
            request_extra = copy.deepcopy(client.endpoint.extra_body)
            request_extra.update(caller_extra)
            return client.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra=request_extra,
            )

        try:
            response = call(chosen)
        except ModelError as primary_error:
            if internal_vision:
                raise primary_error
            if decision.route == "cloud":
                try:
                    chosen = self.local
                    response = call(self.local)
                    decision = RouteDecision(
                        "local", 0.50, "复杂/云端模型不可达，降级到默认模型", "failover"
                    )
                except ModelError:
                    raise primary_error
            else:
                try:
                    chosen = self.cloud
                    response = call(self.cloud)
                    decision = RouteDecision(
                        "cloud", 0.50, "默认模型不可达，改用复杂/云端模型", "failover"
                    )
                except ModelError:
                    raise primary_error
        metadata = decision.to_dict()
        metadata["model_id"] = chosen.endpoint.endpoint_id
        metadata["model_name"] = chosen.endpoint.display_name
        response["reachy_router"] = metadata
        return response, decision

    def health(self, *, deep: bool = False) -> dict[str, Any]:
        if not deep:
            return {
                "local": {
                    "ok": True,
                    "probed": False,
                    "reason": "路由器在线；浅检查不会调用模型 API",
                    "model_id": self.config.default_model.endpoint_id,
                    "base_url": self.config.default_model.base_url,
                },
                "cloud": {
                    "probed": False,
                    "reason": "浅健康检查不会请求或唤醒复杂/云端模型",
                    "model_id": self.config.cloud.endpoint_id,
                    "base_url": self.config.cloud.base_url,
                },
            }
        result: dict[str, Any] = {}
        clients = (("local", self.local), ("cloud", self.cloud), ("vision", self.vision))
        for name, client in clients:
            try:
                models = client.health()
                result[name] = {"ok": True, "models": models.get("data", [])}
            except ModelError as exc:
                result[name] = {"ok": False, "error": str(exc)}
        return result
