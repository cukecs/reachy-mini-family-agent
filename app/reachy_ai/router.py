from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .config import RouterConfig
from .openai_client import ModelError, OpenAICompatClient, assistant_text


RouteName = Literal["local", "cloud"]


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    confidence: float
    reason: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXPLICIT_MEDIUM = re.compile(
    r"(用|调用|切到).{0,8}(大模型|后端|中等模型|R9700)|"
    r"(深入|深度|严谨).{0,8}(分析|研究|推理)|全面调研|完整审计",
    re.IGNORECASE,
)
EXPLICIT_CLOUD = re.compile(
    r"(用|调用|切到|交给).{0,8}(DeepSeek|深度求索)|DeepSeek.{0,8}(回答|分析|处理)",
    re.IGNORECASE,
)
EXPLICIT_LOCAL = re.compile(
    r"(用|调用|切到|交给).{0,8}(本地模型|本机模型|9B模型)|"
    r"(?:本地模型|本机模型|9B模型).{0,8}(回答|处理)",
    re.IGNORECASE,
)
HIGH_STAKES = re.compile(
    r"诊断|处方|剂量|急救|法律意见|诉讼|合同审查|投资建议|资产配置|税务筹划|"
    r"medical diagnosis|dosage|legal advice|investment advice",
    re.IGNORECASE,
)
COMPLEX_MARKERS = re.compile(
    r"架构设计|根因分析|性能调优|迁移方案|安全审计|比较.{0,10}(方案|模型)|"
    r"证明|推导|长文|论文|代码库|多文件|research|benchmark|architecture|debug",
    re.IGNORECASE,
)
SIMPLE_LOCAL = re.compile(
    r"^(你好|您好|早上好|晚上好|谢谢|再见|你是谁|几点了|讲个笑话|"
    r"hello|hi|thanks|thank you)[！!。.?？ ]*$",
    re.IGNORECASE,
)


def _message_text(messages: list[dict[str, Any]]) -> str:
    user_messages = [message for message in messages if message.get("role") == "user"]
    selected_messages = user_messages[-1:] if user_messages else messages[-1:]
    chunks: list[str] = []
    for message in selected_messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
    return "\n".join(chunks).strip()


def _contains_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and item.get("type") in {"image_url", "input_image"}
            for item in content
        ):
            return True
    return False


class SmartRouter:
    def __init__(self, local_client: OpenAICompatClient, config: RouterConfig):
        self.local_client = local_client
        self.config = config
    def decide(self, messages: list[dict[str, Any]], force_route: str | None = None) -> RouteDecision:
        if force_route in {"deepseek", "medium"}:
            force_route = "cloud"
        if force_route in {"local", "cloud"}:
            return RouteDecision(force_route, 1.0, "调用方强制指定", "override")

        text = _message_text(messages)
        has_image = _contains_image(messages)
        if EXPLICIT_CLOUD.search(text):
            return RouteDecision("cloud", 0.99, "用户明确要求 DeepSeek", "hard-rule")
        if EXPLICIT_LOCAL.search(text):
            return RouteDecision("local", 0.99, "用户明确要求默认模型", "hard-rule")
        if EXPLICIT_MEDIUM.search(text):
            return RouteDecision("cloud", 0.99, "用户明确要求强模型或深度分析", "hard-rule")
        if HIGH_STAKES.search(text):
            return RouteDecision("cloud", 0.98, "高风险领域需要更强推理", "hard-rule")
        if len(text) >= self.config.long_input_chars:
            return RouteDecision("cloud", 0.92, "输入较长，超过默认模型路由阈值", "hard-rule")
        if text.count("```") >= 4 or text.count("\n") >= 35:
            return RouteDecision("cloud", 0.90, "多段代码或长结构化输入", "hard-rule")
        if SIMPLE_LOCAL.match(text):
            return RouteDecision("local", 0.99, "日常实时对话使用默认模型", "hard-rule")

        marker_count = len(COMPLEX_MARKERS.findall(text))
        if marker_count >= 2:
            return RouteDecision("cloud", 0.88, "命中多个复杂任务特征", "hard-rule")

        # Robot conversation must feel immediate. Short text without a known
        # complex/risk marker is ordinary dialogue; running a second 4B call
        # merely to classify it used to double every turn's latency.
        if len(text) <= 180 and marker_count == 0:
            return RouteDecision("local", 0.96, "日常实时对话使用默认模型", "hard-rule")

        # 对边界任务才调用 4B 分类器，减少普通对话的额外首字延迟。
        return self._classify_with_local(text, has_image, marker_count)

    def _classify_with_local(self, text: str, has_image: bool, marker_count: int) -> RouteDecision:
        prompt = (
            "你是机器人请求路由器。只输出一个JSON对象，不要markdown。字段："
            'route只能是"local"或"cloud"；confidence为0到1；reason不超过30个汉字。'
            "local是用户选择的默认模型，适合日常聊天、简单问答、单步工具和普通看图；"
            "cloud是用户选择的复杂/云端模型，适合复杂推理、长代码、多约束规划、严肃专业问题、深度视觉分析。"
            "拿不准或完成任务需要多阶段验证时选cloud。\n"
            f"是否包含图像：{has_image}\n请求：{text[:2200]}"
        )
        try:
            response = self.local_client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=160,
                temperature=0.0,
                extra={
                    "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            parsed = json.loads(assistant_text(response).strip())
            route = parsed.get("route")
            confidence = float(parsed.get("confidence", 0))
            reason = str(parsed.get("reason", "本地分类器判断"))[:80]
            if route not in {"local", "cloud"} or not 0 <= confidence <= 1:
                raise ValueError("invalid route response")
            if confidence < self.config.confidence_threshold:
                return RouteDecision("cloud", confidence, "分类置信度不足，安全升级", "validator")
            return RouteDecision(route, confidence, reason, "local-classifier")
        except (ModelError, ValueError, TypeError, json.JSONDecodeError):
            # 本地模型未启动时仍能路由；有复杂迹象则升级，否则本地处理。
            if marker_count or len(text) > 500 or has_image:
                return RouteDecision("cloud", 0.60, "分类器不可用且任务有复杂特征", "fallback")
            return RouteDecision("local", 0.58, "分类器不可用，保留本地处理", "fallback")
