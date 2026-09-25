from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path(r"D:\ReachyAI\config\settings.json")

SEARCH_TRIGGER = re.compile(
    r"搜索|搜一下|搜搜|查一下|查查|上网|网上查|检索|最新|新闻|热搜|今日.{0,8}(消息|事件)|"
    r"现在的.{0,12}(价格|股价|汇率|赛程|排名|政策|规定)|"
    r"search (?:the )?web|look (?:it|this|that) up|latest|today'?s news|current (?:price|news|events)",
    re.IGNORECASE,
)
WEATHER_TRIGGER = re.compile(
    r"天气|气温|温度|下雨|降雨|下雪|台风|weather|forecast|temperature",
    re.IGNORECASE,
)
TIME_TRIGGER = re.compile(
    r"几点|现在.{0,5}时间|当地时间|时差|what time|time in|time difference",
    re.IGNORECASE,
)

DEFAULT_TOOLS = {
    "search": {
        "provider": "ddgs",
        "searxng_url": "https://search.mectov.my.id/",
        "url": "https://pollen-robotics-reachy-mini-search-tool.hf.space/gradio_api/mcp/",
        "remote_name": "reachy_mini_search_tool_search_web",
    },
    "weather": {
        "url": "https://pollen-robotics-reachy-mini-weather-tool.hf.space/gradio_api/mcp/",
        "remote_name": "reachy_mini_weather_tool_get_weather",
    },
    "time": {
        "url": "https://pollen-robotics-reachy-mini-time-tool.hf.space/gradio_api/mcp/",
        "remote_name": "reachy_mini_time_tool_get_time",
    },
}

TIMEZONES = {
    "北京": "Asia/Shanghai",
    "上海": "Asia/Shanghai",
    "中国": "Asia/Shanghai",
    "东京": "Asia/Tokyo",
    "日本": "Asia/Tokyo",
    "纽约": "America/New_York",
    "洛杉矶": "America/Los_Angeles",
    "伦敦": "Europe/London",
    "巴黎": "Europe/Paris",
    "悉尼": "Australia/Sydney",
    "新加坡": "Asia/Singapore",
    "香港": "Asia/Hong_Kong",
    "台北": "Asia/Taipei",
}

WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    56: "轻微冻雨",
    57: "较强冻雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻微冻雨",
    67: "较强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴并伴有小冰雹",
    99: "雷暴并伴有大冰雹",
}

_CHINESE_PROVINCES = (
    "北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|"
    "河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|"
    "宁夏|新疆|香港|澳门"
)


@dataclass(frozen=True)
class NetworkToolResult:
    tool: str
    text: str
    arguments: dict[str, Any]


def _configuration() -> tuple[bool, dict[str, dict[str, str]]]:
    tools = {name: dict(value) for name, value in DEFAULT_TOOLS.items()}
    enabled = True
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        configured = raw.get("network_tools", {})
        enabled = bool(configured.get("enabled", True))
        for name in tools:
            item = configured.get(name, {})
            if isinstance(item, dict):
                for field in ("provider", "searxng_url", "url", "remote_name"):
                    if item.get(field):
                        tools[name][field] = str(item[field])
    except (OSError, ValueError, TypeError):
        pass
    return enabled, tools


@asynccontextmanager
async def _mcp_session(url: str):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(timeout=40.0, follow_redirects=False) as client:
        async with streamable_http_client(url, http_client=client) as transport:
            read_stream, write_stream = transport[:2]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


async def _call_mcp(url: str, remote_name: str, arguments: dict[str, Any]) -> str:
    async with _mcp_session(url) as session:
        result = await session.call_tool(remote_name, arguments=arguments)
    if bool(getattr(result, "isError", False)):
        raise RuntimeError("remote tool returned an error")
    chunks = [
        str(getattr(block, "text", "")).strip()
        for block in getattr(result, "content", [])
        if str(getattr(block, "text", "")).strip()
    ]
    if chunks:
        return "\n".join(chunks)
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    raise RuntimeError("remote tool returned no readable content")


def _weather_location_candidates(location: str) -> list[str]:
    cleaned = re.sub(r"^(?:中国|中华人民共和国)", "", location.strip())
    candidates = [cleaned]
    without_province = re.sub(rf"^(?:{_CHINESE_PROVINCES})(?:省|市|自治区|特别行政区)?", "", cleaned)
    if without_province and without_province != cleaned:
        candidates.append(without_province)
    without_suffix = re.sub(r"(?:市|县|区)$", "", without_province)
    if without_suffix and without_suffix not in candidates:
        candidates.append(without_suffix)
    return candidates


async def _open_meteo_weather(location: str) -> str:
    """Return a current + seven-day forecast with Chinese place-name support."""
    import httpx

    place = None
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for candidate in _weather_location_candidates(location):
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": candidate, "count": 5, "language": "zh", "format": "json"},
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            if results:
                chinese_results = [item for item in results if item.get("country_code") == "CN"]
                place = (chinese_results or results)[0]
                break
        if place is None:
            raise RuntimeError(f"天气服务找不到地点：{location}")

        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "timezone": place.get("timezone") or "auto",
                "forecast_days": 7,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
            },
        )
        response.raise_for_status()
        forecast = response.json()

    current = forecast.get("current") or {}
    daily = forecast.get("daily") or {}
    days = []
    dates = daily.get("time") or []
    for index, date in enumerate(dates):
        code = (daily.get("weather_code") or [None] * len(dates))[index]
        days.append(
            {
                "date": date,
                "condition": WEATHER_CODES.get(code, f"天气代码 {code}"),
                "temperature_min_c": (daily.get("temperature_2m_min") or [None] * len(dates))[index],
                "temperature_max_c": (daily.get("temperature_2m_max") or [None] * len(dates))[index],
                "precipitation_probability_percent": (
                    daily.get("precipitation_probability_max") or [None] * len(dates)
                )[index],
                "precipitation_mm": (daily.get("precipitation_sum") or [None] * len(dates))[index],
            }
        )

    payload = {
        "provider": "Open-Meteo",
        "resolved_location": {
            "name": place.get("name"),
            "admin1": place.get("admin1"),
            "country": place.get("country"),
            "timezone": place.get("timezone"),
        },
        "current": {
            "observation_time": current.get("time"),
            "condition": WEATHER_CODES.get(current.get("weather_code"), "未知"),
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "relative_humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
        },
        "daily_forecast": days,
    }
    return json.dumps(payload, ensure_ascii=False)


async def _searxng_search(url: str, query: str, max_results: int) -> str:
    import httpx

    endpoint = url.rstrip("/") + "/search"
    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": "ReachyAI/1.0"},
    ) as client:
        response = await client.get(
            endpoint,
            params={
                "q": query,
                "format": "json",
                # "all" is more reliable than zh-CN on public instances and
                # still returns Chinese results for Chinese queries.
                "language": "all",
                "safesearch": 1,
                "categories": "general",
            },
        )
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in (payload.get("results") or [])[: max(1, min(max_results, 10))]:
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "snippet": str(item.get("content") or item.get("snippet") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "engine": item.get("engine"),
                "published_date": item.get("publishedDate") or item.get("published_date"),
            }
        )
    results = [item for item in results if item["title"] and item["url"]]
    if not results:
        raise RuntimeError("SearXNG 没有返回可用搜索结果")
    return json.dumps(
        {"provider": "SearXNG", "query": query, "results": results},
        ensure_ascii=False,
    )


def _ddgs_search(query: str, max_results: int) -> str:
    from ddgs import DDGS

    raw_results = DDGS(timeout=15).text(
        query,
        region="cn-zh",
        safesearch="moderate",
        max_results=max(1, min(max_results, 10)),
    )
    results = [
        {
            "title": str(item.get("title") or "").strip(),
            "snippet": str(item.get("body") or item.get("content") or "").strip(),
            "url": str(item.get("href") or item.get("url") or "").strip(),
            "engine": item.get("source"),
        }
        for item in raw_results
    ]
    results = [item for item in results if item["title"] and item["url"]]
    if not results:
        raise RuntimeError("DDGS 没有返回可用搜索结果")
    return json.dumps(
        {"provider": "DDGS", "query": query, "results": results},
        ensure_ascii=False,
    )


def _weather_location(text: str) -> str:
    cleaned = re.sub(
        r"请|帮我|查一下|查查|看看|今天|明天|后天|现在|目前|的|天气|气温|温度|"
        r"怎么样|如何|多少|会不会|下雨|降雨|forecast|weather|temperature|[?？!！。]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    location = re.sub(r"\s+", " ", cleaned).strip()
    return location[:80] or "上海"


def _time_arguments(text: str) -> dict[str, str]:
    found = [timezone for name, timezone in TIMEZONES.items() if name in text]
    if len(found) >= 2:
        return {"timezone": found[0], "compare_timezone": found[1]}
    return {
        "timezone": found[0] if found else "Asia/Shanghai",
        "compare_timezone": "",
    }


def _search_query(text: str) -> str:
    cleaned = re.sub(
        r"请|麻烦|帮我|给我|上网|网上|联网|搜索|搜一下|搜搜|查一下|查查|检索|看一下|看看",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ，,。！？!?：:")[:500] or text.strip()[:500]


def select_network_tool(text: str) -> tuple[str, dict[str, Any]] | None:
    if WEATHER_TRIGGER.search(text):
        return "weather", {"location": _weather_location(text)}
    if TIME_TRIGGER.search(text):
        return "time", _time_arguments(text)
    if SEARCH_TRIGGER.search(text):
        return "search", {"query": _search_query(text), "max_results": 5}
    return None


def maybe_use_network_tool(text: str) -> NetworkToolResult | None:
    enabled, tools = _configuration()
    selected = select_network_tool(text) if enabled else None
    if selected is None:
        return None
    name, arguments = selected
    if name == "weather":
        # The official hosted weather MCP currently fails to resolve many Chinese
        # city names (including Zhuzhou). Open-Meteo supplies geocoding and a
        # dated seven-day forecast, so relative requests such as "tomorrow" can
        # be answered from real data instead of being guessed by the LLM.
        result = asyncio.run(_open_meteo_weather(str(arguments["location"])))
        return NetworkToolResult(name, result, arguments)
    spec = tools[name]
    if name == "search":
        provider = spec.get("provider", "ddgs").lower()
        if provider == "ddgs":
            try:
                result = _ddgs_search(
                    str(arguments["query"]), int(arguments.get("max_results", 5))
                )
                return NetworkToolResult(name, result, arguments)
            except Exception as exc:
                print(f"DDGS failed, falling back to SearXNG: {exc}")
        if provider in {"ddgs", "searxng"} and spec.get("searxng_url"):
            try:
                result = asyncio.run(
                    _searxng_search(
                        spec["searxng_url"],
                        str(arguments["query"]),
                        int(arguments.get("max_results", 5)),
                    )
                )
                return NetworkToolResult(name, result, arguments)
            except Exception as exc:
                # Public SearXNG instances can rate-limit or temporarily disappear.
                # Keep the existing hosted MCP as an automatic fallback so the robot
                # does not lose web access when that happens.
                print(f"SearXNG failed, falling back to search MCP: {exc}")
    result = asyncio.run(_call_mcp(spec["url"], spec["remote_name"], arguments))
    return NetworkToolResult(name, result, arguments)


def ground_network_reply(
    user_text: str, result: NetworkToolResult, model_reply: str
) -> str:
    """Prevent an LLM from denying a tool call the Agent already completed."""
    denial = re.search(
        r"(?:无法|不能|不会|不具备|没有).{0,10}(?:联网|上网|访问网络|实时信息)|"
        r"(?:联网|网络).{0,8}(?:不可用|受限)",
        model_reply,
        re.IGNORECASE,
    )
    try:
        payload = json.loads(result.text)
    except (TypeError, ValueError):
        payload = None

    if result.tool == "weather" and isinstance(payload, dict):
        place = payload.get("resolved_location") or {}
        current = payload.get("current") or {}
        forecasts = payload.get("daily_forecast") or []
        target = forecasts[1] if "明天" in user_text and len(forecasts) > 1 else (
            forecasts[2] if "后天" in user_text and len(forecasts) > 2 else None
        )
        location = "".join(
            str(place.get(key) or "") for key in ("admin1", "name")
        ) or str(result.arguments.get("location") or "当地")
        if target:
            return (
                f"我刚刚联网查到：{location}{target.get('date', '')}"
                f"{target.get('condition', '')}，最低{target.get('temperature_min_c')}℃，"
                f"最高{target.get('temperature_max_c')}℃，降水概率"
                f"{target.get('precipitation_probability_percent')}%。"
            )
        return (
            f"我刚刚联网查到：{location}当前{current.get('condition', '')}，"
            f"{current.get('temperature_c')}℃，体感{current.get('apparent_temperature_c')}℃，"
            f"湿度{current.get('relative_humidity_percent')}%，"
            f"风速{current.get('wind_speed_kmh')}公里每小时。"
        )

    capability_question = bool(
        re.search(r"(?:会|能|可以|有没有).{0,8}(?:联网|上网|搜索)|联网能力", user_text)
    )
    if result.tool == "search" and isinstance(payload, dict) and (
        denial or capability_question or not model_reply.strip()
    ):
        items = payload.get("results") or []
        summaries = []
        for item in items[:5]:
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            url = str(item.get("url") or "").strip()
            if title:
                summaries.append(
                    title + (f"：{snippet[:180]}" if snippet else "") + (f"（{url}）" if url else "")
                )
        provider = str(payload.get("provider") or "搜索服务")
        evidence = "\n".join(f"{index + 1}. {value}" for index, value in enumerate(summaries))
        prefix = "可以联网。" if capability_question else ""
        return f"{prefix}我刚才已通过{provider}完成实时搜索。" + (
            f"\n{evidence}" if evidence else ""
        )
    if result.tool == "time" and not model_reply.strip():
        return "我刚刚联网查询到：" + result.text[:1200]
    if denial:
        return "联网工具已经执行成功。" + model_reply.replace(denial.group(0), "已联网查询")
    return model_reply
