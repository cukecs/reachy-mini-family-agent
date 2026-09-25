from __future__ import annotations

import json
import unittest

from reachy_ai.network_tools import (
    NetworkToolResult,
    _weather_location_candidates,
    ground_network_reply,
    select_network_tool,
)


class NetworkToolSelectionTests(unittest.TestCase):
    def test_explicit_search_uses_search_tool(self) -> None:
        selected = select_network_tool("帮我上网查一下 Reachy Mini 的最新消息")
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], "search")
        self.assertEqual(selected[1]["query"], "Reachy Mini 的最新消息")

    def test_weather_extracts_location(self) -> None:
        selected = select_network_tool("北京今天会不会下雨？")
        self.assertEqual(selected, ("weather", {"location": "北京"}))

    def test_weather_geocoding_retries_without_province(self) -> None:
        self.assertEqual(_weather_location_candidates("湖南株洲"), ["湖南株洲", "株洲"])

    def test_time_maps_known_city_to_timezone(self) -> None:
        selected = select_network_tool("东京现在几点？")
        self.assertEqual(selected[0], "time")
        self.assertEqual(selected[1]["timezone"], "Asia/Tokyo")

    def test_ordinary_chat_does_not_access_network(self) -> None:
        self.assertIsNone(select_network_tool("给我讲一个简短的笑话"))

    def test_weather_tool_directly_answers_without_an_llm(self) -> None:
        result = NetworkToolResult(
            "weather",
            json.dumps(
                {
                    "resolved_location": {"admin1": "湖南", "name": "株洲"},
                    "current": {
                        "condition": "晴",
                        "temperature_c": 25.0,
                        "apparent_temperature_c": 28.0,
                        "relative_humidity_percent": 82,
                        "wind_speed_kmh": 11.8,
                    },
                    "daily_forecast": [],
                },
                ensure_ascii=False,
            ),
            {"location": "株洲"},
        )
        reply = ground_network_reply("株洲天气怎么样", result, "")
        self.assertIn("联网查到", reply)
        self.assertIn("湖南株洲", reply)
        self.assertIn("25.0℃", reply)


if __name__ == "__main__":
    unittest.main()
