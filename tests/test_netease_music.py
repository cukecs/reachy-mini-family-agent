from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from reachy_ai.netease_music import (
    MusicCandidate,
    NeteaseMusicController,
    OriginalArtistResolution,
    choose_original_artist,
    extract_original_artist_names,
    select_exact_candidate,
)


class NeteaseMusicTests(unittest.TestCase):
    def test_artist_exact_match_wins_over_cover(self) -> None:
        candidates = [
            MusicCandidate(1, "晴天", ("翻唱歌手",)),
            MusicCandidate(186016, "晴天", ("周杰伦",), "叶惠美"),
        ]
        selected, _ = select_exact_candidate(candidates, "晴天", "周杰伦")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.song_id, 186016)

    def test_same_artist_duplicate_releases_are_safe(self) -> None:
        candidates = [
            MusicCandidate(10, "One Last Kiss", ("宇多田ヒカル",), "A"),
            MusicCandidate(11, "One Last Kiss", ("宇多田ヒカル",), "B"),
            MusicCandidate(12, "One Last Kiss", ("翻唱者",), "C"),
        ]
        selected, exact = select_exact_candidate(
            candidates, "One Last Kiss", "宇多田ヒカル"
        )
        self.assertEqual(selected.song_id, 10)
        self.assertEqual([item.song_id for item in exact], [10, 11])

    def test_ambiguous_same_title_without_artist_is_not_guessed(self) -> None:
        candidates = [
            MusicCandidate(1, "晴天", ("甲",)),
            MusicCandidate(2, "晴天", ("乙",)),
        ]
        selected, exact = select_exact_candidate(candidates, "晴天")
        self.assertIsNone(selected)
        self.assertEqual(len(exact), 2)

    def test_single_exact_title_is_safe_without_artist(self) -> None:
        candidates = [
            MusicCandidate(3, "Mojito", ("周杰伦",)),
            MusicCandidate(4, "Mojito (Live)", ("某歌手",)),
        ]
        selected, _ = select_exact_candidate(candidates, "Mojito")
        self.assertEqual(selected.song_id, 3)

    def test_title_normalization_ignores_spacing_and_punctuation(self) -> None:
        candidates = [MusicCandidate(5, "One Last Kiss", ("宇多田ヒカル",))]
        selected, _ = select_exact_candidate(
            candidates, "One Last Kiss", "宇多田ヒカル"
        )
        self.assertEqual(selected.song_id, 5)

    def test_extracts_explicit_original_and_first_singer_claims(self) -> None:
        self.assertIn(
            "周杰伦",
            extract_original_artist_names("【翻唱】晴天（原唱：周杰伦）。", "晴天"),
        )
        self.assertIn(
            "陈芬兰",
            extract_original_artist_names(
                "《月亮代表我的心》1972年由陈芬兰首唱。", "月亮代表我的心"
            ),
        )
        self.assertIn(
            "邓丽君",
            extract_original_artist_names(
                "甜蜜蜜（1979年邓丽君演唱歌曲）_百度百科", "甜蜜蜜"
            ),
        )

    def test_original_artist_requires_independent_evidence(self) -> None:
        results = [
            {
                "url": "https://a.example/song",
                "title": "晴天",
                "body": "这首歌原唱：周杰伦。",
            },
            {
                "url": "https://b.example/song",
                "title": "晴天资料",
                "body": "翻唱版本注明原唱为周杰伦。",
            },
        ]
        resolution = choose_original_artist("晴天", results)
        self.assertEqual(resolution.artist, "周杰伦")
        self.assertEqual(resolution.confidence, "high")
        self.assertEqual(resolution.evidence_count, 2)

    def test_other_song_claims_in_same_search_snippet_do_not_pollute_target(self) -> None:
        text = (
            "《晴天》原唱：周杰伦 "
            "《太阳》原唱：张杰 "
            "《相忘江湖之间》原唱：曹宇航"
        )
        self.assertEqual(
            extract_original_artist_names(text, "晴天"),
            ["周杰伦"],
        )

    def test_one_original_artist_source_is_low_confidence(self) -> None:
        resolution = choose_original_artist(
            "老歌",
            [
                {
                    "url": "https://one.example/",
                    "title": "老歌资料",
                    "body": "原唱：甲歌手。",
                }
            ],
        )
        self.assertEqual(resolution.artist, "甲歌手")
        self.assertEqual(resolution.confidence, "low")

    def test_multiple_pages_on_same_site_count_as_one_source(self) -> None:
        resolution = choose_original_artist(
            "晴天",
            [
                {
                    "url": "https://video.example/a",
                    "title": "《晴天》原唱：周杰伦",
                    "body": "",
                },
                {
                    "url": "https://video.example/b",
                    "title": "晴天翻唱",
                    "body": "《晴天》原唱：周杰伦。",
                },
            ],
        )
        self.assertEqual(resolution.artist, "周杰伦")
        self.assertEqual(resolution.confidence, "low")
        self.assertEqual(resolution.evidence_count, 1)

    def test_play_resolves_original_artist_before_netease_matching(self) -> None:
        controller = NeteaseMusicController()
        candidate = MusicCandidate(88, "老歌", ("陈芬兰",), "老专辑")
        with (
            patch.object(
                controller,
                "_cloudmusic_exe",
                return_value=Path(__file__),
            ),
            patch.object(controller, "ensure_client", return_value=True),
            patch.object(
                controller,
                "resolve_original_artist",
                return_value=OriginalArtistResolution(
                    "陈芬兰", "high", 3, ("资料一", "资料二", "资料三")
                ),
            ),
            patch.object(controller, "search", return_value=[candidate]) as search,
            patch.object(controller, "_dismiss_no_source_popup"),
            patch.object(controller, "_keep_client_in_background"),
            patch.object(controller, "_play_song_via_cdp", return_value=True) as play_cdp,
            patch.object(controller, "_no_source_visible", return_value=False),
            patch("reachy_ai.netease_music.time.sleep"),
        ):
            result = controller.play("老歌")
        self.assertEqual(result.status, "playing")
        self.assertEqual(result.artist, "陈芬兰")
        search.assert_called_once_with("老歌 陈芬兰", limit=15)
        play_cdp.assert_called_once_with(88)

    def test_unresolved_original_artist_plays_first_netease_result(self) -> None:
        controller = NeteaseMusicController()
        first = MusicCandidate(1, "老歌", ("网易云第一位",), "A")
        second = MusicCandidate(2, "老歌", ("第二位",), "B")
        with (
            patch.object(controller, "_cloudmusic_exe", return_value=Path(__file__)),
            patch.object(controller, "ensure_client", return_value=True),
            patch.object(
                controller,
                "resolve_original_artist",
                return_value=OriginalArtistResolution(None, "none"),
            ),
            patch.object(controller, "search", return_value=[first, second]) as search,
            patch.object(controller, "_dismiss_no_source_popup"),
            patch.object(controller, "_keep_client_in_background"),
            patch.object(controller, "_play_song_via_cdp", return_value=True),
            patch.object(controller, "_no_source_visible", return_value=False),
            patch("reachy_ai.netease_music.time.sleep"),
        ):
            result = controller.play("老歌")
        self.assertEqual(result.status, "playing")
        self.assertEqual(result.candidate, first)
        self.assertEqual(result.artist, "网易云第一位")
        search.assert_called_once_with("老歌", limit=15)

    def test_low_confidence_original_artist_does_not_block_playback(self) -> None:
        controller = NeteaseMusicController()
        first = MusicCandidate(3, "老歌", ("列表第一位",), "A")
        with (
            patch.object(controller, "_cloudmusic_exe", return_value=Path(__file__)),
            patch.object(controller, "ensure_client", return_value=True),
            patch.object(
                controller,
                "resolve_original_artist",
                return_value=OriginalArtistResolution("疑似原唱", "low", 1, ("资料",)),
            ),
            patch.object(controller, "search", return_value=[first]) as search,
            patch.object(controller, "_dismiss_no_source_popup"),
            patch.object(controller, "_keep_client_in_background"),
            patch.object(controller, "_play_song_via_cdp", return_value=True),
            patch.object(controller, "_no_source_visible", return_value=False),
            patch("reachy_ai.netease_music.time.sleep"),
        ):
            result = controller.play("老歌")
        self.assertEqual(result.status, "playing")
        self.assertEqual(result.candidate, first)
        search.assert_called_once_with("老歌", limit=15)

    def test_confirmed_artist_catalog_miss_falls_back_to_first_title_result(self) -> None:
        controller = NeteaseMusicController()
        fallback = MusicCandidate(4, "老歌", ("网易云首条",), "B")
        with (
            patch.object(controller, "_cloudmusic_exe", return_value=Path(__file__)),
            patch.object(controller, "ensure_client", return_value=True),
            patch.object(
                controller,
                "resolve_original_artist",
                return_value=OriginalArtistResolution("确认原唱", "high", 2, ("甲", "乙")),
            ),
            patch.object(
                controller,
                "search",
                side_effect=[
                    [MusicCandidate(9, "别的歌", ("确认原唱",), "X")],
                    [fallback],
                ],
            ) as search,
            patch.object(controller, "_dismiss_no_source_popup"),
            patch.object(controller, "_keep_client_in_background"),
            patch.object(controller, "_play_song_via_cdp", return_value=True),
            patch.object(controller, "_no_source_visible", return_value=False),
            patch("reachy_ai.netease_music.time.sleep"),
        ):
            result = controller.play("老歌")
        self.assertEqual(result.status, "playing")
        self.assertEqual(result.candidate, fallback)
        self.assertEqual(
            [call.args[0] for call in search.call_args_list],
            ["老歌 确认原唱", "老歌"],
        )

    def test_stop_uses_internal_netease_dispatch_only(self) -> None:
        controller = NeteaseMusicController()
        with (
            patch.object(controller, "_is_client_running", return_value=True),
            patch.object(controller, "_cdp_ready", return_value=True),
            patch.object(controller, "_ensure_client_runtime", return_value=True),
            patch.object(controller, "_keep_client_in_background"),
            patch.object(controller, "_cdp_evaluate", return_value=True) as evaluate,
        ):
            self.assertTrue(controller.stop())
        expression = evaluate.call_args.args[0]
        self.assertIn("playing/pause", expression)
        self.assertIn("playingList/clearCurPlayingList", expression)
        self.assertTrue(evaluate.call_args.kwargs["await_promise"])


if __name__ == "__main__":
    unittest.main()
