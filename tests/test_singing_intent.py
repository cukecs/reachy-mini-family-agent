import unittest

from reachy_ai.singing import SingingEngine
from reachy_ai.robot_agent import (
    PITCH_LISTEN_REQUEST,
    SONG_GUESS_REQUEST,
    SINGING_META_QUESTION,
    RobotBrain,
    extract_promised_song_title,
    extract_requested_song_artist,
    extract_requested_song_title,
    is_valid_requested_song_title,
    strip_singing_tags,
    strip_song_tags,
)


class SingingIntentTests(unittest.TestCase):
    def test_bracketed_existing_song(self):
        self.assertEqual(
            extract_requested_song_title("请唱《七子之歌·澳门》给我听"),
            "七子之歌·澳门",
        )

    def test_plain_existing_song(self):
        self.assertEqual(extract_requested_song_title("唱一首小星星吧"), "小星星")

    def test_play_existing_song(self):
        self.assertEqual(extract_requested_song_title("播放《小星星》"), "小星星")
        self.assertEqual(extract_requested_song_title("放一首小星星"), "小星星")

    def test_artist_and_title_are_parsed_separately(self):
        text = "播放周杰伦的《晴天》"
        self.assertEqual(extract_requested_song_title(text), "晴天")
        self.assertEqual(extract_requested_song_artist(text, "晴天"), "周杰伦")
        text = "唱周杰伦的晴天"
        self.assertEqual(extract_requested_song_title(text), "晴天")
        self.assertEqual(extract_requested_song_artist(text, "晴天"), "周杰伦")

    def test_asr_particle_is_not_part_of_song_title(self):
        self.assertEqual(extract_requested_song_title("请唱一首小星星来"), "小星星")

    def test_generic_request_is_not_a_title(self):
        self.assertIsNone(extract_requested_song_title("给我唱一首歌吧"))
        self.assertIsNone(extract_requested_song_title("唱之歌"))

    def test_singing_mechanism_question_is_not_a_song_title(self):
        text = "那如果我想让你唱歌该怎么实现呢？因为我用曲库的形式让你唱。"
        self.assertIsNotNone(SINGING_META_QUESTION.search(text))
        self.assertIsNone(extract_requested_song_title(text))
        self.assertFalse(is_valid_requested_song_title("歌该怎么实现"))

    def test_singing_mechanism_question_never_starts_singing(self):
        reply, route = RobotBrain().answer(
            "那如果我想让你唱歌该怎么实现呢？因为我用曲库的形式让你唱。"
        )
        self.assertEqual(route["route"], "singing-help")
        self.assertNotIn("singing", route)
        self.assertNotIn("singing_title", route)
        self.assertIn("网易云", reply)

    def test_how_to_sing_named_song_is_a_question_not_playback(self):
        text = "《小星星》应该怎么唱？"
        self.assertIsNotNone(SINGING_META_QUESTION.search(text))
        self.assertIsNone(extract_requested_song_title(text))

    def test_song_identification_is_not_an_outbound_singing_request(self):
        examples = (
            "你猜我唱的是什么歌",
            "猜唱的什么歌？",
            "听听我刚才唱的是哪首歌",
            "你能识别这段哼唱的歌名吗",
        )
        for text in examples:
            self.assertIsNotNone(SONG_GUESS_REQUEST.search(text), text)
            self.assertIsNone(extract_requested_song_title(text), text)
            self.assertIsNotNone(PITCH_LISTEN_REQUEST.search(text), text)

    def test_question_words_are_never_valid_song_titles(self):
        for title in ("什么歌", "的什么歌", "哪首歌", "这是什么歌", "歌名"):
            self.assertFalse(is_valid_requested_song_title(title), title)

    def test_assistant_song_promise_supplies_missing_route_title(self):
        self.assertEqual(
            extract_promised_song_title("好呀，我来唱《七子之歌·澳门》。"),
            "七子之歌·澳门",
        )

    def test_pronoun_song_request_uses_recent_named_song(self):
        brain = RobotBrain()
        brain.record_external_turn(
            "我刚才唱的是什么？", "你唱的是《七子之歌·澳门》。"
        )
        self.assertEqual(brain._recent_song_title(), "七子之歌·澳门")

    def test_named_song_is_routed_without_waiting_for_an_llm(self):
        reply, route = RobotBrain().answer("唱一遍《七子之歌·澳门》")
        self.assertIn("网易云", reply)
        self.assertEqual(route["route"], "netease-music")
        self.assertEqual(route["music_title"], "七子之歌·澳门")
        self.assertNotIn("singing", route)
        self.assertNotIn("score_download", route)

    def test_named_artist_song_is_routed_to_netease(self):
        reply, route = RobotBrain().answer("播放周杰伦的《晴天》")
        self.assertIn("网易云", reply)
        self.assertEqual(route["route"], "netease-music")
        self.assertEqual(route["music_title"], "晴天")
        self.assertEqual(route["music_artist"], "周杰伦")

    def test_song_and_original_tags_are_separate(self):
        cleaned, title = strip_song_tags("我来唱。[[SONG:小星星]]")
        self.assertEqual(cleaned, "我来唱。")
        self.assertEqual(title, "小星星")
        cleaned, lyrics = strip_singing_tags("这是原创。[[SING:晚风轻轻吹过山岗]]")
        self.assertEqual(cleaned, "这是原创。")
        self.assertEqual(lyrics, "晚风轻轻吹过山岗")

    def test_public_score_alias_matches_chinese_title(self):
        selected = SingingEngine._select_catalog_item(
            "小星星",
            [
                {"name": "Fur_Elise.mxl"},
                {"name": "12_Variations_of_Twinkle_Twinkle_Little_Star.mxl"},
            ],
        )
        self.assertIsNotNone(selected)
        self.assertIn("Twinkle_Twinkle", selected["name"])

    def test_catalog_alias_tolerates_asr_particle(self):
        selected = SingingEngine._select_catalog_item(
            "小星星来",
            [{"name": "12_Variations_of_Twinkle_Twinkle_Little_Star.mxl"}],
        )
        self.assertIsNotNone(selected)

    def test_two_tigers_maps_to_frere_jacques(self):
        self.assertEqual(SingingEngine._catalog_search_term("两只老虎"), "Frere Jacques")

    def test_ambiguous_classical_aliases_are_specific(self):
        self.assertEqual(SingingEngine._catalog_search_term("卡农"), "Pachelbel Canon")
        self.assertEqual(
            SingingEngine._catalog_search_term("土耳其进行曲"), "Rondo Alla Turca"
        )

    def test_title_normalization_tolerates_asr_particle(self):
        self.assertEqual(SingingEngine.normalize_title("小星星来"), "小星星")

    def test_unknown_song_does_not_match_unrelated_public_score(self):
        selected = SingingEngine._select_catalog_item(
            "一首不存在的现代歌曲",
            [{"name": "Fur_Elise.mxl"}, {"name": "Canon_in_D.mxl"}],
        )
        self.assertIsNone(selected)

    def test_mutopia_parser_accepts_only_allowlisted_midi(self):
        urls = SingingEngine._extract_mutopia_midi_urls(
            '<a href="/ftp/Traditional/greensleeves/greensleeves.mid">MIDI</a>'
            '<a href="https://evil.example/ftp/song.mid">bad</a>'
            '<a href="/ftp/Traditional/score.pdf">PDF</a>'
        )
        self.assertEqual(
            urls,
            [
                "https://www.mutopiaproject.org/ftp/Traditional/greensleeves/"
                "greensleeves.mid"
            ],
        )

    def test_mutopia_selection_prefers_title_match(self):
        selected = SingingEngine._select_mutopia_midi(
            "绿袖子",
            [
                "https://www.mutopiaproject.org/ftp/Anonymous/division.mid",
                "https://www.mutopiaproject.org/ftp/Traditional/greensleeves/greensleeves.mid",
            ],
        )
        self.assertIsNotNone(selected)
        self.assertIn("greensleeves", selected)


if __name__ == "__main__":
    unittest.main()
