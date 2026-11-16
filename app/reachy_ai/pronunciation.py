"""Add missing Mandarin phrase pronunciations to MeloTTS, not to chat text.

The native frontend uses longest phrase matching (up to 10 characters).
Unknown phrases otherwise fall back to the first pronunciation of each character.
This is a dictionary frontend, not a full semantic pronunciation model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pypinyin
from pypinyin.contrib.tone_convert import to_tone3
from pypinyin.phrases_dict import phrases_dict
from pypinyin.pinyin_dict import pinyin_dict


DEFAULT_OVERRIDES = Path(r"D:\ReachyAI\config\tts-pronunciation.json")
DEFAULT_CACHE = Path(r"D:\ReachyAI\cache\tts-pronunciation")
FRONTEND_VERSION = 3
ADJECTIVE_PREFIXES = {
    "很": "hen3", "太": "tai4", "好": "hao3", "更": "geng4",
    "较": "jiao4", "比较": "bi3 jiao4", "这么": "zhe4 me5",
    "那么": "na4 me5", "多": "duo1", "有多": "you3 duo1",
}


def common_phrases() -> dict[str, str]:
    # Explicitly disambiguated words/short contexts. Do not globally replace
    # characters: that would fix "重新" while breaking "重要", for example.
    phrases = {
        "重新": "chong2 xin1", "重启": "chong2 qi3",
        "重来": "chong2 lai2", "重试": "chong2 shi4",
        "重做": "chong2 zuo4", "重说": "chong2 shuo1",
        "重读": "chong2 du2", "重播": "chong2 bo1",
        "重装": "chong2 zhuang1", "重连": "chong2 lian2",
        "重复": "chong2 fu4", "重置": "chong2 zhi4",
        "重建": "chong2 jian4", "重叠": "chong2 die2",
        "重庆": "chong2 qing4", "重要": "zhong4 yao4",
        "重量": "zhong4 liang4", "重音": "zhong4 yin1",
        "重视": "zhong4 shi4", "重点": "zhong4 dian3",
        "长大": "zhang3 da4", "长高": "zhang3 gao1",
        "长胖": "zhang3 pang4", "长个": "zhang3 ge4",
        "长得": "zhang3 de5", "长出": "zhang3 chu1",
        "长出来": "zhang3 chu1 lai2", "长身体": "zhang3 shen1 ti3",
        "成长": "cheng2 zhang3", "生长": "sheng1 zhang3",
        "增长": "zeng1 zhang3", "家长": "jia1 zhang3",
        "校长": "xiao4 zhang3", "队长": "dui4 zhang3",
        "行长": "hang2 zhang3", "银行行长": "yin2 hang2 hang2 zhang3",
        "长时间": "chang2 shi2 jian1", "长一点": "chang2 yi4 dian3",
        "长一些": "chang2 yi4 xie1", "长度": "chang2 du4",
        "长短": "chang2 duan3", "长久": "chang2 jiu3",
        "长远": "chang2 yuan3", "长发": "chang2 fa4",
        "长句": "chang2 ju4", "长文": "chang2 wen2",
        "长音": "chang2 yin1", "音乐": "yin1 yue4",
        "乐曲": "yue4 qu3", "乐谱": "yue4 pu3",
        "快乐": "kuai4 le4", "欢乐": "huan1 le4",
        "银行": "yin2 hang2", "行业": "hang2 ye4",
        "行走": "xing2 zou3", "行不行": "xing2 bu4 xing2",
        "还要": "hai2 yao4", "还有": "hai2 you3",
        "还是": "hai2 shi4", "还没": "hai2 mei2",
        "还给": "huan2 gei3", "归还": "gui1 huan2",
        "还钱": "huan2 qian2", "还书": "huan2 shu1",
        "睡着": "shui4 zhao2", "着急": "zhao2 ji2",
        "看着": "kan4 zhe5", "听着": "ting1 zhe5",
        "说着": "shuo1 zhe5", "想着": "xiang3 zhe5",
        "调节": "tiao2 jie2", "调整": "tiao2 zheng3",
        "调试": "tiao2 shi4", "调大": "tiao2 da4",
        "调小": "tiao2 xiao3", "调高": "tiao2 gao1",
        "调低": "tiao2 di1", "音调": "yin1 diao4",
        "声调": "sheng1 diao4", "调查": "diao4 cha2",
        "便宜": "pian2 yi5", "方便": "fang1 bian4",
        "朝向": "chao2 xiang4", "朝着": "chao2 zhe5",
        "朝阳区": "chao2 yang2 qu1", "朝阳": "zhao1 yang2",
        "得了": "de2 le5", "觉得": "jue2 de5",
        "没收": "mo4 shou1", "没有": "mei2 you3",
        "角色": "jue2 se4", "角度": "jiao3 du4",
        "教我": "jiao1 wo3", "教你": "jiao1 ni3",
        "教学": "jiao4 xue2", "教室": "jiao4 shi4",
    }
    # Common conversational contexts absent from the exported model lexicon.
    for prefix, reading in ADJECTIVE_PREFIXES.items():
        phrases[prefix + "长"] = reading + " chang2"
        phrases[prefix + "重"] = reading + " zhong4"
    for verb, reading in {"说": "shuo1", "讲": "jiang3", "读": "du2",
                          "念": "nian4", "播放": "bo1 fang4",
                          "回答": "hui2 da2", "解释": "jie3 shi4"}.items():
        phrases["重" + verb] = "chong2 " + reading
    return phrases


def melo_entry(word: str, reading: str, tokens: set[str]) -> str:
    """Convert tone-number pinyin to the model's Chinese phones and tones.

    Uses the OpenCPOP/Melo phone conventions (AA/EE/OO, ir, i0, v).
    Validate against tokens.txt so incompatible models cannot silently misread.
    """
    syllables = reading.split()
    if not re.fullmatch(r"[\u4e00-\u9fff]{1,10}", word) or len(syllables) != len(word):
        raise ValueError(f"Invalid Mandarin phrase or syllable count: {word!r}")
    phones: list[str] = []
    tones: list[str] = []
    for syllable in syllables:
        match = re.fullmatch(r"([a-zv]+)([1-5])", syllable)
        if match is None:
            raise ValueError(f"Use tone-number pinyin (e.g. chong2): {syllable!r}")
        body, tone = match.groups()
        initial_match = re.match(r"zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]", body)
        if initial_match:
            initial = initial_match.group()
            final = body[len(initial):]
        else:
            initial = {"a": "AA", "e": "EE", "o": "OO"}.get(body[0], "")
            final = body
        if body in {"zhi", "chi", "shi", "ri"}:
            final = "ir"
        elif body in {"zi", "ci", "si"}:
            final = "i0"
        elif initial in {"j", "q", "x", "y"} and final.startswith("u"):
            final = "v" + final[1:]
        elif body == "yan":
            final = "En"
        elif body == "ye":
            final = "E"
        if initial not in tokens or final not in tokens:
            raise ValueError(f"Unsupported Melo syllable: {syllable!r}")
        phones.extend((initial, final))
        tones.extend((tone, tone))
    return " ".join((word, *phones, *tones))


@lru_cache(maxsize=4)
def melo_aliases(tokens: frozenset[str]) -> dict[str, str]:
    """Stable private-use symbols for all supported Mandarin syllable tones."""
    bodies = {to_tone3(value, neutral_tone_with_five=True)[:-1]
              for alternatives in pinyin_dict.values() for value in alternatives.split(",")}
    result = {}
    for body in sorted(bodies):
        for tone in "12345":
            reading = body + tone
            try:
                melo_entry("一", reading, tokens)
            except ValueError:
                continue
            result[reading] = chr(0xE000 + len(result))
    if len(result) > 0x1900:
        raise ValueError("Mandarin pronunciation aliases exceed private-use range")
    return result


def prepare_melo_lexicon(
    model_dir: Path,
    *,
    overrides_path: Path = DEFAULT_OVERRIDES,
    cache_dir: Path = DEFAULT_CACHE,
) -> Path:
    """Build a cached merged lexicon; never modify downloaded model assets."""
    base = model_dir / "lexicon.txt"
    token_path = model_dir / "tokens.txt"
    source = base.read_bytes()
    token_source = token_path.read_bytes()
    custom_source = overrides_path.read_bytes() if overrides_path.exists() else b"{}"
    signature = hashlib.sha256(
        source + token_source + custom_source
        + f"{FRONTEND_VERSION}:{pypinyin.__version__}".encode()
        + json.dumps(common_phrases(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    target = cache_dir / f"melo-lexicon-{signature}.txt"
    if target.is_file():
        return target

    tokens = {line.split()[0] for line in token_source.decode("utf-8").splitlines() if line.split()}
    entries: dict[str, str] = {}
    for line in source.decode("utf-8").splitlines():
        if line.strip():
            entries.setdefault(line.split()[0], line)
    # Extend only missing phrases; retain the model's existing English entries
    # and Mandarin phrase readings unless an explicit override corrects them.
    for word, values in phrases_dict.items():
        if word in entries or not re.fullmatch(r"[\u4e00-\u9fff]{2,10}", word):
            continue
        reading = " ".join(to_tone3(value[0], neutral_tone_with_five=True) for value in values)
        try:
            entries[word] = melo_entry(word, reading, tokens)
        except ValueError:
            # Rare syllabic m/ng etc. are not representable in this model.
            continue
    overrides = common_phrases()
    custom = json.loads(custom_source.decode("utf-8-sig"))
    if not isinstance(custom, dict) or any(not isinstance(v, str) for v in custom.values()):
        raise ValueError("tts-pronunciation.json must map phrases to tone-number pinyin strings")
    overrides.update(custom)
    for word, reading in overrides.items():
        entries[word] = melo_entry(word, reading, tokens)
    # Guard overlap: adding "很长" must not consume the first character of
    # "很长大"; "很重" must not steal 重 from "很重要" or "很重新".
    # Expand prefix+compound contexts so the native longest matcher keeps the
    # compound's pronunciation. A user's explicit full-context entry wins.
    compounds = [(word, row.split()[1:]) for word, row in entries.items()
                 if len(word) > 1 and word.startswith(("长", "重"))
                 and re.fullmatch(r"[\u4e00-\u9fff]+", word)]
    for prefix, reading in ADJECTIVE_PREFIXES.items():
        prefix_row = melo_entry(prefix, reading, tokens).split()[1:]
        cut = len(prefix_row) // 2
        for word, row in compounds:
            context = prefix + word
            if context in entries or len(context) > 10:
                continue
            midpoint = len(row) // 2
            entries[context] = " ".join([
                context, *prefix_row[:cut], *row[:midpoint],
                *prefix_row[cut:], *row[midpoint:],
            ])
    for reading, alias in melo_aliases(frozenset(tokens)).items():
        phone_row = melo_entry("一", reading, tokens).split()[1:]
        entries[alias] = " ".join([alias, *phone_row])
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Unique temp filename permits diagnostic tests alongside a running Agent.
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(entries.values()) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
