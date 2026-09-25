from __future__ import annotations

import json
import base64
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from html import unescape
from xml.sax.saxutils import escape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx


SETTINGS_PATH = Path(r"D:\ReachyAI\config\settings.json")
PUBLIC_SCORE_CATALOG_API = (
    "https://api.github.com/repos/sevagh/musicxml-library/contents/scores"
)
PUBLIC_SCORE_RAW_PREFIX = (
    "https://raw.githubusercontent.com/sevagh/musicxml-library/"
)
MUTOPIA_SEARCH_URL = "https://www.mutopiaproject.org/cgibin/make-table.cgi"
MUTOPIA_SCORE_PREFIX = "https://www.mutopiaproject.org/ftp/"
QUPU123_HOSTS = {"www.qupu123.com", "qupu123.com"}
AUDIVERIS_DEFAULT = Path(r"D:\ReachyAI\runtime\audiveris\Audiveris\Audiveris.exe")
MELODY_EXTRACTOR_DEFAULT = Path(r"D:\ReachyAI\scripts\extract-melody.py")
MACHINE_SCORE_SUFFIXES = {".musicxml", ".xml", ".mxl", ".mid", ".midi"}
IMAGE_SCORE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
AUDIO_SCORE_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}
KNOWN_OPEN_SCORE_SOURCES = {
    # Published by the University of Macau on a public course page.  Keeping a
    # curated source also makes the result deterministic when general search
    # engines rank paywalled score sites above the official PDF.
    "七子之歌澳门": {
        "url": "https://cpc.fah.um.edu.mo/wp-content/uploads/2025/08/1-e4b883e5ad90e4b98be6ad8c-e6beb3e997a8.pdf",
        "page": "https://cpc.fah.um.edu.mo/zh-hant/36984/",
        "source": "University of Macau",
    },
}
PUBLIC_SCORE_ALIASES = {
    "小星星": "Twinkle Twinkle Little Star",
    "一闪一闪亮晶晶": "Twinkle Twinkle Little Star",
    "两只老虎": "Frere Jacques",
    "两只老虎来": "Frere Jacques",
    "雅克兄弟": "Frere Jacques",
    "生日快乐": "Happy Birthday To You",
    "祝你生日快乐": "Happy Birthday To You",
    "欢乐颂": "Ode to Joy",
    "致爱丽丝": "Fur Elise",
    "卡农": "Pachelbel Canon",
    "D大调卡农": "Pachelbel Canon",
    "月光奏鸣曲": "Moonlight Sonata",
    "天鹅湖": "Swan Lake",
    "绿袖子": "Greensleeves",
    "土耳其进行曲": "Rondo Alla Turca",
    "奇异恩典": "Amazing Grace",
    "婚礼进行曲": "Wedding March",
    "蓝色多瑙河": "Blue Danube",
    "勃拉姆斯摇篮曲": "Brahms Lullaby",
    "摇篮曲": "Brahms Lullaby",
}


class _MidiLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = next((value for name, value in attrs if name.casefold() == "href"), None)
        if href and re.search(r"\.midi?(?:$|[?#])", href, re.IGNORECASE):
            self.links.append(href)


@dataclass(frozen=True)
class SingingConfig:
    enabled: bool = True
    python: Path = Path(r"D:\ReachyAI\runtime\singing\.venv\Scripts\python.exe")
    renderer: Path = Path(r"D:\ReachyAI\scripts\singing-render.py")
    voice_bank: Path = Path(r"D:\ReachyAI\models\singing\AsaritsuV1.0")
    output_root: Path = Path(r"D:\ReachyAI\data\singing-output\requests")
    score_root: Path = Path(r"D:\ReachyAI\data\scores")
    speaker: str = "Gentle"
    max_chars: int = 16
    pitch_steps: int = 6
    variance_steps: int = 6
    acoustic_steps: int = 12
    timeout_seconds: float = 600.0
    web_score_search: bool = True
    audiveris: Path = AUDIVERIS_DEFAULT
    melody_extractor: Path = MELODY_EXTRACTOR_DEFAULT

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "SingingConfig":
        defaults = cls()
        if not path.exists():
            return defaults
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = raw.get("singing", {})
        return cls(
            enabled=bool(values.get("enabled", defaults.enabled)),
            python=Path(values.get("python", str(defaults.python))),
            renderer=Path(values.get("renderer", str(defaults.renderer))),
            voice_bank=Path(values.get("voice_bank", str(defaults.voice_bank))),
            output_root=Path(values.get("output_root", str(defaults.output_root))),
            score_root=Path(values.get("score_root", str(defaults.score_root))),
            speaker=str(values.get("speaker", defaults.speaker)),
            max_chars=max(2, min(32, int(values.get("max_chars", defaults.max_chars)))),
            pitch_steps=max(1, int(values.get("pitch_steps", defaults.pitch_steps))),
            variance_steps=max(1, int(values.get("variance_steps", defaults.variance_steps))),
            acoustic_steps=max(1, int(values.get("acoustic_steps", defaults.acoustic_steps))),
            timeout_seconds=max(30.0, float(values.get("timeout_seconds", defaults.timeout_seconds))),
            web_score_search=bool(values.get("web_score_search", defaults.web_score_search)),
            audiveris=Path(values.get("audiveris", str(defaults.audiveris))),
            melody_extractor=Path(
                values.get("melody_extractor", str(defaults.melody_extractor))
            ),
        )


@dataclass(frozen=True)
class ScoreResolution:
    path: Path
    source: str
    local: bool


class SingingEngine:
    """Isolated DiffSinger renderer used only for explicit singing requests."""

    def __init__(self, config: SingingConfig | None = None):
        self.config = config or SingingConfig.load()

    def validate(self) -> None:
        if not self.config.enabled:
            raise RuntimeError("歌声合成已在配置中关闭")
        for label, path in (
            ("歌声 Python", self.config.python),
            ("歌声渲染脚本", self.config.renderer),
            ("中文女声声库", self.config.voice_bank),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label}不存在：{path}")

    @staticmethod
    def _score_key(value: str) -> str:
        return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", value.casefold())

    @classmethod
    def normalize_title(cls, title: str) -> str:
        """Canonicalize common spoken/ASR variants of an existing song title."""
        candidate = title.strip(" 《》\"'“”‘’。，！？,.!?\n\t")
        key = cls._score_key(candidate)
        particle_keys = {
            "来",
            "吧",
            "呀",
            "啊",
            "呢",
            "啦",
            "给我听",
            "给大家听",
            "听听",
            "唱一下",
            "唱一遍",
        }
        for alias in PUBLIC_SCORE_ALIASES:
            alias_key = cls._score_key(alias)
            if key == alias_key:
                return alias
            if key.startswith(alias_key) and key[len(alias_key) :] in particle_keys:
                return alias
        candidate = re.sub(
            r"(?:给我听|给大家听|来听听|听一听|听听|好吗|可以吗|好不好|"
            r"行不行|一下|一遍|吧|呀|啊|呢|啦|来)+$",
            "",
            candidate,
        ).strip(" 《》\"'“”‘’。，！？,.!?")
        return candidate

    @classmethod
    def _catalog_search_term(cls, title: str) -> str:
        normalized = cls.normalize_title(title)
        return PUBLIC_SCORE_ALIASES.get(cls._score_key(normalized), normalized)

    @classmethod
    def _select_catalog_item(
        cls, title: str, items: list[dict[str, object]]
    ) -> dict[str, object] | None:
        wanted = cls._score_key(cls._catalog_search_term(title))
        candidates: list[tuple[int, float, int, dict[str, object]]] = []
        for item in items:
            name = str(item.get("name") or "")
            suffix = Path(name).suffix.casefold()
            if suffix not in {".musicxml", ".xml", ".mxl", ".mid", ".midi"}:
                continue
            key = cls._score_key(Path(name).stem)
            contains = int(bool(wanted and wanted in key))
            similarity = SequenceMatcher(None, wanted, key).ratio() if wanted else 0.0
            if contains or similarity >= 0.62:
                candidates.append((contains, similarity, -len(name), item))
        return max(candidates, key=lambda value: value[:3])[3] if candidates else None

    @classmethod
    def _extract_mutopia_midi_urls(cls, html: str) -> list[str]:
        parser = _MidiLinkParser()
        parser.feed(html)
        accepted: list[str] = []
        for href in parser.links:
            url = urljoin(MUTOPIA_SEARCH_URL, href)
            parsed = urlparse(url)
            if (
                parsed.scheme == "https"
                and parsed.netloc == "www.mutopiaproject.org"
                and url.startswith(MUTOPIA_SCORE_PREFIX)
                and Path(parsed.path).suffix.casefold() in {".mid", ".midi"}
            ):
                accepted.append(url)
        return list(dict.fromkeys(accepted))

    @classmethod
    def _select_mutopia_midi(cls, title: str, urls: list[str]) -> str | None:
        if not urls:
            return None
        wanted = cls._score_key(cls._catalog_search_term(title))
        ranked: list[tuple[int, float, int, str]] = []
        for url in urls:
            parsed = urlparse(url)
            path_key = cls._score_key(Path(parsed.path).stem)
            full_key = cls._score_key(parsed.path)
            contains = int(bool(wanted and (wanted in full_key or path_key in wanted)))
            similarity = SequenceMatcher(None, wanted, path_key).ratio() if wanted else 0.0
            ranked.append((contains, similarity, -len(parsed.path), url))
        return max(ranked, key=lambda value: value[:3])[3]

    def find_score(self, title: str) -> Path | None:
        root = self.config.score_root
        if not root.exists():
            return None
        wanted = self._score_key(self.normalize_title(title))
        supported = MACHINE_SCORE_SUFFIXES
        exact: list[Path] = []
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.casefold() not in supported:
                continue
            if any(part.casefold() in {"quarantine", "omr-output"} for part in candidate.parts):
                continue
            # Cache names are TITLE__source.ext.  Match the title component
            # exactly: substring matching previously made similarly named
            # songs silently reuse the wrong score.
            key = self._score_key(candidate.stem.split("__", 1)[0])
            if key == wanted:
                exact.append(candidate)
        for candidate in sorted(exact, key=lambda path: (len(path.stem), str(path))):
            try:
                self.validate_singing_score(title, candidate)
                return candidate
            except RuntimeError:
                continue
        return None

    def find_source_asset(self, title: str) -> Path | None:
        """Find an exact-title local score image, PDF, numbered score, or audio."""
        root = self.config.score_root
        if not root.exists():
            return None
        wanted = self._score_key(self.normalize_title(title))
        supported = IMAGE_SCORE_SUFFIXES | AUDIO_SCORE_SUFFIXES | {".jianpu", ".numbered"}
        matches: list[Path] = []
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.casefold() not in supported:
                continue
            if any(part.casefold() in {"quarantine", "omr-output", "source-pdf"} for part in candidate.parts):
                continue
            if self._score_key(candidate.stem.split("__", 1)[0]) == wanted:
                matches.append(candidate)
        return min(matches, key=lambda path: (len(path.stem), str(path))) if matches else None

    @staticmethod
    def _musicxml_bytes(path: Path) -> bytes | None:
        suffix = path.suffix.casefold()
        if suffix == ".mxl":
            if not zipfile.is_zipfile(path):
                return None
            with zipfile.ZipFile(path) as archive:
                names = [
                    name for name in archive.namelist()
                    if name.casefold().endswith((".musicxml", ".xml"))
                    and "container.xml" not in name.casefold()
                ]
                return archive.read(names[0]) if names else None
        if suffix in {".musicxml", ".xml"}:
            return path.read_bytes()
        return None

    def validate_singing_score(self, title: str, path: Path) -> None:
        """Reject accompaniment dumps and scores without lyric alignment."""
        self._validate_score_file(path)
        sidecar = path.with_suffix(".txt")
        sidecar_syllables = 0
        if sidecar.exists():
            sidecar_syllables = len(re.findall(r"[A-Za-z]+|[\u3400-\u9fff]", sidecar.read_text(encoding="utf-8", errors="ignore")))
        xml = self._musicxml_bytes(path)
        if xml is not None:
            note_count = len(re.findall(br"<note(?:\s|>)", xml))
            lyric_count = len(re.findall(br"<lyric(?:\s|>)", xml))
            part_count = len(re.findall(br"<part(?:\s|>)", xml))
            chord_count = len(re.findall(br"<chord(?:\s|/>)", xml))
            has_lyrics = lyric_count >= 4 or sidecar_syllables >= 4
            # A multi-part/chord-heavy OMR result without aligned lyrics is
            # almost certainly an accompaniment score, not a vocal melody.
            if not has_lyrics and (part_count > 1 or chord_count > 8 or note_count > 180):
                raise RuntimeError(
                    f"《{title}》缓存的是伴奏/编配谱，缺少可验证的歌词音节对齐"
                )
            if not has_lyrics:
                raise RuntimeError(f"《{title}》乐谱没有嵌入歌词或同名 .txt 歌词")
        elif path.suffix.casefold() in {".mid", ".midi"} and sidecar_syllables < 4:
            raise RuntimeError(f"《{title}》MIDI 缺少同名 .txt 歌词，无法可靠对齐")

    def resolve_score(self, title: str) -> ScoreResolution:
        """Resolve one validated singing score through local and network stages."""
        title = self.normalize_title(title)
        local = self.find_score(title)
        if local is not None:
            return ScoreResolution(local, "local-machine-score", True)
        asset = self.find_source_asset(title)
        if asset is not None:
            converted = self._convert_local_asset(title, asset)
            self.validate_singing_score(title, converted)
            return ScoreResolution(converted, f"local-{asset.suffix.casefold()[1:]}", True)
        acquired = self.acquire_score(title)
        if acquired is None:
            raise FileNotFoundError(
                f"已检查本地 MIDI、MusicXML、简谱、图片和音频，并尝试联网获取；"
                f"仍未得到《{title}》可靠的单声部主旋律、节拍和歌词音节对齐。"
            )
        self.validate_singing_score(title, acquired)
        return ScoreResolution(acquired, "network-acquired", False)

    def acquire_score(self, title: str) -> Path | None:
        """Acquire a directly accessible score, without bypassing access controls."""
        return self.download_public_domain_score(title)

    def download_public_domain_score(self, title: str) -> Path | None:
        """Backward-compatible name for the general score acquisition pipeline."""
        title = self.normalize_title(title)
        errors: list[str] = []
        for downloader in (
            self._download_qupu123_score,
            self._download_musicxml_library_score,
            self._download_mutopia_score,
            self._download_known_open_pdf_score,
            self._search_open_web_score,
        ):
            try:
                score = downloader(title)
            except Exception as exc:
                errors.append(f"{downloader.__name__}: {exc}")
                continue
            if score is not None:
                try:
                    self.validate_singing_score(title, score)
                except RuntimeError as exc:
                    errors.append(f"{downloader.__name__}: 质量校验未通过：{exc}")
                    continue
                return score
        if errors and len(errors) == 5:
            raise RuntimeError("联网乐谱服务暂时不可用：" + "；".join(errors))
        return None

    def _convert_local_asset(self, title: str, asset: Path) -> Path:
        suffix = asset.suffix.casefold()
        if suffix in IMAGE_SCORE_SUFFIXES:
            try:
                return self._omr_local_asset(title, asset)
            except RuntimeError:
                return self._transcribe_jianpu_with_cloud(title, asset)
        if suffix in AUDIO_SCORE_SUFFIXES:
            return self._extract_local_audio_melody(title, asset)
        raise RuntimeError(f"暂不支持这种本地旋律来源：{asset.suffix}")

    def _omr_local_asset(self, title: str, asset: Path) -> Path:
        if not self.config.audiveris.exists():
            raise RuntimeError(f"Audiveris 不存在：{self.config.audiveris}")
        output_dir = self.config.score_root / "omr-output"
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                str(self.config.audiveris), "-batch", "-constant",
                "org.audiveris.omr.text.Language.defaultSpecification=chi_sim+eng",
                "-transcribe", "-export", "-output", str(output_dir), "--", str(asset),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=max(180.0, self.config.timeout_seconds), check=False,
        )
        candidates = sorted(output_dir.glob(f"{asset.stem}*.mxl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if completed.returncode != 0 or not candidates:
            tail = (completed.stderr or completed.stdout)[-1000:]
            raise RuntimeError(f"本地图片/PDF 识谱失败：{tail}")
        target = self.config.score_root / f"{self._safe_title(title)}__local-omr.mxl"
        shutil.copy2(candidates[0], target)
        lyric_sidecar = asset.with_suffix(".txt")
        if lyric_sidecar.exists():
            shutil.copy2(lyric_sidecar, target.with_suffix(".txt"))
        return target

    def _extract_local_audio_melody(self, title: str, asset: Path) -> Path:
        if not self.config.melody_extractor.exists():
            raise RuntimeError(f"旋律提取脚本不存在：{self.config.melody_extractor}")
        target = self.config.score_root / f"{self._safe_title(title)}__audio-melody.mid"
        completed = subprocess.run(
            [
                str(self.config.python), str(self.config.melody_extractor),
                "--input", str(asset), "--output", str(target),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=max(120.0, self.config.timeout_seconds), check=False,
        )
        if completed.returncode != 0 or not target.exists():
            tail = (completed.stderr or completed.stdout)[-1000:]
            raise RuntimeError(f"音频主旋律提取失败：{tail}")
        sidecar = asset.with_suffix(".txt")
        if sidecar.exists():
            shutil.copy2(sidecar, target.with_suffix(".txt"))
        return target

    def _transcribe_jianpu_with_cloud(self, title: str, asset: Path) -> Path:
        """Transcribe a clear monophonic numbered score with the cloud vision model."""
        from .config import load_config
        from .openai_client import OpenAICompatClient, assistant_text

        endpoint = load_config().cloud
        if not endpoint.api_key:
            raise RuntimeError("简谱需要云端视觉转谱，但未配置云端 API Key")
        mime = "image/png" if asset.suffix.casefold() == ".png" else "image/jpeg"
        encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
        prompt = (
            f"把图片中的《{title}》单声部声乐简谱转成结构化音符。"
            "抄录完整主旋律，不要抄伴奏、和弦、前奏或其他声部。谱面有多段并列歌词时全部抄录。"
            "严格按谱面，不得凭记忆补旋律或歌词。返回一个JSON对象，不要markdown。字段："
            "title、time_beats（每小节拍数，例如2）、events。events每项字段："
            "measure（从1开始）、midi（整数36到96，休止符为null）、beats（以四分音符为1）、"
            "lyrics（数组，每段歌词在该音符对应的一个汉字；延音/休止为空字符串）。"
            "反复段按一段歌词的实际演唱顺序展开一次，程序会为其他段重复旋律。"
            "输出前逐小节复核：除弱起和末小节外，同一measure内所有beats之和必须等于time_beats。"
        )
        response = OpenAICompatClient(endpoint).chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }
            ],
            max_tokens=6000,
            temperature=0.0,
            extra=endpoint.extra_body,
        )
        raw = assistant_text(response)
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise RuntimeError("云端简谱转写没有返回结构化 JSON")
        payload = json.loads(match.group(0))
        returned_title = str(payload.get("title") or "")
        if self._score_key(title) not in self._score_key(returned_title):
            raise RuntimeError(f"简谱标题不匹配：请求《{title}》，识别为《{returned_title}》")
        time_beats = float(payload.get("time_beats") or 0)
        if time_beats not in {2.0, 3.0, 4.0, 6.0}:
            raise RuntimeError("简谱拍号无法可靠识别")
        events = payload.get("events")
        if not isinstance(events, list) or not 8 <= len(events) <= 600:
            raise RuntimeError("简谱主旋律音符数量异常")
        base_events: list[dict[str, object]] = []
        measure_totals: dict[int, float] = {}
        for item in events:
            if not isinstance(item, dict):
                raise RuntimeError("简谱事件结构无效")
            measure = int(item.get("measure") or 0)
            beats = float(item.get("beats") or 0)
            midi = item.get("midi")
            raw_lyrics = item.get("lyrics")
            if isinstance(raw_lyrics, list):
                lyrics = [str(value or "").strip() for value in raw_lyrics]
            else:
                lyrics = [str(item.get("lyric") or "").strip()]
            if measure < 1 or beats <= 0 or beats > 8:
                raise RuntimeError("简谱小节或时值异常")
            if midi is not None and not 36 <= int(midi) <= 96:
                raise RuntimeError("简谱音高超出人声范围")
            if any(len(lyric) > 2 for lyric in lyrics):
                raise RuntimeError("歌词没有按音节对齐")
            measure_totals[measure] = measure_totals.get(measure, 0.0) + beats
            base_events.append({"measure": measure, "beats": beats, "midi": None if midi is None else int(midi), "lyrics": lyrics})
        verse_count = max(len(event["lyrics"]) for event in base_events)
        verse_lyric_counts = [
            sum(
                bool(event["lyrics"][verse] if verse < len(event["lyrics"]) else "")
                for event in base_events
            )
            for verse in range(verse_count)
        ]
        usable_verses = [index for index, count in enumerate(verse_lyric_counts) if count >= 8]
        if not usable_verses or verse_lyric_counts[usable_verses[0]] / len(base_events) < 0.25:
            raise RuntimeError("简谱没有提取出足够的逐音符歌词")
        complete = [total for key, total in measure_totals.items() if key not in {min(measure_totals), max(measure_totals)}]
        if complete and sum(abs(total - time_beats) <= 0.26 for total in complete) / len(complete) < 0.65:
            raise RuntimeError("简谱小节拍数校验失败")

        max_measure = max(measure_totals)
        cleaned: list[dict[str, object]] = []
        for output_verse, verse in enumerate(usable_verses):
            for event in base_events:
                lyrics = event["lyrics"]
                lyric = lyrics[verse] if verse < len(lyrics) else ""
                cleaned.append(
                    {
                        "measure": int(event["measure"]) + output_verse * max_measure,
                        "beats": event["beats"],
                        "midi": event["midi"],
                        "lyric": lyric,
                    }
                )

        divisions = 24
        denominator = 4
        numerator = int(time_beats)
        measures: dict[int, list[dict[str, object]]] = {}
        for event in cleaned:
            measures.setdefault(int(event["measure"]), []).append(event)
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            '<score-partwise version="3.1">',
            f'<work><work-title>{escape(title)}</work-title></work>',
            '<part-list><score-part id="P1"><part-name>Vocal melody</part-name></score-part></part-list>',
            '<part id="P1">',
        ]
        steps = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
        alters = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)
        for measure_no in sorted(measures):
            xml_lines.append(f'<measure number="{measure_no}">')
            if measure_no == min(measures):
                xml_lines.append(
                    f'<attributes><divisions>{divisions}</divisions><time><beats>{numerator}</beats>'
                    f'<beat-type>{denominator}</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>'
                )
            for event in measures[measure_no]:
                duration = max(1, int(round(float(event["beats"]) * divisions)))
                xml_lines.append('<note>')
                midi = event["midi"]
                if midi is None:
                    xml_lines.append('<rest/>')
                else:
                    midi_value = int(midi)
                    pitch_class = midi_value % 12
                    xml_lines.append(
                        f'<pitch><step>{steps[pitch_class]}</step>'
                        + (f'<alter>{alters[pitch_class]}</alter>' if alters[pitch_class] else '')
                        + f'<octave>{midi_value // 12 - 1}</octave></pitch>'
                    )
                xml_lines.append(f'<duration>{duration}</duration><voice>1</voice>')
                lyric = str(event["lyric"])
                if lyric:
                    xml_lines.append(f'<lyric><syllabic>single</syllabic><text>{escape(lyric)}</text></lyric>')
                xml_lines.append('</note>')
            xml_lines.append('</measure>')
        xml_lines.extend(['</part>', '</score-partwise>'])
        target = self.config.score_root / f"{self._safe_title(title)}__jianpu-cloud.musicxml"
        target.write_text("\n".join(xml_lines), encoding="utf-8")
        target.with_suffix(".txt").write_text("".join(str(x["lyric"]) for x in cleaned), encoding="utf-8")
        target.with_suffix(target.suffix + ".transcription.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.validate_singing_score(title, target)
        return target

    @staticmethod
    def _qupu_candidate_score(title: str, label: str) -> tuple[int, int, int, int, int]:
        wanted = SingingEngine._score_key(title)
        visible_title = label.split("[", 1)[0].strip()
        title_without_version = re.sub(r"[（(][^）)]*[）)]", "", visible_title).strip()
        key = SingingEngine._score_key(title_without_version)
        raw_key = SingingEngine._score_key(visible_title)
        exact = 3 if key == wanted else (2 if raw_key.startswith(wanted) and "（" in visible_title else int(bool(wanted and wanted in key)))
        parseable = 2 if re.search(r"五线谱|线简", label) else int("简谱" in label)
        vocal = 2 if re.search(r"民歌曲谱|通俗曲谱|少儿曲谱|声乐", label) else int(bool(re.search(r"弹唱|个人园地|谱友", label)))
        single = int(not bool(re.search(
            r"钢琴|伴奏|合奏|总谱|合唱|二声部|三声部|四声部|器乐|独奏|"
            r"萨克斯|手风琴|电子琴|古琴|古筝|胡琴|笛|箫|吉他|提琴|铜管|正谱",
            label,
        )))
        return exact, parseable, vocal, single, -len(label)

    def _download_qupu123_score(self, title: str) -> Path | None:
        """Prefer a vocal/single-melody page from 中国曲谱网, then run OMR."""
        search = httpx.get(
            "https://www.qupu123.com/Search",
            params={"keys": title},
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 ReachyAI/1.0"},
        )
        search.raise_for_status()
        rows = re.findall(r"<tr>(.*?)</tr>", search.text, re.I | re.S)
        site_results: list[dict[str, object]] = []
        for row in rows:
            link = re.search(r'<a\s+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', row, re.I | re.S)
            cells = re.findall(r'<td[^>]*class=["\']f[134]["\'][^>]*>(.*?)</td>', row, re.I | re.S)
            if not link:
                continue
            clean = lambda value: re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", value))).strip()
            candidate_title = clean(link.group(2))
            category_match = re.search(r"<span>\[([^\]]+)\]</span>", row, re.I | re.S)
            category = clean(category_match.group(1)) if category_match else ""
            detail = " / ".join(clean(cell) for cell in cells)
            site_results.append(
                {
                    "title": f"{candidate_title} [{category}]",
                    "snippet": detail,
                    "url": urljoin(str(search.url), link.group(1)),
                }
            )
        ranked: list[tuple[tuple[int, int, int, int, int], dict[str, object]]] = []
        for item in site_results:
            url = str(item.get("url") or "")
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.netloc.casefold() not in QUPU123_HOSTS:
                continue
            label = f"{item.get('title', '')} {item.get('snippet', '')}"
            if self._score_key(title) not in self._score_key(label):
                continue
            ranked.append((self._qupu_candidate_score(title, label), item))
        ranked.sort(key=lambda value: value[0], reverse=True)
        candidates = [item for _, item in ranked[:24]]
        candidates = self._rank_score_candidates_with_cloud(title, candidates)
        for item in candidates[:6]:
            try:
                score = self._download_qupu123_page(title, str(item["url"]), str(item.get("title") or ""))
                if score is not None:
                    self.validate_singing_score(title, score)
                    return score
            except (httpx.HTTPError, RuntimeError, ValueError):
                continue
        return None

    @staticmethod
    def _rank_score_candidates_with_cloud(
        title: str, candidates: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Let the configured cloud model disambiguate versions, never invent notes."""
        if len(candidates) < 2:
            return candidates
        try:
            from .config import load_config
            from .openai_client import OpenAICompatClient, assistant_text

            endpoint = load_config().cloud
            if not endpoint.api_key:
                return candidates
            compact = [
                {
                    "index": index,
                    "title": str(item.get("title") or "")[:180],
                    "snippet": str(item.get("snippet") or "")[:300],
                    "url": str(item.get("url") or ""),
                }
                for index, item in enumerate(candidates)
            ]
            prompt = (
                "你是歌曲版本匹配器。用户要唱的名称是：" + title + "。"
                "请优先选择同名、声乐单旋律、简谱且包含歌词的版本；"
                "降低钢琴伴奏、器乐独奏、合奏、总谱、多声部合唱的优先级。"
                "如果用户给了歌手或版本，必须一致。只返回JSON："
                '{"order":[候选index...]}。候选：' + json.dumps(compact, ensure_ascii=False)
            )
            response = OpenAICompatClient(endpoint).chat(
                [
                    {"role": "system", "content": "只做候选排序，不补写未知信息，只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.0,
                extra=endpoint.extra_body,
            )
            text = assistant_text(response)
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return candidates
            order = json.loads(match.group(0)).get("order") or []
            selected: list[dict[str, object]] = []
            used: set[int] = set()
            for value in order:
                index = int(value)
                if 0 <= index < len(candidates) and index not in used:
                    selected.append(candidates[index])
                    used.add(index)
            selected.extend(item for index, item in enumerate(candidates) if index not in used)
            return selected
        except Exception:
            return candidates

    def _download_qupu123_page(self, title: str, page_url: str, search_title: str) -> Path | None:
        parsed = urlparse(page_url)
        if parsed.netloc.casefold() not in QUPU123_HOSTS:
            return None
        headers = {"User-Agent": "Mozilla/5.0 ReachyAI/1.0"}
        response = httpx.get(page_url, timeout=30.0, follow_redirects=True, headers=headers)
        response.raise_for_status()
        html = response.text
        page_id = None
        match = re.search(r"(?:/p|Mobile-view-id-)(\d+)", str(response.url))
        if match:
            page_id = match.group(1)
            mobile_url = f"https://www.qupu123.com/Mobile-view-id-{page_id}.html"
            mobile = httpx.get(mobile_url, timeout=30.0, follow_redirects=True, headers=headers)
            if mobile.status_code == 200:
                html += "\n" + mobile.text
        score_regions = re.findall(
            r'<div\s+class=["\']imageList["\']>(.*?)(?:<div\s+class=["\']bd_share|</div>\s*</div>)',
            html,
            re.I | re.S,
        )
        score_region = "\n".join(score_regions)
        image_urls: list[str] = []
        for raw in re.findall(r"(?:src|href)=[\"']([^\"']+\.(?:png|jpe?g))(?:\?[^\"']*)?[\"']", score_region, re.I):
            url = urljoin(str(response.url), raw)
            image_host = urlparse(url).netloc.casefold()
            if image_host in QUPU123_HOSTS or image_host.endswith(".qupu123.com"):
                if "/Public/Uploads/" in url:
                    image_urls.append(url)
        image_urls = list(dict.fromkeys(image_urls))
        if not image_urls:
            return None
        source_dir = self.config.score_root / "source-images" / self._safe_title(title)
        source_dir.mkdir(parents=True, exist_ok=True)
        assets: list[Path] = []
        for index, url in enumerate(image_urls[:12], 1):
            data = httpx.get(url, timeout=45.0, follow_redirects=True, headers={**headers, "Referer": page_url})
            data.raise_for_status()
            if len(data.content) < 5000 or len(data.content) > 12 * 1024 * 1024:
                continue
            suffix = Path(urlparse(url).path).suffix.casefold() or ".jpg"
            asset = source_dir / f"page-{index:02d}{suffix}"
            asset.write_bytes(data.content)
            assets.append(asset)
        if not assets:
            return None
        # Audiveris can process an image directly.  Prefer the first page for
        # now; multi-page merging is attempted only after each page passes OMR.
        try:
            score = self._omr_local_asset(title, assets[0])
        except RuntimeError:
            score = self._transcribe_jianpu_with_cloud(title, assets[0])
        metadata = {
            "title_requested": title,
            "catalog_title": search_title,
            "source_page": page_url,
            "catalog": "中国曲谱网 qupu123.com",
            "page_id": page_id,
            "image_count": len(assets),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        score.with_suffix(score.suffix + ".source.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return score

    @staticmethod
    def _safe_title(title: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u3400-\u9fff._-]+", "_", title).strip("._")

    @staticmethod
    def _validate_score_file(path: Path) -> None:
        suffix = path.suffix.casefold()
        content = path.read_bytes()
        if suffix == ".mxl":
            if not zipfile.is_zipfile(path):
                raise RuntimeError("MXL 不是有效压缩乐谱")
            with zipfile.ZipFile(path) as archive:
                score_names = [
                    name
                    for name in archive.namelist()
                    if name.casefold().endswith((".musicxml", ".xml"))
                    and "container.xml" not in name.casefold()
                ]
                if not score_names:
                    raise RuntimeError("MXL 不包含 MusicXML 乐谱")
                xml = archive.read(score_names[0])
        elif suffix in {".musicxml", ".xml"}:
            xml = content
        elif suffix in {".mid", ".midi"}:
            if not content.startswith(b"MThd"):
                raise RuntimeError("文件不是有效 MIDI")
            return
        else:
            raise RuntimeError("不支持的乐谱格式")
        if b"<score-" not in xml[:4000] or len(re.findall(br"<note(?:\s|>)", xml)) < 8:
            raise RuntimeError("MusicXML 中没有足够的可演唱音符")

    def _download_known_open_pdf_score(self, title: str) -> Path | None:
        source = KNOWN_OPEN_SCORE_SOURCES.get(self._score_key(title))
        if source is None:
            return None
        return self._download_and_omr_pdf(title, source["url"], source.get("page"), source["source"])

    def _download_and_omr_pdf(
        self, title: str, source_url: str, source_page: str | None, source_name: str
    ) -> Path:
        if not self.config.audiveris.exists():
            raise RuntimeError(f"Audiveris 不存在：{self.config.audiveris}")
        safe_title = self._safe_title(title) or "score"
        source_dir = self.config.score_root / "source-pdf"
        output_dir = self.config.score_root / "omr-output"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = source_dir / f"{safe_title}.pdf"
        headers = {"User-Agent": "Mozilla/5.0 ReachyAI/1.0"}
        if source_page:
            headers["Referer"] = source_page
        response = httpx.get(
            source_url, timeout=45.0, follow_redirects=True, headers=headers
        )
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF") or len(content) > 20 * 1024 * 1024:
            raise RuntimeError("公开地址没有返回有效 PDF 乐谱")
        pdf_path.write_bytes(content)
        completed = subprocess.run(
            [
                str(self.config.audiveris),
                "-batch",
                "-constant",
                "org.audiveris.omr.text.Language.defaultSpecification=chi_sim+eng",
                "-transcribe",
                "-export",
                "-output",
                str(output_dir),
                "--",
                str(pdf_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(180.0, self.config.timeout_seconds),
            check=False,
        )
        candidates = sorted(output_dir.glob(f"{pdf_path.stem}*.mxl"))
        if completed.returncode != 0 or not candidates:
            tail = (completed.stderr or completed.stdout)[-1000:]
            raise RuntimeError(f"乐谱图片识别失败：{tail}")
        generated = candidates[0]
        self._validate_score_file(generated)
        target = self.config.score_root / f"{safe_title}__web.mxl"
        shutil.copy2(generated, target)
        # Prefer the PDF's real text layer over OCR when available.  This is
        # substantially more accurate for Chinese lyrics and remains local to
        # the downloaded score cache.
        try:
            from pypdf import PdfReader

            extracted = "\n".join(
                page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages
            )
            lyric_lines = []
            for line in extracted.splitlines():
                stripped = line.strip()
                if re.search(r"词$|曲$|制谱|乐谱编号|第\s*\d+\s*页|共\s*\d+\s*页", stripped):
                    continue
                lyric_lines.append(stripped)
            lyric_text = "".join(lyric_lines)
            if len(re.findall(r"[\u3400-\u9fff]", lyric_text)) >= 8:
                target.with_suffix(".txt").write_text(lyric_text, encoding="utf-8")
        except (ImportError, OSError, ValueError):
            pass
        metadata = {
            "title_requested": title,
            "source": source_url,
            "source_page": source_page,
            "catalog": source_name,
            "conversion": "Audiveris OMR",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        target.with_suffix(target.suffix + ".source.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def _search_open_web_score(self, title: str) -> Path | None:
        """Find a directly public score; never log in or bypass a paywall."""
        if not self.config.web_score_search:
            return None
        from .network_tools import _ddgs_search

        queries = [
            f'"{title}" 乐谱 filetype:pdf',
            f'"{title}" 五线谱 PDF',
        ]
        results: list[dict[str, object]] = []
        for query in queries:
            payload = json.loads(_ddgs_search(query, 10))
            results.extend(payload.get("results") or [])
        wanted = self._score_key(title)
        ranked: list[tuple[int, int, int, dict[str, object]]] = []
        seen: set[str] = set()
        for item in results:
            url = str(item.get("url") or "")
            if url in seen:
                continue
            seen.add(url)
            parsed = urlparse(url)
            if parsed.scheme != "https" or Path(parsed.path).suffix.casefold() != ".pdf":
                continue
            label = f"{item.get('title', '')} {item.get('snippet', '')}"
            label_key = self._score_key(label)
            exact = int(bool(wanted and wanted in label_key))
            score_word = int(bool(re.search(r"乐谱|五线谱|简谱|曲谱|score", label, re.I)))
            penalty = int(bool(re.search(r"教案|教学设计|论文|研究|VIP|登录|付费", label, re.I)))
            ranked.append((exact, score_word, -penalty, item))
        for _, _, _, item in sorted(ranked, key=lambda value: value[:3], reverse=True)[:3]:
            try:
                return self._download_and_omr_pdf(
                    title,
                    str(item["url"]),
                    None,
                    f"web search: {item.get('title', '')}",
                )
            except (httpx.HTTPError, RuntimeError, subprocess.SubprocessError):
                continue
        return None

    def _download_musicxml_library_score(self, title: str) -> Path | None:
        response = httpx.get(
            PUBLIC_SCORE_CATALOG_API,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "ReachyAI/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("公共领域乐谱目录返回了无效数据")
        item = self._select_catalog_item(title, payload)
        if item is None:
            return None

        source_url = str(item.get("download_url") or "")
        parsed = urlparse(source_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "raw.githubusercontent.com"
            or not source_url.startswith(PUBLIC_SCORE_RAW_PREFIX)
        ):
            raise RuntimeError("公共领域乐谱下载地址不在可信白名单内")
        declared_size = int(item.get("size") or 0)
        if declared_size <= 0 or declared_size > 5 * 1024 * 1024:
            raise RuntimeError("公共领域乐谱文件大小异常")

        download = httpx.get(
            source_url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "ReachyAI/1.0"},
        )
        download.raise_for_status()
        content = download.content
        if not content or len(content) > 5 * 1024 * 1024:
            raise RuntimeError("公共领域乐谱下载内容为空或过大")

        source_name = Path(str(item.get("name") or "score.mxl"))
        safe_title = re.sub(r"[^0-9A-Za-z\u3400-\u9fff._-]+", "_", title).strip("._")
        target = self.config.score_root / f"{safe_title}__{source_name.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        temporary.write_bytes(content)
        try:
            if target.suffix.casefold() == ".mxl":
                if not zipfile.is_zipfile(temporary):
                    raise RuntimeError("下载的 MXL 不是有效压缩乐谱")
                with zipfile.ZipFile(temporary) as archive:
                    if not any(
                        name.casefold().endswith((".musicxml", ".xml"))
                        and "container.xml" not in name.casefold()
                        for name in archive.namelist()
                    ):
                        raise RuntimeError("下载的 MXL 不包含 MusicXML 乐谱")
            elif target.suffix.casefold() in {".musicxml", ".xml"}:
                if b"<score-" not in content[:2000]:
                    raise RuntimeError("下载文件不是有效 MusicXML")
            elif not content.startswith(b"MThd"):
                raise RuntimeError("下载文件不是有效 MIDI")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

        metadata = {
            "title_requested": title,
            "catalog_title": source_name.stem,
            "source": source_url,
            "catalog": "sevagh/musicxml-library",
            "rights": "Repository-designated public-domain MusicXML",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        target.with_suffix(target.suffix + ".source.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def _download_mutopia_score(self, title: str) -> Path | None:
        search_term = self._catalog_search_term(title)
        response = httpx.get(
            MUTOPIA_SEARCH_URL,
            params={"searchingfor": search_term},
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "ReachyAI/1.0"},
        )
        response.raise_for_status()
        source_url = self._select_mutopia_midi(
            title, self._extract_mutopia_midi_urls(response.text)
        )
        if source_url is None:
            return None

        parsed = urlparse(source_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.mutopiaproject.org"
            or not source_url.startswith(MUTOPIA_SCORE_PREFIX)
        ):
            raise RuntimeError("Mutopia 乐谱下载地址不在可信白名单内")
        download = httpx.get(
            source_url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "ReachyAI/1.0"},
        )
        download.raise_for_status()
        content = download.content
        if not content or len(content) > 5 * 1024 * 1024:
            raise RuntimeError("Mutopia 乐谱下载内容为空或过大")
        if not content.startswith(b"MThd"):
            raise RuntimeError("Mutopia 下载文件不是有效 MIDI")

        source_name = Path(parsed.path).name or "score.mid"
        safe_title = re.sub(r"[^0-9A-Za-z\u3400-\u9fff._-]+", "_", title).strip("._")
        target = self.config.score_root / f"{safe_title}__{source_name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        temporary.write_bytes(content)
        try:
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

        metadata = {
            "title_requested": title,
            "catalog_title": Path(source_name).stem,
            "source": source_url,
            "search": str(response.url),
            "catalog": "Mutopia Project",
            "rights": "Mutopia-designated public-domain or Creative Commons score",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        target.with_suffix(target.suffix + ".source.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def render(
        self,
        lyrics: str,
        *,
        title: str | None = None,
        score: Path | None = None,
    ) -> Path:
        self.validate()
        title = self.normalize_title(title) if title else None
        if title and score is None:
            score = self.resolve_score(title).path
        if title and score is not None:
            self.validate_singing_score(title, score)
        request_dir = self.config.output_root / str(time.time_ns())
        request_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.config.python),
            str(self.config.renderer),
            "--lyrics",
            # A named-song route often carries the title in `lyrics`.  Do not
            # sing the title itself as if it were the song text.  Embedded
            # MusicXML/MIDI lyrics or a local sidecar still take precedence.
            "啦" if title else lyrics,
            "--voice-bank",
            str(self.config.voice_bank),
            "--speaker",
            self.config.speaker,
            "--max-chars",
            str(self.config.max_chars),
            "--pitch-steps",
            str(self.config.pitch_steps),
            "--variance-steps",
            str(self.config.variance_steps),
            "--acoustic-steps",
            str(self.config.acoustic_steps),
            "--output",
            str(request_dir),
        ]
        if score is not None:
            command.extend(["--score", str(score), "--title", title or score.stem])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.config.timeout_seconds,
            check=False,
        )
        log_path = request_dir / "render.log"
        log_path.write_text(
            completed.stdout + "\n--- stderr ---\n" + completed.stderr,
            encoding="utf-8",
        )
        output = request_dir / "step5_final_audio.wav"
        if completed.returncode != 0 or not output.exists():
            tail = (completed.stderr or completed.stdout)[-1200:]
            raise RuntimeError(f"歌声合成失败（日志：{log_path}）：{tail}")
        return output
