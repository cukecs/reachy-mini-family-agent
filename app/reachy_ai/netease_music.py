from __future__ import annotations

import base64
import ast
import csv
import ctypes
import json
import os
import re
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from collections import Counter


SEARCH_ENDPOINT = "https://music.163.com/api/search/get/web"
SONG_DETAIL_ENDPOINT = "https://music.163.com/api/song/detail"
DEFAULT_CDP_PORT = 9223


@dataclass(frozen=True)
class MusicCandidate:
    song_id: int
    name: str
    artists: tuple[str, ...]
    album: str = ""

    @property
    def artist_text(self) -> str:
        return " / ".join(self.artists)


@dataclass(frozen=True)
class MusicPlaybackResult:
    status: str
    title: str
    artist: str = ""
    candidate: MusicCandidate | None = None
    candidates: tuple[MusicCandidate, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class OriginalArtistResolution:
    artist: str | None
    confidence: str
    evidence_count: int = 0
    sources: tuple[str, ...] = ()


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z㐀-鿿]+", "", value)


_PERSON_CHARS = r"A-Za-zÀ-ÿ\u3400-\u9fff·・.'’ "
_ORIGINAL_ARTIST_PATTERNS = (
    # Require an explicit connector after 原唱. This deliberately rejects
    # questions such as “原唱是谁” and phrases such as “原唱身份”.
    re.compile(
        rf"原唱(?:歌手|者)?\s*(?:为|是|：|:)\s*"
        rf"([{_PERSON_CHARS}]{{2,30}}?)"
        rf"(?=[，。；;、）)\]】]|(?:\s(?:于|在|并|收录|演唱|翻唱|作词|作曲))|$)",
        re.IGNORECASE,
    ),
    # Common cover-title form: （原唱 周杰伦）.
    re.compile(
        rf"[（(]\s*原唱\s*(?:：|:)?\s*"
        rf"([{_PERSON_CHARS}]{{2,30}}?)\s*[）)]",
        re.IGNORECASE,
    ),
    re.compile(
        rf"由\s*([^由，。；;、]{{2,30}}?)\s*(?:首唱|原唱|原主唱|主唱|演唱)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"([{_PERSON_CHARS}]{{2,30}}?)\s*(?:原主唱|原唱者)"
        rf"(?=[，。；;、）)\]】]|\s|$)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"[（(]\s*\d{{4}}年\s*([{_PERSON_CHARS}]{{2,30}}?)\s*演唱歌曲[）)]",
        re.IGNORECASE,
    ),
)

_INVALID_ARTIST_NAMES = {
    "身份", "是谁", "谁", "歌手", "演唱者", "原唱", "首唱", "版本",
    "作者", "介绍", "资料", "这首歌", "歌曲", "原版",
}


def _clean_artist_name(value: str) -> str:
    value = re.sub(
        r"^(?:这首歌|该曲|歌曲|后来|随后|后经|最早|首次|正式|版本|歌手|由)+",
        "",
        value.strip(),
    )
    value = re.sub(r"^\d{4}年(?:由)?", "", value)
    value = re.sub(r"^.*?交由", "", value)
    value = value.strip(" ：:，。；;、-—()（）[]【】")
    if value in _INVALID_ARTIST_NAMES or any(
        token in value for token in ("原唱身份", "原唱是谁", "谁是原唱")
    ):
        return ""
    return value


def _target_title_segments(text: str, title: str) -> list[str]:
    """Keep only text near the requested title, stopping before the next 《song》."""
    if not title:
        return [text]
    segments: list[str] = []
    for match in re.finditer(re.escape(title), text, re.IGNORECASE):
        start = max(0, match.start() - 12)
        end = min(len(text), match.end() + 180)
        next_song = text.find("《", match.end())
        if next_song >= 0 and next_song < end:
            end = next_song
        segments.append(text[start:end])
    return segments


def extract_original_artist_names(text: str, title: str = "") -> list[str]:
    """Extract original/first-singer claims that are grammatically tied to title."""
    found: list[str] = []

    if not title:
        patterns = _ORIGINAL_ARTIST_PATTERNS
    else:
        target = re.escape(title)
        patterns = (
            # 《晴天》原唱：周杰伦 / 晴天 原唱是周杰伦
            re.compile(
                rf"(?:《\s*{target}\s*》|{target})"
                rf"[^。；;\n《]{{0,70}}?原唱(?:歌手|者)?\s*"
                rf"(?:为|是|：|:|\s)\s*([{_PERSON_CHARS}]{{2,30}}?)"
                rf"(?=[，。；;、）)\]】《-]|\s(?:于|在|并|收录|演唱|翻唱|作词|作曲)|$)",
                re.IGNORECASE,
            ),
            # 《月亮代表我的心》……1972年由陈芬兰首唱
            re.compile(
                rf"(?:《\s*{target}\s*》|{target})"
                rf"[^。；;\n]{{0,180}}?由\s*([^由，。；;、]{{2,30}}?)\s*"
                rf"(?:首唱|原唱|原主唱|主唱|演唱)",
                re.IGNORECASE,
            ),
            # 晴天（2003年周杰伦演唱的流行摇滚歌曲）
            re.compile(
                rf"{target}\s*[（(]\s*(?:\d{{4}}年)?\s*"
                rf"([{_PERSON_CHARS}]{{2,30}}?)\s*演唱(?:的)?"
                rf"[^）)]{{0,24}}歌曲[）)]",
                re.IGNORECASE,
            ),
            # 《晴天》是由周杰伦创作词曲、编曲并演唱的歌曲
            re.compile(
                rf"(?:《\s*{target}\s*》|{target})[^。；;]{{0,60}}?"
                rf"(?:是|为)(?:由)?\s*([{_PERSON_CHARS}]{{2,30}}?)"
                rf"(?=\s*(?:创作词曲|创作|演唱|主唱|作词|作曲|填词|谱曲|编曲))",
                re.IGNORECASE,
            ),
            # 晴天 (原唱 周杰伦)
            re.compile(
                rf"{target}\s*[（(]\s*原唱\s*(?:：|:)?\s*"
                rf"([{_PERSON_CHARS}]{{2,30}}?)\s*[）)]",
                re.IGNORECASE,
            ),
        )

    for pattern in patterns:
        for match in pattern.finditer(text):
            artist = _clean_artist_name(match.group(1))
            if 1 < len(artist) <= 30:
                found.append(artist)
    return found


def choose_original_artist(
    title: str,
    search_results: list[dict],
) -> OriginalArtistResolution:
    """Vote across independent web results instead of trusting one snippet."""
    source_names: dict[str, set[str]] = {}
    source_titles: dict[str, str] = {}
    for index, item in enumerate(search_results):
        raw_source = str(item.get("url") or item.get("href") or "")
        if raw_source:
            host = urllib.parse.urlparse(raw_source).netloc.casefold()
            source = host or raw_source
        else:
            source = f"result:{index}"
        item_title = str(item.get("title") or "")
        item_body = str(item.get("body") or item.get("snippet") or "")
        if source in source_names:
            text = f"{item_title}。{item_body}"
            source_names[source].update(extract_original_artist_names(text, title))
            continue
        text = f"{item_title}。{item_body}"
        names = set(extract_original_artist_names(text, title))
        if not names and _normalized(title) in _normalized(item_title):
            bracketed_titles = re.findall(r"《([^》]{1,80})》", item_body)
            other_titles = [
                value
                for value in bracketed_titles
                if _normalized(value) != _normalized(title)
            ]
            if not other_titles:
                names.update(extract_original_artist_names(item_body, ""))
        if names:
            source_names[source] = names
            source_titles[source] = item_title or source

    votes: Counter[str] = Counter()
    normalized_to_display: dict[str, str] = {}
    normalized_sources: dict[str, list[str]] = {}
    for source, names in source_names.items():
        for name in names:
            key = _normalized(name)
            if not key:
                continue
            votes[key] += 1
            normalized_to_display.setdefault(key, name)
            normalized_sources.setdefault(key, []).append(source_titles[source])
    if not votes:
        return OriginalArtistResolution(None, "none")

    ranked = votes.most_common()
    top_key, top_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    # Two independent results are enough when they agree. One explicit result
    # is retained as low confidence for obscure songs, but is not auto-played.
    if top_count >= 2 and top_count > second_count:
        return OriginalArtistResolution(
            normalized_to_display[top_key],
            "high",
            top_count,
            tuple(normalized_sources[top_key]),
        )
    return OriginalArtistResolution(
        normalized_to_display[top_key],
        "low",
        top_count,
        tuple(normalized_sources[top_key]),
    )


def select_exact_candidate(
    candidates: list[MusicCandidate],
    title: str,
    artist: str = "",
) -> tuple[MusicCandidate | None, tuple[MusicCandidate, ...]]:
    """Prefer an exact title/artist match when one is available."""
    wanted_title = _normalized(title)
    exact_title = [
        item for item in candidates if _normalized(item.name) == wanted_title
    ]
    if artist.strip():
        wanted_artist = _normalized(artist)
        exact_artist = [
            item
            for item in exact_title
            if any(_normalized(name) == wanted_artist for name in item.artists)
        ]
        return (
            exact_artist[0] if exact_artist else None,
            tuple(exact_artist or exact_title),
        )
    if len(exact_title) == 1:
        return exact_title[0], tuple(exact_title)
    return None, tuple(exact_title)


class NeteaseMusicController:
    """Play finished tracks through the signed-in official NetEase client."""

    def __init__(self, cdp_port: int = DEFAULT_CDP_PORT):
        self.cdp_port = int(cdp_port)

    @staticmethod
    def _search_original_artist_web(title: str) -> list[dict]:
        """Search multiple providers and reject results unrelated to the title."""
        import asyncio

        from .network_tools import _call_mcp, _configuration, _searxng_search

        queries = (
            f"{title} 原唱 首唱",
            f"{title} 原唱",
            f"{title} 最早 演唱",
        )
        results: list[dict] = []
        seen: set[str] = set()

        wanted = _normalized(title)

        def is_relevant(item: dict) -> bool:
            haystack = (
                str(item.get("title") or "")
                + " "
                + str(
                    item.get("snippet")
                    or item.get("body")
                    or item.get("content")
                    or ""
                )
            )
            return bool(wanted and wanted in _normalized(haystack))

        def add_results(batch: list[dict]) -> None:
            for item in batch:
                if not is_relevant(item):
                    continue
                url = str(item.get("url") or item.get("href") or "").strip()
                key = url or (
                    str(item.get("title") or "").strip()
                    + "\n"
                    + str(
                        item.get("snippet")
                        or item.get("body")
                        or item.get("content")
                        or ""
                    ).strip()
                )
                if not key or key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "title": str(item.get("title") or "").strip(),
                        "body": str(
                            item.get("snippet")
                            or item.get("body")
                            or item.get("content")
                            or ""
                        ).strip(),
                        "url": url,
                    }
                )

        try:
            _, tools = _configuration()
            searxng_url = (
                tools.get("search", {}).get("searxng_url")
                or "https://search.mectov.my.id/"
            )
        except Exception:
            searxng_url = "https://search.mectov.my.id/"

        for query in queries:
            try:
                raw = asyncio.run(
                    asyncio.wait_for(
                        _searxng_search(searxng_url, query, 8),
                        timeout=8.0,
                    )
                )
                payload = json.loads(raw)
                before = len(results)
                add_results(list(payload.get("results") or []))
            except Exception:
                continue
            if choose_original_artist(title, results).confidence == "high":
                return results
            if len(results) == before:
                # Provider returned data, but none of it even mentioned the
                # requested title.  Do not waste more point-song latency on
                # two variants of the same broken source.
                break

        # The public SearXNG instance can occasionally return unrelated Bing
        # utility pages for Chinese queries.  Use the project's hosted Reachy
        # search MCP before falling back to DDGS.
        try:
            _, tools = _configuration()
            search_tool = tools["search"]
            raw = asyncio.run(
                asyncio.wait_for(
                    _call_mcp(
                        search_tool["url"],
                        search_tool["remote_name"],
                        {"query": queries[0], "max_results": 8},
                    ),
                    timeout=18.0,
                )
            )
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                payload = ast.literal_eval(raw)
            if isinstance(payload, dict):
                add_results(list(payload.get("results") or []))
        except Exception:
            pass
        if choose_original_artist(title, results).confidence == "high":
            return results

        # Last resort: one short DDGS attempt.
        try:
            from ddgs import DDGS

            batch = DDGS(timeout=8).text(
                queries[0],
                region="cn-zh",
                safesearch="moderate",
                max_results=8,
            )
            add_results(list(batch))
        except Exception:
            pass
        return results

    def resolve_original_artist(self, title: str) -> OriginalArtistResolution:
        try:
            results = self._search_original_artist_web(title)
        except Exception:
            return OriginalArtistResolution(None, "none")
        return choose_original_artist(title, results)

    @staticmethod
    def _is_client_running() -> bool:
        if os.name != "nt":
            return False
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq cloudmusic.exe", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=flags,
            )
            return "cloudmusic.exe" in completed.stdout.casefold()
        except Exception:
            return False

    @staticmethod
    def _cloudmusic_pids() -> set[int]:
        if os.name != "nt":
            return set()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq cloudmusic.exe", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=flags,
            )
        except Exception:
            return set()
        pids: set[int] = set()
        for row in csv.reader(completed.stdout.splitlines()):
            if len(row) < 2 or row[0].casefold() != "cloudmusic.exe":
                continue
            try:
                pids.add(int(row[1]))
            except ValueError:
                continue
        return pids

    @classmethod
    def _keep_client_in_background(cls) -> None:
        """Hide NetEase windows so playback stays tray-only."""
        if os.name != "nt":
            return
        pids = cls._cloudmusic_pids()
        if not pids:
            return
        user32 = ctypes.windll.user32
        process_id = ctypes.c_ulong()
        callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )

        def callback(hwnd, _lparam):
            process_id.value = 0
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if process_id.value in pids and user32.IsWindowVisible(hwnd):
                # SW_HIDE removes the main window from the taskbar while the
                # NetEase tray process keeps running.
                user32.ShowWindowAsync(hwnd, 0)
            return True

        user32.EnumWindows(callback_type(callback), 0)

    @staticmethod
    def _hidden_startupinfo():
        if os.name != "nt":
            return None
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        return startupinfo

    @classmethod
    def _popen_background(cls, args: list[str], *, cwd: str) -> subprocess.Popen:
        return subprocess.Popen(
            args,
            cwd=cwd,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            startupinfo=cls._hidden_startupinfo(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _cdp_ready(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.cdp_port}/json", timeout=1
            ) as response:
                targets = json.load(response)
            return any(
                item.get("type") == "page"
                and str(item.get("url", "")).startswith("orpheus://")
                for item in targets
            )
        except Exception:
            return False

    def ensure_client(self) -> bool:
        """Ensure the client is running hidden with localhost CDP enabled."""
        exe = self._cloudmusic_exe()
        if not exe.is_file():
            return False
        if self._is_client_running() and self._cdp_ready():
            self._keep_client_in_background()
            return True
        if self._is_client_running() and not self._cdp_ready():
            try:
                subprocess.run(
                    ["taskkill", "/IM", "cloudmusic.exe", "/T", "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=8,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                return False
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and self._is_client_running():
                time.sleep(0.15)
        self._popen_background(
            [
                str(exe),
                "--remote-debugging-address=127.0.0.1",
                f"--remote-debugging-port={self.cdp_port}",
            ],
            cwd=str(exe.parent),
        )
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self._cdp_ready():
                self._keep_client_in_background()
                return True
            time.sleep(0.25)
        running = self._is_client_running() and self._cdp_ready()
        if running:
            self._keep_client_in_background()
        return running

    @staticmethod
    def _cloudmusic_exe() -> Path:
        configured = os.environ.get("CLOUDMUSIC_EXE")
        candidates = [
            Path(configured) if configured else None,
            Path("D:/Program Files/NetEase/CloudMusic/cloudmusic.exe"),
            Path("C:/Program Files/NetEase/CloudMusic/cloudmusic.exe"),
            Path("C:/Program Files (x86)/NetEase/CloudMusic/cloudmusic.exe"),
            Path.home() / "AppData" / "Local" / "NetEase" / "CloudMusic" / "cloudmusic.exe",
        ]
        return next(
            (path for path in candidates if path is not None and path.is_file()),
            Path("D:/Program Files/NetEase/CloudMusic/cloudmusic.exe"),
        )

    def search(self, query: str, limit: int = 10) -> list[MusicCandidate]:
        cleaned = query.strip()
        if not cleaned:
            return []
        body = urllib.parse.urlencode(
            {
                "s": cleaned,
                "type": 1,
                "offset": 0,
                "total": "true",
                "limit": max(1, min(int(limit), 20)),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            SEARCH_ENDPOINT,
            data=body,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://music.163.com/",
            },
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.load(response)
        if payload.get("code") != 200:
            raise RuntimeError(f"网易云搜索失败：code={payload.get('code')!r}")
        songs = (payload.get("result") or {}).get("songs") or []
        return [
            MusicCandidate(
                song_id=int(item["id"]),
                name=str(item.get("name") or ""),
                artists=tuple(
                    str(artist.get("name") or "")
                    for artist in item.get("artists", [])
                    if artist.get("name")
                ),
                album=str((item.get("album") or {}).get("name") or ""),
            )
            for item in songs
        ]

    @staticmethod
    def _play_url(song_id: int) -> str:
        payload = json.dumps(
            {"type": "song", "id": str(int(song_id)), "cmd": "play"},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "orpheus://" + base64.b64encode(payload).decode("ascii")

    def _cdp_evaluate(self, expression: str, *, await_promise: bool = False):
        try:
            import websocket
        except ImportError:
            return None
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.cdp_port}/json", timeout=1.5
            ) as response:
                targets = json.load(response)
            target = next(
                (
                    item
                    for item in targets
                    if item.get("type") == "page"
                    and str(item.get("url", "")).startswith("orpheus://")
                ),
                None,
            )
            if not target:
                return None
            connection = websocket.create_connection(
                target["webSocketDebuggerUrl"],
                timeout=3,
                suppress_origin=True,
            )
            try:
                connection.send(
                    json.dumps(
                        {
                            "id": 1,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": expression,
                                "returnByValue": True,
                                "awaitPromise": bool(await_promise),
                            },
                        }
                    )
                )
                while True:
                    message = json.loads(connection.recv())
                    if message.get("id") == 1:
                        result = message.get("result") or {}
                        if result.get("exceptionDetails"):
                            return None
                        return (result.get("result") or {}).get("value")
            finally:
                connection.close()
        except Exception:
            return None

    def _ensure_client_runtime(self) -> bool:
        if self._cdp_evaluate("typeof __gpt_cm_require__==='function'"):
            return True
        return bool(
            self._cdp_evaluate(
                "webpackJsonp.push([[987660],{987660:function(module,exports,require){"
                "window.__gpt_cm_require__=require;}},[[987660]]]);true"
            )
        )

    @staticmethod
    def _fetch_track_detail(song_id: int) -> dict | None:
        params = urllib.parse.urlencode(
            {"ids": json.dumps([int(song_id)], separators=(",", ":"))}
        )
        request = urllib.request.Request(
            f"{SONG_DETAIL_ENDPOINT}?{params}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://music.163.com/",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.load(response)
        except Exception:
            return None
        songs = payload.get("songs") or []
        return songs[0] if songs else None

    def _play_song_via_cdp(self, song_id: int) -> bool:
        track = self._fetch_track_detail(song_id)
        if not track or not self._ensure_client_runtime():
            return False
        payload = {
            "tracks": [track],
            "from": {
                "text": "Reachy AI",
                "href": "",
                "resourceType": "track",
                "scene": "search",
                "fromInfo": {
                    "originalScene": "search",
                    "originalResourceType": "track",
                    "computeSourceResourceType": "track",
                    "sourceData": {},
                },
            },
            "options": {
                "clear": True,
                "play": True,
                "playId": int(song_id),
                "offset": -1,
            },
            "triggerScene": "search",
            "triggerAction": "playAll",
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result = self._cdp_evaluate(
            "(async()=>{const r=__gpt_cm_require__,c=r.c||{};let api=null;"
            "for(const id of Object.keys(c)){try{const e=c[id]&&c[id].exports;"
            "const vals=[e];if(e&&typeof e==='object')vals.push(...Object.values(e));"
            "for(const v of vals){if(v&&(typeof v==='object'||typeof v==='function')"
            "&&typeof v.getStore==='function'&&typeof v.getDispatch==='function'){"
            "api=v;break;}}}catch(_){}if(api)break;}"
            "if(!api)throw new Error('NetEase store module not found');"
            "const d=api.getDispatch();"
            f"await d({{type:'playing/play',payload:{serialized}}});"
            "return true})()",
            await_promise=True,
        )
        self._keep_client_in_background()
        return bool(result)

    def stop(self) -> bool:
        """Stop NetEase playback and clear its queue without touching other media."""
        if not self._is_client_running() or not self._cdp_ready():
            return False
        if not self._ensure_client_runtime():
            return False
        result = self._cdp_evaluate(
            "(async()=>{const r=__gpt_cm_require__,c=r.c||{};let api=null;"
            "for(const id of Object.keys(c)){try{const e=c[id]&&c[id].exports;"
            "const vals=[e];if(e&&typeof e==='object')vals.push(...Object.values(e));"
            "for(const v of vals){if(v&&(typeof v==='object'||typeof v==='function')"
            "&&typeof v.getStore==='function'&&typeof v.getDispatch==='function'){"
            "api=v;break;}}}catch(_){}if(api)break;}"
            "if(!api)throw new Error('NetEase store module not found');"
            "const d=api.getDispatch();"
            "await d({type:'playing/pause',payload:{}});"
            "await d({type:'playingList/clearCurPlayingList',payload:{"
            "triggerScene:'playingList'}});return true})()",
            await_promise=True,
        )
        self._keep_client_in_background()
        return bool(result)

    def _dismiss_no_source_popup(self) -> None:
        self._cdp_evaluate(
            """(()=>{const b=Array.from(document.querySelectorAll('button')).find(
            x=>(x.innerText||'').trim()==='知道了');if(b)b.click();return true})()"""
        )

    def _no_source_visible(self) -> bool:
        text = self._cdp_evaluate(
            "String(document.body && document.body.innerText || '').slice(-2500)"
        )
        return bool(
            isinstance(text, str)
            and (
                "当前歌曲暂无音源" in text
                or "该资源暂时无法收听" in text
                or "因合作方要求" in text
            )
        )

    def play(self, title: str, artist: str = "") -> MusicPlaybackResult:
        exe = self._cloudmusic_exe()
        if not exe.is_file():
            return MusicPlaybackResult(
                "client_missing",
                title,
                artist,
                message="没有找到网易云音乐 Windows 客户端。",
            )
        if not self.ensure_client():
            return MusicPlaybackResult(
                "error",
                title,
                artist,
                message="网易云音乐客户端没有成功启动。",
            )

        explicit_artist = artist.strip()
        resolved_artist = explicit_artist
        if not explicit_artist:
            resolution = self.resolve_original_artist(title)
            if resolution.artist and resolution.confidence == "high":
                resolved_artist = resolution.artist
            else:
                # Original-artist research improves matching but must never
                # block a normal point-song request.
                resolved_artist = ""

        query = f"{title} {resolved_artist}".strip()
        try:
            candidates = self.search(query, limit=15)
        except Exception as exc:
            return MusicPlaybackResult(
                "error",
                title,
                resolved_artist,
                message=f"网易云搜索失败：{exc}",
            )

        selected, exact = select_exact_candidate(
            candidates, title, resolved_artist
        )
        fallback_used = False
        if selected is None:
            # Matching is best-effort, never a playback gate. If an automatically
            # resolved artist has no exact catalog hit, broaden back to title-only.
            if resolved_artist and not explicit_artist:
                try:
                    candidates = self.search(title, limit=15)
                except Exception as exc:
                    return MusicPlaybackResult(
                        "error",
                        title,
                        resolved_artist,
                        message=f"网易云回退搜索失败：{exc}",
                    )
            selected = candidates[0] if candidates else None
            fallback_used = selected is not None

        if selected is None:
            return MusicPlaybackResult(
                "not_found",
                title,
                resolved_artist,
                candidates=tuple(candidates[:5]),
                message="网易云没有返回可播放候选。",
            )

        self._dismiss_no_source_popup()
        self._keep_client_in_background()
        if not self._play_song_via_cdp(selected.song_id):
            return MusicPlaybackResult(
                "error",
                title,
                resolved_artist,
                candidate=selected,
                message="网易云后台控制通道没有成功播放这首歌。",
            )
        time.sleep(0.35)
        self._keep_client_in_background()
        time.sleep(0.85)
        self._keep_client_in_background()
        result_artist = (
            selected.artist_text
            if fallback_used or not resolved_artist
            else resolved_artist
        )
        if self._no_source_visible():
            return MusicPlaybackResult(
                "unavailable",
                title,
                result_artist,
                candidate=selected,
                message="网易云客户端提示这首歌当前暂无音源。",
            )
        return MusicPlaybackResult(
            "playing",
            title,
            result_artist,
            candidate=selected,
            message=(
                "原唱未可靠确认，已按网易云搜索结果第一条播放。"
                if fallback_used
                else "已交给网易云官方客户端后台播放。"
            ),
        )
