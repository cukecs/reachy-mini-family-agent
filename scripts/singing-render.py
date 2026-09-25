from __future__ import annotations

import argparse
import json
import re
import statistics
import wave
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import mido
from diffsinger_utau.voice_bank import PredAll
from diffsinger_utau.voice_bank.commons.ds_reader import DSReader, DSTextReplacer
from diffsinger_utau.voice_bank.commons.phome_num_counter import Phome


MELODY = ("C4", "D4", "E4", "G4", "A4", "G4", "E4", "D4")
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def chinese_characters(text: str, limit: int | None) -> list[str]:
    characters = re.findall(r"[\u3400-\u9fff]", text)
    if not characters:
        raise ValueError("目前的中文女声只接受中文歌词")
    return characters if limit is None else characters[:limit]


def _note_name(midi_note: int) -> str:
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


def _transpose_for_voice(notes: list[int]) -> list[int]:
    if not notes:
        return notes
    shifted = list(notes)
    median = statistics.median(shifted)
    while median < 60:
        shifted = [note + 12 for note in shifted]
        median += 12
    while median > 76:
        shifted = [note - 12 for note in shifted]
        median -= 12
    return [max(48, min(84, note)) for note in shifted]


def _musicxml_root(path: Path) -> ET.Element:
    if path.suffix.casefold() != ".mxl":
        return ET.parse(path).getroot()
    with zipfile.ZipFile(path) as archive:
        score_name = None
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is not None:
                score_name = rootfile.attrib.get("full-path")
        except KeyError:
            pass
        if not score_name:
            score_name = next(
                name
                for name in archive.namelist()
                if name.casefold().endswith((".musicxml", ".xml"))
                and "container.xml" not in name.casefold()
            )
        return ET.fromstring(archive.read(score_name))


def _read_musicxml(path: Path, fallback_lyrics: str, limit: int) -> tuple[list[str], list[str], list[float]]:
    root = _musicxml_root(path)
    best: tuple[list[str], list[int], list[float]] | None = None
    best_rank: tuple[int, int] | None = None
    for part in root.findall(".//{*}part"):
        divisions = 1.0
        tempo = 100.0
        tracks: dict[tuple[str, str], list[tuple[str, int, float]]] = {}
        for measure in part.findall("{*}measure"):
            divisions_node = measure.find(".//{*}attributes/{*}divisions")
            if divisions_node is not None and divisions_node.text:
                divisions = max(1.0, float(divisions_node.text))
            sound = measure.find(".//{*}direction/{*}sound")
            if sound is not None and sound.attrib.get("tempo"):
                tempo = max(30.0, min(240.0, float(sound.attrib["tempo"])))
            for note in measure.findall("{*}note"):
                if note.find("{*}rest") is not None or note.find("{*}chord") is not None:
                    continue
                pitch = note.find("{*}pitch")
                duration_node = note.find("{*}duration")
                if pitch is None or duration_node is None or not duration_node.text:
                    continue
                step = pitch.findtext("{*}step")
                octave_text = pitch.findtext("{*}octave")
                if not step or octave_text is None:
                    continue
                alter = int(float(pitch.findtext("{*}alter", "0")))
                semitone = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
                midi_note = 12 * (int(octave_text) + 1) + semitone + alter
                seconds = float(duration_node.text) / divisions * 60.0 / tempo
                lyric = note.findtext(".//{*}lyric/{*}text", "").strip()
                voice = note.findtext("{*}voice", "1")
                staff = note.findtext("{*}staff", "1")
                track = tracks.setdefault((voice, staff), [])
                tie_types = {
                    tie.attrib.get("type", "") for tie in note.findall("{*}tie")
                }
                if "stop" in tie_types and track and track[-1][1] == midi_note:
                    previous_lyric, _, previous_seconds = track[-1]
                    track[-1] = (
                        previous_lyric or lyric,
                        midi_note,
                        previous_seconds + seconds,
                    )
                else:
                    track.append((lyric, midi_note, seconds))
        for (voice, staff), entries in tracks.items():
            lyric_entries = [
                entry for entry in entries if re.search(r"[\u3400-\u9fff]", entry[0])
            ]
            selected = lyric_entries if len(lyric_entries) >= 2 else entries
            if not selected:
                continue
            # MusicXML piano scores interleave treble melody and bass voices in
            # one part.  Prefer an explicitly sung track, otherwise the normal
            # treble voice (voice 1, staff 1), instead of alternating melody
            # notes with left-hand accompaniment.
            priority = (
                0
                if len(lyric_entries) >= 2
                else 1
                if voice == "1" and staff == "1"
                else 2
                if staff == "1"
                else 3
            )
            rank = (priority, -len(selected))
            if best_rank is None or rank < best_rank:
                embedded = [entry[0] for entry in selected]
                best = (
                    embedded,
                    [entry[1] for entry in selected],
                    [entry[2] for entry in selected],
                )
                best_rank = rank
    if best is None:
        raise ValueError("MusicXML 中没有找到可演唱的旋律声部")
    embedded, midi_notes, durations = best
    embedded_chars = [
        chars[0]
        for text in embedded
        if (chars := re.findall(r"[\u3400-\u9fff]", text))
    ]
    sidecar = path.with_suffix(".txt")
    if sidecar.exists():
        chars = chinese_characters(sidecar.read_text(encoding="utf-8"), limit)
    elif len(embedded_chars) >= 2:
        chars = embedded_chars[:limit]
    else:
        fallback_chars = re.findall(r"[\u3400-\u9fff]", fallback_lyrics)
        # A named-song request normally passes only the title.  In that
        # case sing the authentic score on a neutral "la" syllable rather
        # than inventing lyrics or rejecting a valid instrumental score.
        chars = (
            fallback_chars[:limit]
            if len(fallback_chars) >= 5
            else ["啦"] * min(limit, len(midi_notes))
        )
    count = min(len(chars), len(midi_notes), limit)
    midi_notes = _transpose_for_voice(midi_notes[:count])
    return chars[:count], [_note_name(note) for note in midi_notes], durations[:count]


def _read_midi(path: Path, lyrics: str, limit: int) -> tuple[list[str], list[str], list[float]]:
    sidecar = path.with_suffix(".txt")
    if sidecar.exists():
        chars = chinese_characters(sidecar.read_text(encoding="utf-8"), limit)
    else:
        embedded_text = "".join(
            str(getattr(message, "text", ""))
            for track in mido.MidiFile(path).tracks
            for message in track
            if message.type in {"lyrics", "text"}
        )
        embedded_chars = re.findall(r"[\u3400-\u9fff]", embedded_text)
        fallback_chars = re.findall(r"[\u3400-\u9fff]", lyrics)
        if len(embedded_chars) >= 2:
            chars = embedded_chars[:limit]
        elif len(fallback_chars) >= 5:
            chars = fallback_chars[:limit]
        else:
            # The MIDI notes remain the real score; only the neutral syllable
            # is synthesized when no authorized Chinese lyrics are available.
            chars = ["啦"] * limit
    midi = mido.MidiFile(path)
    tempo = next(
        (
            message.tempo
            for track in midi.tracks
            for message in track
            if message.type == "set_tempo"
        ),
        mido.bpm2tempo(100),
    )
    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for track in midi.tracks:
        absolute = 0
        active: dict[int, int] = {}
        notes: list[tuple[int, int]] = []
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                active[message.note] = absolute
            elif message.type in {"note_off", "note_on"} and message.note in active:
                start = active.pop(message.note)
                notes.append((message.note, max(1, absolute - start)))
        if len(notes) >= min(2, len(chars)):
            name = str(getattr(track, "name", "")).casefold()
            priority = 0 if any(word in name for word in ("vocal", "voice", "melody", "soprano")) else 1
            candidates.append((priority, notes))
    if not candidates:
        raise ValueError("MIDI 中没有找到可演唱的旋律轨")
    _, note_events = min(candidates, key=lambda item: (item[0], abs(len(item[1]) - len(chars))))
    count = min(len(chars), len(note_events), limit)
    midi_notes = _transpose_for_voice([item[0] for item in note_events[:count]])
    durations = [
        mido.tick2second(item[1], midi.ticks_per_beat, tempo)
        for item in note_events[:count]
    ]
    return chars[:count], [_note_name(note) for note in midi_notes], durations


def build_score(lyrics: str, limit: int, score_path: Path | None = None) -> dict[str, str | int]:
    return build_scores(lyrics, limit, score_path)[0]


def _build_section(
    chars: list[str], notes: list[str], durations: list[float], offset: float
) -> dict[str, str | int | float]:
    durations = [max(0.18, min(1.80, float(value))) for value in durations]
    replacer = DSTextReplacer()
    phonemes = replacer.get_phonemes("".join(chars), replacer.opencpop_dict)
    phonemes = ["SP", *phonemes, "SP"]
    ph_num = Phome(phonemes).get_ph_num()
    expected_words = len(chars) + 2
    if len(ph_num) != expected_words:
        raise ValueError(
            f"歌词音素分组失败：{len(chars)} 个字得到 {len(ph_num) - 2} 个音节"
        )
    return {
        "offset": round(offset, 3),
        "text": " ".join(["SP", *chars, "SP"]),
        "ph_seq": " ".join(phonemes),
        "ph_num": " ".join(map(str, ph_num)),
        "note_seq": " ".join(["rest", *notes, "rest"]),
        "note_dur": " ".join(map(str, [0.18, *durations, 0.30])),
        "note_slur": " ".join(["0"] * expected_words),
        "lang": "zh",
    }


def build_scores(
    lyrics: str, chunk_size: int, score_path: Path | None = None
) -> list[dict[str, str | int | float]]:
    """Build every vocal section; chunk_size is no longer a total-song cap."""
    if score_path is None:
        chars = chinese_characters(lyrics, None)
        notes = [MELODY[index % len(MELODY)] for index in range(len(chars))]
        durations = [0.52] * len(chars)
        if durations:
            durations[-1] = 0.82
    elif score_path.suffix.casefold() in {".musicxml", ".xml", ".mxl"}:
        chars, notes, durations = _read_musicxml(score_path, lyrics, 4096)
    elif score_path.suffix.casefold() in {".mid", ".midi"}:
        chars, notes, durations = _read_midi(score_path, lyrics, 4096)
    else:
        raise ValueError(f"不支持的乐谱格式：{score_path.suffix}")
    count = min(len(chars), len(notes), len(durations))
    if count == 0:
        raise ValueError("乐谱中没有可演唱内容")
    sections: list[dict[str, str | int | float]] = []
    offset = 0.0
    for start in range(0, count, max(2, chunk_size)):
        end = min(count, start + max(2, chunk_size))
        section_durations = durations[start:end]
        sections.append(
            _build_section(chars[start:end], notes[start:end], section_durations, offset)
        )
        offset += 0.18 + sum(section_durations) + 0.30
    return sections


def _join_wavs(sources: list[Path], target: Path) -> None:
    if not sources:
        raise ValueError("没有生成任何歌声音频段")
    params = None
    chunks: list[bytes] = []
    for source_path in sources:
        with wave.open(str(source_path), "rb") as source:
            current = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
                source.getcomptype(),
            )
            if params is None:
                params = current
            elif current != params:
                raise ValueError("歌声音频段格式不一致，无法拼接")
            chunks.append(source.readframes(source.getnframes()))
    assert params is not None
    with wave.open(str(target), "wb") as output:
        output.setnchannels(params[0])
        output.setsampwidth(params[1])
        output.setframerate(params[2])
        output.setcomptype(params[3], "not compressed")
        for chunk in chunks:
            output.writeframes(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render short Chinese singing with DiffSinger")
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--voice-bank", type=Path, required=True)
    parser.add_argument("--speaker", default="Gentle")
    parser.add_argument("--max-chars", type=int, default=16)
    parser.add_argument("--pitch-steps", type=int, default=6)
    parser.add_argument("--variance-steps", type=int, default=6)
    parser.add_argument("--acoustic-steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    score_path = args.output / "score.ds"
    score_path.write_text(
        json.dumps(
            build_scores(args.lyrics, args.max_chars, args.score),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictor = PredAll(args.voice_bank)
    sections = DSReader(score_path).read_ds()
    sources: list[Path] = []
    for index, section in enumerate(sections):
        section_dir = args.output / f"section_{index:03d}"
        predictor.predict_full_pipeline(
            ds=section,
            lang="zh",
            speaker=args.speaker,
            pitch_steps=args.pitch_steps,
            variance_steps=args.variance_steps,
            acoustic_steps=args.acoustic_steps,
            output_dir=str(section_dir),
            save_intermediate=False,
        )
        sources.append(section_dir / "step5_final_audio.wav")
    target = args.output / "step5_final_audio.wav"
    _join_wavs(sources, target)
    print(target)


if __name__ == "__main__":
    main()
