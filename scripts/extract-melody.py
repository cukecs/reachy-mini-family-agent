from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import mido
import numpy as np


def compact_events(f0: np.ndarray, voiced: np.ndarray, hop_seconds: float) -> list[tuple[int, float]]:
    midi = np.rint(librosa.hz_to_midi(f0)).astype(float)
    midi[~voiced | ~np.isfinite(midi)] = np.nan
    events: list[tuple[int, float]] = []
    current: list[float] = []
    gaps = 0

    def finish() -> None:
        nonlocal current, gaps
        duration = len(current) * hop_seconds
        if current and duration >= 0.12:
            note = int(round(float(np.nanmedian(current))))
            if 36 <= note <= 96:
                if events and events[-1][0] == note:
                    events[-1] = (note, events[-1][1] + duration)
                else:
                    events.append((note, duration))
        current = []
        gaps = 0

    for value in midi:
        if not np.isfinite(value):
            gaps += 1
            if gaps > 2:
                finish()
            continue
        gaps = 0
        if current and abs(value - float(np.nanmedian(current[-8:]))) >= 1.0:
            finish()
        current.append(float(value))
    finish()
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a monophonic melody from clean vocal/humming audio")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    samples, rate = librosa.load(args.input, sr=22050, mono=True)
    hop = 256
    f0, voiced, probabilities = librosa.pyin(
        samples,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=rate,
        hop_length=hop,
    )
    voiced = np.asarray(voiced, dtype=bool)
    confidence = float(np.nanmedian(np.asarray(probabilities)[voiced])) if np.any(voiced) else 0.0
    ratio = float(np.mean(voiced)) if len(voiced) else 0.0
    events = compact_events(np.asarray(f0), voiced, hop / rate)
    if confidence < 0.62 or ratio < 0.25 or len(events) < 8:
        raise RuntimeError(
            f"音频不像清晰的单声部人声/哼唱（confidence={confidence:.2f}, voiced={ratio:.2f}, notes={len(events)}）"
        )

    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.name = "vocal melody"
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90), time=0))
    seconds_per_tick = mido.tick2second(1, midi.ticks_per_beat, mido.bpm2tempo(90))
    for note, duration in events:
        ticks = max(1, int(round(duration / seconds_per_tick)))
        track.append(mido.Message("note_on", note=note, velocity=80, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=ticks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(args.output)
    args.output.with_suffix(args.output.suffix + ".analysis.json").write_text(
        json.dumps(
            {"confidence": confidence, "voiced_ratio": ratio, "notes": len(events)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
