from __future__ import annotations

import math
from collections import defaultdict

from a2s.data.note_event import NoteEvent, VOICES
from a2s.stage1_amt.voice_assignment import CENTERS, RANGES


FEATURE_NAMES = [
    "pitch", "pitch_class", "octave", "duration_sec", "onset_sec", "offset_sec",
    "velocity", "confidence", "sample_pos", "onset_frac_beat120",
    "group_size", "rank_high", "rank_low", "group_pitch_min", "group_pitch_max",
    "group_pitch_mean", "group_pitch_span", "above_gap", "below_gap",
    "prev_pitch_delta", "next_pitch_delta", "prev_onset_delta", "next_onset_delta",
    *[f"rule_voice_{voice}" for voice in VOICES],
    *[f"range_violation_{voice}" for voice in VOICES],
    *[f"center_distance_{voice}" for voice in VOICES],
]


def onset(note: NoteEvent) -> float:
    return float(note.onset_sec if note.onset_sec is not None else note.onset_beat or 0.0)


def offset(note: NoteEvent) -> float:
    if note.offset_sec is not None:
        return float(note.offset_sec)
    if note.offset_beat is not None:
        return float(note.offset_beat)
    return onset(note)


def duration(note: NoteEvent) -> float:
    return max(1e-6, offset(note) - onset(note))


def _onset_groups(notes: list[NoteEvent], onset_group_sec: float) -> dict[int, list[int]]:
    ordered = sorted(range(len(notes)), key=lambda i: (onset(notes[i]), -notes[i].pitch, i))
    groups: dict[int, list[int]] = {}
    group_id = 0
    index = 0
    while index < len(ordered):
        anchor = onset(notes[ordered[index]])
        group: list[int] = []
        while index < len(ordered) and onset(notes[ordered[index]]) - anchor <= onset_group_sec:
            group.append(ordered[index])
            index += 1
        groups[group_id] = group
        group_id += 1
    return groups


def feature_rows(notes: list[NoteEvent], onset_group_sec: float = 0.04) -> list[list[float]]:
    if not notes:
        return []
    n = len(notes)
    max_offset = max(offset(note) for note in notes) or 1.0
    groups = _onset_groups(notes, onset_group_sec)
    group_by_index: dict[int, list[int]] = {}
    for group in groups.values():
        for index in group:
            group_by_index[index] = group
    ordered = sorted(range(n), key=lambda i: (onset(notes[i]), notes[i].pitch, i))
    prev_index: dict[int, int | None] = {}
    next_index: dict[int, int | None] = {}
    for position, index in enumerate(ordered):
        prev_index[index] = ordered[position - 1] if position else None
        next_index[index] = ordered[position + 1] if position + 1 < len(ordered) else None

    rows: list[list[float]] = []
    for index, note in enumerate(notes):
        group = group_by_index.get(index, [index])
        group_pitches = sorted([notes[i].pitch for i in group], reverse=True)
        high_rank = group_pitches.index(note.pitch) if note.pitch in group_pitches else 0
        low_rank = len(group_pitches) - 1 - high_rank
        pitch_min, pitch_max = min(group_pitches), max(group_pitches)
        pitch_mean = sum(group_pitches) / len(group_pitches)
        above = [pitch for pitch in group_pitches if pitch > note.pitch]
        below = [pitch for pitch in group_pitches if pitch < note.pitch]
        prev_i, next_i = prev_index[index], next_index[index]
        note_onset = onset(note)
        row: list[float] = [
            float(note.pitch), float(note.pitch % 12), float(note.pitch // 12 - 1),
            duration(note), note_onset, offset(note),
            float(note.velocity if note.velocity is not None else 80),
            float(note.confidence), note_onset / max_offset,
            math.fmod(note_onset * 2.0, 1.0),  # approximate eighth-note phase at 120 BPM
            float(len(group)), float(high_rank), float(low_rank),
            float(pitch_min), float(pitch_max), float(pitch_mean),
            float(pitch_max - pitch_min),
            float(min(above) - note.pitch if above else 24.0),
            float(note.pitch - max(below) if below else 24.0),
            float(note.pitch - notes[prev_i].pitch if prev_i is not None else 0.0),
            float(notes[next_i].pitch - note.pitch if next_i is not None else 0.0),
            float(note_onset - onset(notes[prev_i]) if prev_i is not None else 0.0),
            float(onset(notes[next_i]) - note_onset if next_i is not None else 0.0),
        ]
        row.extend(1.0 if note.voice == voice else 0.0 for voice in VOICES)
        for voice in VOICES:
            low, high = RANGES[voice]
            row.append(float(max(0, low - note.pitch, note.pitch - high)))
        for voice in VOICES:
            row.append(float(abs(note.pitch - CENTERS[voice])))
        rows.append(row)
    return rows


def notes_by_onset_group(notes: list[NoteEvent], onset_group_sec: float = 0.04) -> list[list[int]]:
    groups = _onset_groups(notes, onset_group_sec)
    return [groups[key] for key in sorted(groups)]
