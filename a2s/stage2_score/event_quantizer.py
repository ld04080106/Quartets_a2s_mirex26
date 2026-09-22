from __future__ import annotations

from dataclasses import replace

from a2s.data.note_event import NoteEvent

from .temporal_alignment import align_event_boundaries


def sec_to_beat(seconds: float, tempo_bpm: float, origin_sec: float = 0.0) -> float:
    return max(0.0, (seconds - origin_sec) * tempo_bpm / 60.0)


def quantize_value(value: float, subdivisions_per_beat: int = 4) -> float:
    return round(value * subdivisions_per_beat) / subdivisions_per_beat


def quantize_events(
    notes: list[NoteEvent], tempo_bpm: float = 120.0,
    subdivisions_per_beat: int = 4, min_duration_beat: float | None = None,
    alignment_tolerance_beat: float = 0.0,
    alignment_max_span_beat: float | None = None,
) -> list[NoteEvent]:
    minimum = min_duration_beat or 1.0 / subdivisions_per_beat
    beat_notes: list[NoteEvent] = []
    for note in notes:
        onset, offset = note.onset_beat, note.offset_beat
        if onset is None and note.onset_sec is not None:
            onset = sec_to_beat(note.onset_sec, tempo_bpm)
        if offset is None and note.offset_sec is not None:
            offset = sec_to_beat(note.offset_sec, tempo_bpm)
        if onset is None or offset is None:
            continue
        beat_notes.append(replace(
            note, onset_beat=onset, offset_beat=offset, duration_beat=offset - onset
        ))
    aligned = align_event_boundaries(
        beat_notes, alignment_tolerance_beat, alignment_max_span_beat
    )
    result: list[NoteEvent] = []
    for note in aligned:
        onset, offset = note.onset_beat, note.offset_beat
        if onset is None or offset is None:
            continue
        q_on = quantize_value(onset, subdivisions_per_beat)
        q_off = max(q_on + minimum, quantize_value(offset, subdivisions_per_beat))
        result.append(replace(note, onset_beat=q_on, offset_beat=q_off, duration_beat=q_off - q_on))
    return sorted(result, key=lambda n: (n.onset_beat or 0.0, n.voice, -n.confidence, n.pitch))
