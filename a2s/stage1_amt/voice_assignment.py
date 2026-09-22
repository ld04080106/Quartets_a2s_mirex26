from __future__ import annotations

import itertools
from dataclasses import replace

from a2s.data.note_event import NoteEvent, VOICES

CENTERS = {"violin_1": 79, "violin_2": 72, "viola": 64, "cello": 52}
RANGES = {"violin_1": (55, 105), "violin_2": (55, 100), "viola": (48, 88), "cello": (36, 76)}


def assign_voices(notes: list[NoteEvent], onset_group_sec: float = 0.04) -> list[NoteEvent]:
    result = [replace(note) for note in notes]
    unknown = sorted((note for note in result if note.voice == "unknown"), key=lambda n: (n.onset_sec if n.onset_sec is not None else n.onset_beat or 0.0, -n.pitch))
    last_pitch = dict(CENTERS)
    last_offset = {voice: -1e9 for voice in VOICES}
    index = 0
    while index < len(unknown):
        anchor = _onset(unknown[index])
        group: list[NoteEvent] = []
        while index < len(unknown) and _onset(unknown[index]) - anchor <= onset_group_sec:
            group.append(unknown[index])
            index += 1
        if len(group) > 4:
            assignments = _assign_dense_onset_group(group, last_pitch, last_offset)
            for note, voice in assignments:
                note.voice = voice
            for voice in VOICES:
                voice_notes = [note for note, assigned in assignments if assigned == voice]
                if voice_notes:
                    representative = max(voice_notes, key=lambda note: note.confidence)
                    last_pitch[voice] = representative.pitch
                    last_offset[voice] = max(_offset(note) for note in voice_notes)
            continue
        best_cost, best_voices = float("inf"), ()
        for voices in itertools.permutations(VOICES, len(group)):
            cost = sum(_assignment_cost(note, voice, last_pitch, last_offset) for note, voice in zip(group, voices))
            pitches_by_rank = [note.pitch for _, note in sorted(zip(voices, group), key=lambda pair: VOICES.index(pair[0]))]
            cost += sum(5.0 for a, b in zip(pitches_by_rank, pitches_by_rank[1:]) if a < b - 3)
            if cost < best_cost:
                best_cost, best_voices = cost, voices
        for note, voice in zip(group, best_voices):
            note.voice = voice
            last_pitch[voice] = note.pitch
            last_offset[voice] = _offset(note)
    return sorted(result, key=lambda n: (_onset(n), VOICES.index(n.voice) if n.voice in VOICES else 4, n.pitch))


def _assign_dense_onset_group(
    group: list[NoteEvent], last_pitch: dict[str, int], last_offset: dict[str, float],
) -> list[tuple[NoteEvent, str]]:
    """Assign double-stops/chords without silently leaving notes unknown."""
    assignments: list[tuple[NoteEvent, str]] = []
    counts = {voice: 0 for voice in VOICES}
    pitches = {voice: [] for voice in VOICES}
    capacity = max(2, (len(group) + len(VOICES) - 1) // len(VOICES))
    for note in sorted(group, key=lambda item: (-item.pitch, -item.confidence)):
        best_voice, best_cost = "unknown", float("inf")
        for voice in VOICES:
            if counts[voice] >= capacity:
                continue
            # Simultaneous overlap is expected for a double-stop, so only range,
            # center and melodic-continuity terms are used here.
            cost = _assignment_cost(note, voice, last_pitch, last_offset, ignore_overlap=True)
            cost += counts[voice] * 2.5
            if pitches[voice]:
                cost += min(abs(note.pitch - pitch) for pitch in pitches[voice]) * 0.08
            if cost < best_cost:
                best_voice, best_cost = voice, cost
        if best_voice == "unknown":
            best_voice = min(VOICES, key=lambda voice: counts[voice])
        assignments.append((note, best_voice))
        counts[best_voice] += 1
        pitches[best_voice].append(note.pitch)
    return assignments


def _onset(note: NoteEvent) -> float:
    return note.onset_sec if note.onset_sec is not None else note.onset_beat or 0.0


def _offset(note: NoteEvent) -> float:
    return note.offset_sec if note.offset_sec is not None else note.offset_beat or _onset(note)


def _assignment_cost(
    note: NoteEvent, voice: str, last_pitch: dict[str, int],
    last_offset: dict[str, float], ignore_overlap: bool = False,
) -> float:
    low, high = RANGES[voice]
    range_cost = max(0, low - note.pitch, note.pitch - high) * 4.0
    center_cost = abs(note.pitch - CENTERS[voice]) * 0.12
    jump_cost = abs(note.pitch - last_pitch[voice]) * 0.18
    overlap_cost = 0.0 if ignore_overlap else (20.0 if _onset(note) < last_offset[voice] - 0.02 else 0.0)
    return range_cost + center_cost + jump_cost + overlap_cost
