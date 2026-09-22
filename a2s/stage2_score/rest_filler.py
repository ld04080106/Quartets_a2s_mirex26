from __future__ import annotations

from fractions import Fraction

from a2s.data.note_event import NoteEvent

from .duration_spelling import spell_duration, token_duration
from .score_grid import Segment


def _fraction(value: float) -> Fraction:
    return Fraction(value).limit_denominator(192)


def make_voice_segments(notes: list[NoteEvent], total_beats: Fraction) -> list[Segment]:
    groups: dict[Fraction, list[NoteEvent]] = {}
    for note in notes:
        if note.onset_beat is None or note.offset_beat is None:
            continue
        groups.setdefault(_fraction(note.onset_beat), []).append(note)
    onsets = sorted(groups)
    segments: list[Segment] = []
    cursor = Fraction(0)
    for index, onset in enumerate(onsets):
        group = sorted(groups[onset], key=lambda n: (-n.confidence, n.pitch))
        if onset < cursor:
            continue
        if onset > cursor:
            segments.extend(_split_spellable(Segment(cursor, onset - cursor)))
        next_onset = onsets[index + 1] if index + 1 < len(onsets) else total_beats
        desired_offset = max(_fraction(note.offset_beat or float(onset)) for note in group)
        offset = min(max(onset + Fraction(1, 16), desired_offset), next_onset, total_beats)
        if offset <= onset:
            continue
        pitches = tuple(sorted({note.pitch for note in group}))
        segments.extend(_split_spellable(Segment(onset, offset - onset, pitches)))
        cursor = offset
    if cursor < total_beats:
        segments.extend(_split_spellable(Segment(cursor, total_beats - cursor)))
    return segments


def _split_spellable(segment: Segment) -> list[Segment]:
    result: list[Segment] = []
    cursor = segment.onset
    tokens = spell_duration(segment.duration)
    for index, token in enumerate(tokens):
        duration = token_duration(token)
        result.append(Segment(
            cursor, duration, segment.pitches,
            tie_from=bool(segment.pitches) and (segment.tie_from or index > 0),
            tie_to=bool(segment.pitches) and (segment.tie_to or index < len(tokens) - 1),
        ))
        cursor += duration
    return result
