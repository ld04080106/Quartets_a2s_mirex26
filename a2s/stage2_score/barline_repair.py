from __future__ import annotations

from fractions import Fraction

from .duration_spelling import spell_duration, token_duration
from .score_grid import Segment


def split_at_barlines(segments: list[Segment], measure_beats: Fraction) -> list[Segment]:
    result: list[Segment] = []
    for segment in segments:
        cursor, end = segment.onset, segment.offset
        parts: list[tuple[Fraction, Fraction]] = []
        while cursor < end:
            next_bar = (cursor // measure_beats + 1) * measure_beats
            part_end = min(end, next_bar)
            parts.append((cursor, part_end - cursor))
            cursor = part_end
        expanded: list[tuple[Fraction, Fraction]] = []
        for onset, duration in parts:
            part_cursor = onset
            for token in spell_duration(duration):
                value = token_duration(token)
                expanded.append((part_cursor, value))
                part_cursor += value
        for index, (onset, duration) in enumerate(expanded):
            result.append(Segment(
                onset, duration, segment.pitches,
                tie_from=bool(segment.pitches) and (segment.tie_from or index > 0),
                tie_to=bool(segment.pitches) and (segment.tie_to or index < len(expanded) - 1),
            ))
    return result
