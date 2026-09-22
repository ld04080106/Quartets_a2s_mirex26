from fractions import Fraction

from a2s.data.note_event import NoteEvent
from a2s.stage2_score.rest_filler import make_voice_segments


def test_rest_filling_closes_voice():
    segments = make_voice_segments([NoteEvent(60, "violin_1", onset_beat=1, offset_beat=2)], Fraction(4))
    assert segments[0].is_rest
    assert segments[-1].offset == 4
    assert sum((segment.duration for segment in segments), Fraction()) == 4
