from a2s.data.note_event import NoteEvent
from a2s.stage2_score.event_quantizer import quantize_events


def test_sec_to_beat_quantization():
    note = NoteEvent(60, onset_sec=0.13, offset_sec=0.62)
    result = quantize_events([note], tempo_bpm=120, subdivisions_per_beat=4)[0]
    assert result.onset_beat == 0.25
    assert result.offset_beat == 1.25
