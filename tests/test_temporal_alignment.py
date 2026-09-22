from a2s.data.note_event import NoteEvent
from a2s.stage2_score.temporal_alignment import align_event_boundaries


def test_nearby_cross_voice_boundaries_are_aligned() -> None:
    notes = [
        NoteEvent(72, "violin_1", 0.96, 1.96, 1.0, confidence=0.9),
        NoteEvent(67, "violin_2", 1.04, 2.05, 1.01, confidence=0.8),
    ]
    aligned = align_event_boundaries(notes, tolerance_beat=0.1, max_cluster_span_beat=0.2)
    assert aligned[0].onset_beat == aligned[1].onset_beat
    assert aligned[0].offset_beat == aligned[1].offset_beat


def test_same_note_onset_and_offset_never_collapse() -> None:
    notes = [NoteEvent(72, "violin_1", 1.0, 1.08, 0.08)]
    aligned = align_event_boundaries(notes, tolerance_beat=0.1, max_cluster_span_beat=0.2)
    assert aligned[0].offset_beat > aligned[0].onset_beat


def test_boundaries_outside_span_remain_separate() -> None:
    notes = [
        NoteEvent(72, "violin_1", 1.0, 2.0, 1.0),
        NoteEvent(67, "violin_2", 1.25, 2.25, 1.0),
    ]
    aligned = align_event_boundaries(notes, tolerance_beat=0.1, max_cluster_span_beat=0.2)
    assert aligned[0].onset_beat != aligned[1].onset_beat
