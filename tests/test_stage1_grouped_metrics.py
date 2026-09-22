from a2s.data.note_event import NoteEvent
from a2s.evaluation.note_f1 import note_match_counts


def test_per_sample_matching_prevents_cross_sample_matches() -> None:
    # These notes would match if two samples were incorrectly concatenated.
    sample_a = note_match_counts([NoteEvent(60, onset_sec=0, offset_sec=1)], [])
    sample_b = note_match_counts([], [NoteEvent(60, onset_sec=0, offset_sec=1)])
    assert sample_a["note_matches"] + sample_b["note_matches"] == 0
