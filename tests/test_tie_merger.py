from a2s.data.note_event import NoteEvent
from a2s.stage2_score.tie_merger import merge_tied_notes


def test_merge_simple_three_part_tie() -> None:
    notes = [
        NoteEvent(72, "violin_1", 0.0, 1.0, 1.0, original_token="4cc[", is_tied_start=True),
        NoteEvent(72, "violin_1", 1.0, 2.0, 1.0, original_token="4cc_"),
        NoteEvent(72, "violin_1", 2.0, 3.0, 1.0, original_token="4cc]", is_tied_stop=True),
    ]
    result = merge_tied_notes(notes)
    assert len(result) == 1
    assert result[0].onset_beat == 0.0
    assert result[0].offset_beat == 3.0
    assert result[0].duration_beat == 3.0


def test_do_not_merge_broken_tie_chain() -> None:
    notes = [
        NoteEvent(72, "violin_1", 0.0, 1.0, 1.0, original_token="4cc[", is_tied_start=True),
        NoteEvent(72, "violin_1", 1.5, 2.5, 1.0, original_token="4cc]", is_tied_stop=True),
    ]
    assert len(merge_tied_notes(notes)) == 2
