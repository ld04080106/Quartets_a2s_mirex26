from a2s.data.note_event import NoteEvent, VOICES
from a2s.stage1_amt.voice_assignment import assign_voices


def test_dense_onset_group_assigns_every_note() -> None:
    notes = [
        NoteEvent(pitch, "unknown", onset_sec=0.0, offset_sec=1.0)
        for pitch in (84, 81, 76, 72, 69, 64, 55, 48)
    ]
    assigned = assign_voices(notes)
    assert len(assigned) == len(notes)
    assert all(note.voice in VOICES for note in assigned)
    assert set(note.voice for note in assigned) == set(VOICES)
