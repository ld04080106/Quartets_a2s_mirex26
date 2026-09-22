from a2s.data.note_event import NoteEvent, NoteEventSequence


def test_note_event_schema_roundtrip():
    sequence = NoteEventSequence("x", [NoteEvent(74, "violin_1", onset_beat=0, offset_beat=1)])
    restored = NoteEventSequence.from_dict(sequence.to_dict())
    assert restored.notes[0].duration_beat == 1


def test_note_event_rejects_bad_pitch():
    try:
        NoteEventSequence("x", [NoteEvent(200)]).to_dict()
    except ValueError:
        return
    raise AssertionError("out-of-range MIDI pitch must raise ValueError")
