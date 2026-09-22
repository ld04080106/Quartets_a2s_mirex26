from __future__ import annotations

from a2s.data.note_event import NoteEvent, NoteEventSequence
from a2s.stage1_amt.yourmt3_dataset import instrument_to_gm_program, oracle_notes_in_seconds


def test_instrument_program_mapping() -> None:
    assert instrument_to_gm_program("violn") == 40
    assert instrument_to_gm_program("viola") == 41
    assert instrument_to_gm_program("cello") == 42
    assert instrument_to_gm_program("flt") == 73


def test_oracle_beats_are_converted_to_seconds() -> None:
    sequence = NoteEventSequence(sample_id="x", notes=[
        NoteEvent(pitch=72, voice="violin_1", onset_beat=1.0, offset_beat=2.0, duration_beat=1.0)
    ])
    notes = oracle_notes_in_seconds(sequence, 120.0, {"violin_1": 73})
    assert len(notes) == 1
    assert notes[0]["program"] == 73
    assert notes[0]["onset"] == 0.5
    assert notes[0]["offset"] == 1.0

