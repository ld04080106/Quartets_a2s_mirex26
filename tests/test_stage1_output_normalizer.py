from a2s.stage1_amt.output_normalizer import instrument_to_voice, sequence_from_json


def test_note_sequence_json_is_normalized() -> None:
    payload = {"notes": [{
        "pitch": 74, "startTime": 1.25, "endTime": 1.75,
        "velocity": 81, "program": 41,
    }]}
    sequence = sequence_from_json(payload, "sample", "mt3_test")
    assert sequence.sample_id == "sample"
    assert sequence.notes[0].onset_sec == 1.25
    assert sequence.notes[0].offset_sec == 1.75
    assert sequence.notes[0].voice == "viola"


def test_ambiguous_violin_program_stays_unknown() -> None:
    assert instrument_to_voice("Violin", 40) == "unknown"
    assert instrument_to_voice("Violin 1", 40) == "violin_1"
    assert instrument_to_voice("Cello", 42) == "cello"
