from a2s.data.kern_parser import kern_pitch_to_midi, parse_kern_notes, reciprocal_to_beats


def test_kern_pitch_and_duration():
    assert kern_pitch_to_midi("4c#") == (61, "C#4")
    assert float(reciprocal_to_beats("8.")) == 0.75


def test_parse_four_spines():
    text = """**kern\t**kern\t**kern\t**kern
*Icello\t*Iviola\t*Ivioln\t*Ivioln
*M4/4\t*M4/4\t*M4/4\t*M4/4
1C\t1c\t1e\t1g
=2\t=2\t=2\t=2
*-\t*-\t*-\t*-
"""
    sequence = parse_kern_notes(text, "test")
    assert len(sequence.notes) == 4
    assert {note.voice for note in sequence.notes} == {"cello", "viola", "violin_1", "violin_2"}
