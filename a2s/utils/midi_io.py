from __future__ import annotations

from pathlib import Path

from a2s.data.note_event import NoteEventSequence


def events_to_midi(sequence: NoteEventSequence, path: str | Path) -> None:
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(initial_tempo=sequence.tempo_bpm or 120.0)
    for voice in sequence.staves:
        instrument = pretty_midi.Instrument(program=40, name=voice)
        for note in sequence.notes:
            if note.voice != voice or note.onset_sec is None or note.offset_sec is None:
                continue
            instrument.notes.append(pretty_midi.Note(note.velocity or 80, note.pitch, note.onset_sec, note.offset_sec))
        midi.instruments.append(instrument)
    midi.write(str(path))


def events_to_midi_beats(sequence: NoteEventSequence, path: str | Path) -> None:
    from a2s.stage2_score.midi_tokenizer import beats_to_seconds

    events_to_midi(beats_to_seconds(sequence, sequence.tempo_bpm), path)
