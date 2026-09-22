from __future__ import annotations

from dataclasses import replace

from a2s.data.note_event import NoteEventSequence

from .voice_assignment import assign_voices


def postprocess(
    sequence: NoteEventSequence,
    assign_unknown_voices: bool = True,
    min_confidence: float = 0.0,
    transpose_semitones: int = 0,
) -> NoteEventSequence:
    def positive_duration(note):
        if note.onset_sec is not None and note.offset_sec is not None:
            return note.offset_sec > note.onset_sec
        if note.onset_beat is not None and note.offset_beat is not None:
            return note.offset_beat > note.onset_beat
        return False

    sequence.notes = [note for note in sequence.notes if note.confidence >= min_confidence and positive_duration(note)]
    if transpose_semitones:
        shifted = []
        for note in sequence.notes:
            pitch = int(note.pitch) + int(transpose_semitones)
            if 0 <= pitch <= 127:
                shifted.append(replace(note, pitch=pitch))
        sequence.notes = shifted
        sequence.metadata["postprocess_transpose_semitones"] = int(transpose_semitones)
    if assign_unknown_voices and any(note.voice == "unknown" for note in sequence.notes):
        sequence.notes = assign_voices(sequence.notes)
    return sequence
