from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from a2s.data.note_event import NoteEvent, NoteEventSequence, VOICES
from a2s.stage2_score.tie_merger import merge_tied_notes


def _q(value: float | None, subdivisions_per_beat: int) -> str:
    number = 0.0 if value is None else float(value)
    step = Fraction(round(number * subdivisions_per_beat), subdivisions_per_beat)
    return f"{float(step):.6f}".rstrip("0").rstrip(".")


def _seconds_to_beats(value_sec: float | None, tempo_bpm: float | None) -> float | None:
    if value_sec is None:
        return None
    return max(0.0, float(value_sec) * float(tempo_bpm or 120.0) / 60.0)


def _onset(note: NoteEvent, tempo_bpm: float | None) -> float | None:
    if note.onset_beat is not None:
        return float(note.onset_beat)
    return _seconds_to_beats(note.onset_sec, tempo_bpm)


def _duration(note: NoteEvent, tempo_bpm: float | None = None) -> float:
    if note.duration_beat is not None:
        return float(note.duration_beat)
    if note.onset_beat is not None and note.offset_beat is not None:
        return float(note.offset_beat - note.onset_beat)
    if note.onset_sec is not None and note.offset_sec is not None:
        return float(note.offset_sec - note.onset_sec) * float(tempo_bpm or 120.0) / 60.0
    return 0.25


def sequence_to_midi_tokens(
    sequence: NoteEventSequence,
    subdivisions_per_beat: int = 24,
    include_confidence: bool = True,
) -> list[str]:
    """Serialize a voice-aware MIDI/note-event sequence to robust tokens.

    This is deliberately not raw SMF bytes.  It is the normalized symbolic MIDI
    layer used by Stage2: pitch, voice/program, quantized onset, duration,
    velocity, confidence.  It can be exported to a real `.mid` file after model
    correction.
    """

    tokens = [
        "<BOS_MIDI>",
        f"METER={sequence.meter or '4/4'}",
        f"KEY={str(sequence.key or 'unknown').replace(' ', '_')}",
    ]
    tempo = sequence.tempo_bpm or 120.0
    def sort_key(note: NoteEvent) -> tuple[float, str, int]:
        return (_onset(note, tempo) or 0.0, note.voice, note.pitch)

    for note in sorted(merge_tied_notes(sequence.notes), key=sort_key):
        onset = _onset(note, tempo)
        if onset is None:
            continue
        duration = max(_duration(note, tempo), 1.0 / subdivisions_per_beat)
        velocity = int(note.velocity if note.velocity is not None else 80)
        confidence = round(float(note.confidence if include_confidence else 1.0) * 10) / 10
        tokens.extend((
            "<N>",
            f"V={note.voice}",
            f"P={int(note.pitch)}",
            f"ON={_q(onset, subdivisions_per_beat)}",
            f"DUR={_q(duration, subdivisions_per_beat)}",
            f"VEL={max(1, min(127, velocity))}",
            f"CONF={confidence:.1f}",
            "</N>",
        ))
    tokens.append("<EOS_MIDI>")
    return tokens


def midi_tokens_to_sequence(
    tokens: list[str],
    sample_id: str,
    meter: str | None = None,
    key: str | None = None,
    tempo_bpm: float | None = 120.0,
) -> NoteEventSequence:
    notes: list[NoteEvent] = []
    parsed_meter = meter
    parsed_key = key
    current: dict[str, str] | None = None
    for token in tokens:
        if token.startswith("METER="):
            parsed_meter = token.split("=", 1)[1].replace("_", " ")
            continue
        if token.startswith("KEY="):
            parsed_key = token.split("=", 1)[1].replace("_", " ")
            continue
        if token == "<N>":
            current = {}
            continue
        if token == "</N>":
            if not current:
                continue
            try:
                voice = current.get("V", "unknown")
                pitch = current["P"]
                onset = current["ON"]
                duration = current["DUR"]
                velocity = current.get("VEL", "80")
                confidence = current.get("CONF", "1.0")
            except KeyError:
                current = None
                continue
            if voice not in VOICES:
                voice = "unknown"
            onset_f = float(onset)
            duration_f = max(float(duration), 1 / 96)
            notes.append(NoteEvent(
                pitch=int(pitch),
                voice=voice,
                onset_beat=onset_f,
                duration_beat=duration_f,
                offset_beat=onset_f + duration_f,
                velocity=int(float(velocity)),
                confidence=max(0.0, min(1.0, float(confidence))),
            ))
            current = None
            continue
        if current is None or "=" not in token:
            continue
        key, value = token.split("=", 1)
        current[key] = value
    notes.sort(key=lambda n: (n.onset_beat or 0.0, n.voice, n.pitch))
    return NoteEventSequence(
        sample_id=sample_id,
        notes=notes,
        meter=parsed_meter,
        key=parsed_key,
        tempo_bpm=tempo_bpm,
        staves=list(VOICES),
        source_model="stage2_midi_neural",
    )


def beats_to_seconds(sequence: NoteEventSequence, tempo_bpm: float | None = None) -> NoteEventSequence:
    bpm = float(tempo_bpm or sequence.tempo_bpm or 120.0)
    seconds_per_beat = 60.0 / bpm
    return NoteEventSequence(
        sample_id=sequence.sample_id,
        meter=sequence.meter,
        key=sequence.key,
        tempo_bpm=bpm,
        staves=sequence.staves,
        source_model=sequence.source_model,
        num_measures=sequence.num_measures,
        composer=sequence.composer,
        metadata=dict(sequence.metadata),
        notes=[
            replace(
                note,
                onset_sec=(note.onset_beat or 0.0) * seconds_per_beat,
                offset_sec=(note.offset_beat or ((note.onset_beat or 0.0) + _duration(note))) * seconds_per_beat,
            )
            for note in sequence.notes
        ],
    )
